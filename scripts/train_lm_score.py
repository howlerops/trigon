#!/usr/bin/env python
"""Train the LM-score readout: LoRA on a causal LM, answers scored as its own tokens.

    python scripts/train_lm_score.py --model Qwen/Qwen3-0.6B --device mps \\
        --extra wanli=4000 --extra mind2web-train=4000 --out runs/lms-q3-06b-s0

`scripts/train_mix.py` trains a readout head from scratch through a mask the
backbone never saw. This keeps the backbone's language-model head and trains
around it (`trigon.backends.lm_score`): each answer's score is
``w * log p(answer | prompt) + residual``, ``w`` starting at 1 and the residual
at zero, so step zero is the zero-shot scorer and training moves away from it
only as far as the data asks. Every candidate is scored in one packed forward
pass, the same function serving uses.

One epoch by default. Public evidence for this design is that more than one
epoch at this scale lowers held-out accuracy, and the mix, held-out corpora and
calibration split are `train_mix.py`'s, so the two are comparable on
`scripts/generality.py`, `scripts/webact.py` and `scripts/jevbench.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from train_mix import MIX, NEVER, _draw  # noqa: E402

from trigon.evals.corpora import corpus  # noqa: E402
from trigon.evals.schema_shift import Reshape, reshape_all  # noqa: E402
from trigon.schema import render_state  # noqa: E402
from trigon.types import NoulQuestion  # noqa: E402


def examples(cases) -> list[tuple[str, list[str], str, object]]:
    """``(prompt, candidates, kind, target)`` per labelled question.

    ``target`` is a label index or a distribution for a Choice or Score, and a
    probability of yes for a Noul.
    """
    from trigon.backends.lm_score import prompt_for

    out = []
    for case in cases:
        state = render_state(case.request.state)
        for qid, question in case.request.questions.items():
            expected = case.expected.get(qid)
            if expected is None:
                continue
            prompt, candidates = prompt_for(state, question)
            if isinstance(question, NoulQuestion):
                if expected.probability is None:
                    continue
                out.append((prompt, candidates, "noul", float(expected.probability)))
            elif expected.distribution is not None:
                out.append((prompt, candidates, "choice", list(expected.distribution)))
            elif expected.label is not None:
                out.append((prompt, candidates, "choice", int(expected.label)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--readout-lr", type=float, default=1e-3)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--extra", action="append", default=[], metavar="CORPUS=N")
    parser.add_argument("--calibration-per-corpus", type=int, default=600)
    parser.add_argument("--reshape-max", type=int, default=77)
    parser.add_argument(
        "--checkpointing", action="store_true", help="trade compute for memory on larger models"
    )
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--name", default=None, help="build name; defaults to the out dir")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trigon.backends.lm_score import Readout, add_lora, packed_scores

    mix = dict(MIX)
    for item in args.extra:
        name, _, count = item.partition("=")
        mix[name] = int(count)
    for name in mix:
        if name in NEVER or not corpus(name).permits("train"):
            raise SystemExit(f"{name} is held out or may not train")

    reshape = Reshape(
        min_options=2, max_options=args.reshape_max, shuffle=True, rename=0.3, criteria_only=0.1
    )
    train, calibration, drawn = [], [], {}
    for name, n in mix.items():
        spec = corpus(name)
        want = int(n * args.scale)
        extra = args.calibration_per_corpus if spec.calibration_evidence else 0
        cases = _draw(name, want + extra, args.seed)
        calibration += cases[:extra]
        train += cases[extra:]
        drawn[name] = {"train": len(cases) - extra, "calibration": min(extra, len(cases))}
        print(f"mix: {name} {drawn[name]}", file=sys.stderr, flush=True)
    calibration = reshape_all(calibration, reshape, seed=args.seed + 11)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # float32: Apple's M1 has no bfloat16 arithmetic, and the frozen weights
    # are 2.4 GB at this size.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    lora = add_lora(model, args.lora_rank, args.lora_alpha)
    model.to(device)
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    readout = Readout(model.config.hidden_size, device)
    optimiser = torch.optim.AdamW(
        [
            {"params": list(lora.values()), "lr": args.lr},
            {"params": readout.parameters(), "lr": args.readout_lr, "weight_decay": 0.0},
        ]
    )
    n_trainable = sum(p.numel() for p in lora.values())
    print(f"trainable LoRA parameters: {n_trainable:,}", file=sys.stderr, flush=True)

    started = time.perf_counter()
    step, running = 0, []
    for epoch in range(args.epochs):
        reshaped = reshape_all(train, reshape, seed=args.seed * 1000 + epoch)
        batch = examples(reshaped)
        random.Random(f"order:{args.seed}:{epoch}").shuffle(batch)
        total_steps = len(batch) * args.epochs
        model.train()
        for i, (prompt, candidates, kind, target) in enumerate(batch):
            lr_scale = 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
            for group, base in zip(optimiser.param_groups, (args.lr, args.readout_lr), strict=True):
                group["lr"] = base * min(1.0, (step + 1) / 200) * lr_scale
            prompt_ids = tokenizer(prompt).input_ids[-args.max_prompt_tokens :]
            encoded = [tokenizer(c, add_special_tokens=False).input_ids for c in candidates]
            sums, hidden = packed_scores(model, prompt_ids, encoded, device)
            scores = readout(sums, hidden)
            if kind == "noul":
                logit = scores[1] - scores[0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, torch.tensor(target, device=logit.device)
                )
            else:
                logp = torch.log_softmax(scores, dim=-1)
                if isinstance(target, int):
                    loss = -logp[target]
                else:
                    loss = -(torch.tensor(target, device=logp.device) * logp).sum()
            (loss / args.accumulate).backward()
            running.append(loss.item())
            step += 1
            if step % args.accumulate == 0 or i == len(batch) - 1:
                torch.nn.utils.clip_grad_norm_(list(lora.values()), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
            if step % args.log_every == 0:
                rate = step / (time.perf_counter() - started)
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {sum(running) / len(running):.4f} w {float(readout.w):.3f} "
                    f"{rate:.2f} q/s",
                    file=sys.stderr,
                    flush=True,
                )
                running = []
    train_seconds = time.perf_counter() - started

    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or args.out.name
    adapter = args.out / "adapter.pt"
    torch.save(
        {
            "kind": "lm-score",
            "base": args.model,
            "name": name,
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "lora": {k: v.detach().cpu() for k, v in lora.items()},
            "readout": readout.state_dict(),
        },
        adapter,
    )
    del model, optimiser
    if device.type == "mps":
        torch.mps.empty_cache()

    from trigon.backends.lm_score import LMScoreBackend
    from trigon.cli import _fit_calibration
    from trigon.engine import Engine

    backend = LMScoreBackend(args.model, device=args.device, adapter=str(adapter))
    scaler, isotonic = _fit_calibration(Engine(backend), calibration)
    scaler.save(args.out / "temperatures.json")
    if isotonic.knots:
        isotonic.save(args.out / "isotonic.json")
    (args.out / "mix.json").write_text(
        json.dumps(
            {
                "kind": "lm-score",
                "base": args.model,
                "model_version": backend.model_version,
                "lora_rank": args.lora_rank,
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed,
                "device": args.device,
                "train_seconds": train_seconds,
                "w": float(backend.readout.w),
                "corpora": drawn,
                "held_out": {"corpora": list(NEVER)},
                "calibrated_on": {
                    "cases": len(calibration),
                    "corpora": [n for n, d in drawn.items() if d["calibration"]],
                },
                "argv": sys.argv[1:],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} in {train_seconds / 3600:.2f} h", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
