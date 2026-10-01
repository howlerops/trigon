#!/usr/bin/env python
"""Train one model on many tasks, and keep the tasks it will be judged on out.

    python scripts/train_mix.py --backbone qwen2.5-0.5b --device mps \\
        --out runs/mix-05b-s0 --epochs 2 --seed 0

`scripts/train_corpus.py` trains one corpus under one declared option set, and
that is how the certified Banking77 model came to learn a set instead of
reading one (`trigon.evals.generality`). This trains across every corpus the
licence audit clears for training, with every Choice's options redrawn per
case per epoch, and writes a bundle -- ``adapter.pt``, ``temperatures.json``,
``isotonic.json``, ``mix.json`` -- that `scripts/generality.py --bundle` reads.

**What is held out, and why each one.** CLINC150 is green and trainable and
is held out by decision: it is the suite's unseen intent task. BoolQ and Circa
are share-alike and could never train. Four whole teacher domains are held
out, so a domain-level transfer can be read off the teacher's test split. The
evaluation is `scripts/generality.py`, not this script: a mix has no single
test split, and the question it is trained to answer is the suite's.

**Calibration is fitted across tasks.** The calibration split is carved from
every real corpus in the mix -- never from teacher labels, which
`_fit_calibration` refuses -- with its Choices reshaped the way training
reshapes them, so the calibrator is fitted on the distribution of option sets
a caller will send rather than on one set. `mix.json` records exactly which
corpora, how many cases of each and what the calibrator was fitted on, which
is what `/healthz` should one day report instead of a bare ``calibrated``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from train_corpus import resolve_device  # noqa: E402

from trigon.cli import _fit_calibration  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals.corpora import corpus, load  # noqa: E402
from trigon.evals.schema_shift import Reshape, reshape_all  # noqa: E402
from trigon.schema import OptionScoring  # noqa: E402

#: corpus -> training cases drawn from its train split. Capped so no one task
#: is most of an epoch: Banking77 alone would otherwise be a third of it.
MIX: dict[str, int] = {
    "banking77": 4000,
    "goemotions": 3000,
    "helpsteer2": 1500,
    "measuring_hate_speech": 1500,
    "hatexplain": 2500,
    "teacher-workflows": 4000,
}

#: Whole domains of the teacher stream no epoch sees.
HELDOUT_DOMAINS = ("clinical_intake", "contract_review", "code_review", "real_estate")

#: Held out by decision or by licence; the mix refuses to load them.
NEVER = ("clinc150", "boolq", "circa")


def _draw(name: str, n: int, seed: int) -> list:
    cases = load(name, "train", purpose="train")
    if name in ("teacher-workflows", "teacher-local"):
        cases = [c for c in cases if c.domain not in HELDOUT_DOMAINS]
    random.Random(f"mix:{name}:{seed}").shuffle(cases)
    return cases[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="qwen2.5-0.5b")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every MIX count")
    parser.add_argument("--calibration-per-corpus", type=int, default=600)
    parser.add_argument("--reshape-max", type=int, default=77)
    parser.add_argument("--reshape-rename", type=float, default=0.3)
    parser.add_argument("--reshape-criteria-only", type=float, default=0.1)
    parser.add_argument(
        "--reshape-crossover-fraction",
        type=float,
        default=0.0,
        help="share of reshapes above --option-crossover, so both Choice heads train",
    )
    parser.add_argument(
        "--option-crossover",
        type=int,
        default=256,
        help=(
            "above this many options a Choice is scored by the dot-product head; saved "
            "with the weights. The per-option head learns option names quickly and the "
            "dot-product head barely moved in three epochs at 0.5B, so the default puts "
            "every corpus in the mix -- and the incumbent's 50-option cap -- under it"
        ),
    )
    parser.add_argument("--max-batch-cells", type=int, default=20_000_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    import torch

    from trigon.backends.qwen_readout import QwenReadoutBackend
    from trigon.training import TrainingConfig
    from trigon.training import train as run_training

    for name in MIX:
        if name in NEVER or not corpus(name).permits("train"):
            raise SystemExit(f"{name} is held out or may not train; remove it from MIX")

    args.out.mkdir(parents=True, exist_ok=True)
    device, hardware = resolve_device(args.device)
    torch.set_num_threads(4)

    reshape = Reshape(
        min_options=2,
        max_options=args.reshape_max,
        shuffle=True,
        rename=args.reshape_rename,
        criteria_only=args.reshape_criteria_only,
        crossover_fraction=args.reshape_crossover_fraction,
        crossover=args.option_crossover,
    )

    train, calibration, drawn = [], [], {}
    for name, n in MIX.items():
        spec = corpus(name)
        want = int(n * args.scale)
        # Calibration comes only from corpora whose labels are evidence.
        extra = args.calibration_per_corpus if spec.calibration_evidence else 0
        cases = _draw(name, want + extra, args.seed)
        calibration += cases[:extra]
        train += cases[extra:]
        drawn[name] = {"train": len(cases) - extra, "calibration": min(extra, len(cases))}
        print(f"mix: {name} {drawn[name]}", file=sys.stderr, flush=True)
    calibration = reshape_all(calibration, reshape, seed=args.seed + 11)

    from trigon.backends.torch_readout import ReadoutConfig

    backend = QwenReadoutBackend.from_backbone(
        args.backbone,
        device=device,
        seed=args.seed,
        lora_rank=args.lora_rank,
        config=ReadoutConfig(option_crossover=args.option_crossover),
    )
    backend.model.checkpointing = True
    backend.to(device)
    compiler = backend.make_compiler(option_scoring=OptionScoring("auto"))
    print(
        f"mix: {len(train)} train, {len(calibration)} calibration on {device} ({hardware})",
        file=sys.stderr,
        flush=True,
    )

    started = time.perf_counter()
    report = run_training(
        backend,
        train,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            accumulate=args.accumulate,
            seed=args.seed,
            log_every=args.log_every,
            max_batch_cells=args.max_batch_cells or None,
            resume_path=str(args.out / "resume.pt"),
            rationale_weight=0.0,
            reshape=reshape,
        ),
        compiler=compiler,
    )
    trained_seconds = time.perf_counter() - started
    backend.save(args.out / "adapter.pt")

    backend.cache_prefixes = True
    scaler, isotonic = _fit_calibration(Engine(backend, compiler=compiler), calibration)
    scaler.save(args.out / "temperatures.json")
    if isotonic.knots:
        isotonic.save(args.out / "isotonic.json")

    (args.out / "mix.json").write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "option_crossover": args.option_crossover,
                "model_version": backend.model_version,
                "device": f"{device} ({hardware})",
                "seed": args.seed,
                "epochs": args.epochs,
                "lr": args.lr,
                "train_seconds": round(trained_seconds),
                "corpora": drawn,
                "held_out": {"corpora": list(NEVER), "teacher_domains": list(HELDOUT_DOMAINS)},
                "reshape": reshape.__dict__ | {"paraphrases": None},
                "calibrated_on": {
                    "cases": len(calibration),
                    "corpora": [n for n, d in drawn.items() if d["calibration"]],
                    "reshaped": True,
                },
                "training": report.to_dict(),
                "argv": sys.argv[1:],
            },
            indent=2,
        )
    )
    print(f"mix: bundle written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
