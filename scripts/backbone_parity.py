#!/usr/bin/env python
"""The in-repo Qwen2 forward against `transformers`, on the real weights.

    python scripts/backbone_parity.py --backbone qwen2.5-1.5b --out reports/backbone/

`trigon.backends.qwen_readout` writes the Qwen2 forward itself so the block
mask it is handed is the one that runs. That only buys anything if the
forward is Qwen2's, so this checks it against the reference implementation
under the one mask the reference understands -- causal, positions 0..T-1 --
on text from the corpora this project serves.

Checked on the final hidden states and on the backbone's own next-token
prediction through the tied embedding, in float32 on CPU: agreement to
float32 rounding means the conversion starts from the pretrained model and
not from something that merely loads its weights.

The two models are never resident together -- 1.5B parameters in float32 is
6 GB -- so the in-repo one runs first and is freed.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

TEXTS = [
    "I was charged twice for the same transaction this morning, can you refund one?",
    "The app will not let me log in after I changed my phone number last week.",
    "Write a short poem about the sea.\n\nThe sea is wide and blue and deep, "
    "it rocks the sailors off to sleep.",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="qwen2.5-1.5b")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import torch

    from trigon.backends.hub import BACKBONES, fetch
    from trigon.backends.qwen_readout import QwenReadoutBackend

    torch.set_num_threads(4)
    backend = QwenReadoutBackend.from_backbone(args.backbone, device="cpu")
    model = backend.model.eval()
    encoded = [backend.tokenizer.encode(t) for t in TEXTS]

    ours = []
    with torch.no_grad():
        for ids in encoded:
            n = len(ids)
            embeddings = model.embed(torch.tensor(ids))[None]
            causal = torch.ones(n, n, dtype=torch.bool).tril()
            positions = torch.arange(n)
            ours.append(model(embeddings, causal, positions, torch.zeros(n, dtype=torch.long))[0])
    embedding = model.embed.weight.detach().clone()
    del backend, model
    gc.collect()

    from transformers import AutoModel

    backbone = BACKBONES[args.backbone]
    reference = AutoModel.from_pretrained(
        str(fetch(backbone, "config.json").parent), torch_dtype=torch.float32
    ).eval()
    rows = []
    with torch.no_grad():
        for text, ids, mine in zip(TEXTS, encoded, ours, strict=True):
            theirs = reference(torch.tensor([ids])).last_hidden_state[0]
            diff = (mine - theirs).abs()
            agree = (mine @ embedding.T).argmax(-1).eq((theirs @ embedding.T).argmax(-1))
            rows.append(
                {
                    "text": text[:60],
                    "tokens": len(ids),
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "hidden_scale": float(theirs.abs().mean()),
                    "next_token_agreement": float(agree.float().mean()),
                }
            )
    for row in rows:
        print(
            f"{row['tokens']:>4} tokens  max |diff| {row['max_abs_diff']:.2e}  "
            f"mean {row['mean_abs_diff']:.2e} (scale {row['hidden_scale']:.2f})  "
            f"next-token agreement {row['next_token_agreement']:.4f}"
        )
    result = {"backbone": args.backbone, "revision": backbone.revision, "rows": rows}
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.backbone}-parity.json").write_text(json.dumps(result, indent=2) + "\n")
    worst = max(r["max_abs_diff"] / max(r["hidden_scale"], 1e-9) for r in rows)
    agreement = min(r["next_token_agreement"] for r in rows)
    ok = worst < 1e-3 and agreement == 1.0
    print("PARITY" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
