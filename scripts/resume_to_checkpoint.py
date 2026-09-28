"""Turn a training run's resume file into a checkpoint `TorchReadoutBackend.load` reads.

A Modal run launched without ``--save-model`` leaves only its resume file on
the Volume -- the four teacher-trained students did. The file carries the
kept epoch's trainables (``best``, the epoch whose validation loss was lowest)
but records nothing about what they were trained over: no backbone, no LoRA
rank, no tokenizer. So this script is *told* them, from the run's own record
(``<prefix>-modal-run.json`` lists its flags), and writes a checkpoint that
records them the way ``save`` always does. From there ``--init-weights``
checks them like any other.

    python scripts/resume_to_checkpoint.py seed0.resume.pt --out seed0.pt \\
        --backbone qwen2.5-1.5b --lora-rank 16 --expect-epoch 2

``--expect-epoch`` and ``--expect-validation`` are the provenance check: the
run's ``-training.json`` names the epoch it kept and that epoch's validation
loss, and a resume file from some other run will not match both.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def convert(args: argparse.Namespace) -> str:
    import torch

    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend

    saved = torch.load(args.resume, map_location="cpu", weights_only=False)
    best = saved["best"]
    if best is not None:
        validation, epoch, weights = best
    else:
        validation, epoch, weights = None, len(saved["epochs"]), saved["weights"]
    if args.expect_epoch is not None and epoch != args.expect_epoch:
        raise SystemExit(f"{args.resume} kept epoch {epoch}, not {args.expect_epoch}")
    if args.expect_validation is not None and not (
        validation is not None and math.isclose(validation, args.expect_validation, rel_tol=1e-9)
    ):
        raise SystemExit(
            f"{args.resume} kept validation loss {validation}, not {args.expect_validation}"
        )

    if args.backbone:
        from trigon.backends.qwen_readout import QwenReadoutBackend

        backend = QwenReadoutBackend.from_backbone(
            args.backbone, device="cpu", lora_rank=args.lora_rank
        )
    else:
        backend = TorchReadoutBackend(
            config=ReadoutConfig(d_model=args.d_model, n_layers=args.layers), seed=0
        )
    model = backend.model
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(weights) != trainable:
        extra, absent = sorted(set(weights) - trainable), sorted(trainable - set(weights))
        raise SystemExit(
            f"{args.resume} does not fit this model: {len(extra)} unexpected "
            f"({extra[:3]}), {len(absent)} missing ({absent[:3]})"
        )
    # Shapes are checked here: a wrong LoRA rank or width raises.
    model.load_state_dict(weights, strict=False)
    model.eval()
    version = backend.stamp_version()
    backend.save(pathlib.Path(args.out))
    shown = f"{validation:.6f}" if validation is not None else "n/a"
    print(f"{args.resume}: epoch {epoch}, validation {shown} -> {args.out} ({version})")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--expect-epoch", type=int, default=None)
    parser.add_argument("--expect-validation", type=float, default=None)
    convert(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
