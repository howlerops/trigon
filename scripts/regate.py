#!/usr/bin/env python
"""Refit the calibration layer on saved checkpoints and re-run the gates.

    python scripts/regate.py reports/iso/iso-seed0.pt reports/iso/iso-seed1.pt
    python scripts/regate.py reports/iso/*.pt --out reports/iso

Training is the expensive part of `trigon train` and the calibration layer is
the part that changes. Refitting on a saved checkpoint takes a minute where
retraining takes twenty, which is the difference between measuring a
calibration change on four seeds and guessing at it.

It is also the honest way to publish a calibration change. A report written by
a training run reflects the calibration rule that ran *then*; re-gating writes
one that reflects the rule now, against the same weights, so a table of
before-and-after numbers is two measurements rather than one measurement and
one memory.

The seed is recovered from the checkpoint's filename (``...-seed<N>.pt``),
because the splits are derived from it and calibrating on the wrong split
would fit on data the model trained on -- the defect `docs/decisions.md`
records under "The temperature was fitted on the split the model trained on".
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def _seed_of(path: pathlib.Path) -> int:
    match = re.search(r"seed(\d+)", path.stem)
    if not match:
        raise SystemExit(
            f"cannot tell which seed {path.name} was trained with; the calibration "
            "split is derived from it, so guessing would fit on the training data"
        )
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out", default=None, help="write a report per checkpoint here")
    parser.add_argument("-n", type=int, default=8000, help="training-split size of the run")
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument("--eval-n", type=int, default=6000)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--floor-trials", type=int, default=40)
    args = parser.parse_args()

    from trigon.backends.torch_readout import TorchReadoutBackend
    from trigon.cli import _fit_calibration, training_splits
    from trigon.engine import Engine
    from trigon.evals import (
        blocking,
        check_gates,
        render_json,
        render_markdown,
        run_calibration_suite,
    )

    print("| Seed | ECE | Adaptive ECE | Lift over baseline | Certified | Blocked on |")
    print("| ---: | ---: | ---: | ---: | --- | --- |")
    worst = 0
    for raw in args.checkpoints:
        path = pathlib.Path(raw)
        seed = _seed_of(path)
        backend = TorchReadoutBackend.load(path)
        compiler = backend.make_compiler()
        _, calibration_cases, eval_cases = training_splits(
            n=args.n,
            calibration_n=args.calibration_n,
            eval_n=args.eval_n,
            seed=seed,
            noise=args.noise,
        )
        plain = Engine(backend, compiler=compiler)
        before, _ = run_calibration_suite(
            plain,
            eval_cases,
            suite="calibration/uncalibrated",
            floor_trials=args.floor_trials,
        )
        scaler, isotonic = _fit_calibration(plain, calibration_cases)
        after, slices = run_calibration_suite(
            Engine(backend, compiler=compiler, scaler=scaler, isotonic=isotonic),
            eval_cases,
            suite="calibration/calibrated",
            floor_trials=args.floor_trials,
        )
        # Q16: the per-question gates block on a pretrained backbone.
        backbone = type(backend).__name__ != "TorchReadoutBackend"
        gates = check_gates(after, slices=slices, require_per_question=backbone)
        by_name = {g.name: g for g in gates}
        blocked = [g.name for g in blocking(gates) if not g.passed]
        worst += bool(blocked)

        if args.out:
            out = pathlib.Path(args.out) / f"{path.stem}-regated.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            header = (
                f"# Re-gated: `{path.name}`\n\n"
                f"Calibration refitted on the seed-{seed} calibration split and the gates "
                f"re-run, against weights trained earlier. Reproduce with:\n\n"
                f"```bash\npython scripts/regate.py {raw} --out {args.out} "
                f"-n {args.n} --calibration-n {args.calibration_n} --eval-n {args.eval_n} "
                f"--noise {args.noise} --floor-trials {args.floor_trials}\n```\n\n"
            )
            out.write_text(header + render_markdown([before, after], gates, slices, gated=after))
            out.with_suffix(".json").write_text(
                render_json([before, after], gates, slices, gated=after)
            )
            scaler.save(out.with_name(f"{out.stem}-temperatures.json"))
            if isotonic.knots:
                isotonic.save(out.with_name(f"{out.stem}-isotonic.json"))

        print(
            f"| {seed} | {by_name['workhorse_ece'].value:.4f} | "
            f"{by_name['workhorse_adaptive_ece'].value:.4f} | "
            f"{by_name['accuracy_over_baseline'].value:+.4f} | "
            f"{'**yes**' if not blocked else 'no'} | {', '.join(blocked) or '—'} |"
        )
    # Non-zero if any checkpoint failed a blocking gate, so this can gate CI
    # exactly as `trigon train` does.
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
