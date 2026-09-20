#!/usr/bin/env python
"""Serve a handful of cases through a checkpoint and print what it answers.

`reports/README.md` shows what the certified model actually says, which is the
only part of that report a reader can check against their intuition rather than
against a threshold. It was produced ad hoc once; this makes it reproducible,
so the block can be regenerated when the reference run is.

    python scripts/served_answers.py reports/reference-run.pt
    python scripts/served_answers.py reports/reference-run.pt --cases 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.engine import Engine  # noqa: E402
from trigon.evals.datasets import synthetic_outcome_cases  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="a checkpoint written by 'trigon train'")
    parser.add_argument("--temperatures", default=None, help="a fitted temperatures.json")
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000, help="the held-out split's seed")
    parser.add_argument("--noise", type=float, default=0.2)
    args = parser.parse_args()

    from trigon.backends.torch_readout import TorchReadoutBackend
    from trigon.calibration.temperature import TemperatureScaler

    backend = TorchReadoutBackend.load(args.weights)
    scaler = TemperatureScaler.load(args.temperatures) if args.temperatures else None
    engine = Engine(backend, scaler=scaler)
    print(f"model: {backend.model_version}\n")

    cases = synthetic_outcome_cases(n=args.cases, seed=args.seed, noise=args.noise)
    for case in cases:
        state = case.request.state
        response = engine.answer(case.request)
        plan, at_risk, size = (response.answers[q] for q in ("plan", "at_risk", "size"))
        top = sorted(plan.probabilities.items(), key=lambda kv: -kv[1])[:2]
        truth_plan = case.request.questions["plan"].options[case.expected["plan"].label].name
        truth_size = case.request.questions["size"].levels[case.expected["size"].label].name
        truth_risk = "yes" if case.expected["at_risk"].probability == 1.0 else "no"

        print(
            f"plan={state['plan']}, seats={state['seats']}, "
            f"tickets={state['open_tickets']}, payment_failed={state['payment_failed']}"
        )
        mark = "OK" if plan.selected == truth_plan else "X"
        distribution = f"[{top[0][0]}={top[0][1]:.3f} {top[1][0]}={top[1][1]:.3f}]"
        rows = [
            (
                f"  plan    -> {plan.selected:<11} conf={plan.confidence:.3f}  {distribution}",
                f"truth: {truth_plan:<11} {mark}",
            ),
            (f"  at_risk -> P(yes)={at_risk.probability:.3f}", f"truth: {truth_risk}"),
            (f"  size    -> {size.score:.2f}   conf={size.confidence:.3f}", f"truth: {truth_size}"),
        ]
        width = max(len(left) for left, _ in rows) + 3
        for left, right in rows:
            print(f"{left.ljust(width)}{right}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
