#!/usr/bin/env python
"""Temperature or isotonic? Scored on heads whose true calibration is known.

    python scripts/calibrator_choice.py

`scripts/decline_rule.py` answers "should this temperature be applied". This
answers the question underneath it: **is a temperature the right tool at all.**

It exists because of a specific dead end. One seed of the 8,000-case sweep has
a Choice head scoring ECE 0.0879 under four different rules about whether to
apply a temperature — the head does not move, because none of those rules is
deciding about something that can help it. Temperature scaling sharpens or
flattens everywhere at once, and that head is *tilted*: overconfident where it
is confident, underconfident where it is not.

Isotonic regression fits an arbitrary non-decreasing map, so a tilt is exactly
what it can correct. The cost is a weaker guarantee and a hunger for data, so
this measures both, at several calibration-split sizes.

Every number is scored on a third draw that neither the fit nor the selection
ever saw.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from decline_rule import _head  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    from trigon.calibration.isotonic import MIN_ISOTONIC_SAMPLES, IsotonicCalibrator
    from trigon.calibration.metrics import report
    from trigon.calibration.temperature import fit_temperature
    from trigon.cli import _help_is_real, _scaled

    shapes = [
        ("already calibrated", dict(stated=0.75, true_rate=0.75, spread=0.18)),
        ("clearly overconfident", dict(stated=0.90, true_rate=0.70, spread=0.05)),
        ("tilted", dict(stated=0.75, true_rate=0.75, spread=0.18, tilt=0.10)),
        ("tilted hard", dict(stated=0.75, true_rate=0.75, spread=0.18, tilt=0.22)),
    ]

    print(f"{args.trials} trials per cell, scored on 4,000 unseen answers")
    print()
    print(f"{'head':<22} {'fit n':>6} {'none':>9} {'temperature':>12} {'isotonic':>10}")
    print("-" * 64)
    for label, shape in shapes:
        for fit_n in (200, MIN_ISOTONIC_SAMPLES, 1000, 4000):
            scores = {"none": [], "temperature": [], "isotonic": []}
            for trial in range(args.trials):
                rng = random.Random(500 + trial)
                fit_rows = _head(rng, fit_n, 4, **shape)
                eval_rows = _head(rng, 4000, 4, **shape)

                probs = [_scaled(x, 1.0) for x, _ in eval_rows]
                labels = [y for _, y in eval_rows]
                scores["none"].append(report(probs, labels, simulate_floor=False).ece)

                # Temperature, under the rule the pipeline ships.
                value = fit_temperature([x for x, _ in fit_rows], [y for _, y in fit_rows])
                half = max(1, len(fit_rows) // 2)
                check = fit_rows[half:]
                keep = _help_is_real(
                    [_scaled(x, 1.0) for x, _ in check],
                    [_scaled(x, value) for x, _ in check],
                    [y for _, y in check],
                    resamples=80,
                )
                scaled = [_scaled(x, value if keep else 1.0) for x, _ in eval_rows]
                scores["temperature"].append(report(scaled, labels, simulate_floor=False).ece)

                # Isotonic, where there is enough data to fit one.
                if fit_n < MIN_ISOTONIC_SAMPLES:
                    scores["isotonic"].append(float("nan"))
                    continue
                calibrator = IsotonicCalibrator()
                fit_probs = [_scaled(x, 1.0) for x, _ in fit_rows]
                calibrator.fit(
                    "choice",
                    [max(p) for p in fit_probs],
                    [
                        int(max(range(len(p)), key=lambda i: p[i]) == y)
                        for p, (_, y) in zip(fit_probs, fit_rows, strict=True)
                    ],
                )
                mapped = [calibrator.apply("choice", p) for p in probs]
                scores["isotonic"].append(report(mapped, labels, simulate_floor=False).ece)

            def worst(key, scores=scores):
                values = [v for v in scores[key] if not math.isnan(v)]
                return f"{max(values):.4f}" if values else "—"

            print(
                f"{label:<22} {fit_n:>6} {worst('none'):>9} "
                f"{worst('temperature'):>12} {worst('isotonic'):>10}"
            )
        print()

    print("Worst-case ECE across trials: a release gate is failed by the bad run.")
    print(f"Isotonic is not attempted below {MIN_ISOTONIC_SAMPLES} answers, where it would be")
    print("memorising rather than calibrating -- with k points it can place k steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
