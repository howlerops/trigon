#!/usr/bin/env python
"""Measure the two temperature accept/decline rules, on heads with known truth.

    python scripts/decline_rule.py

`trigon train` fits a temperature per primitive and then has to decide whether
to apply it. `docs/decisions.md`, "A temperature is a proposal, not a result",
explains why that decision has to exist at all: the fit minimises NLL, NLL is
not ECE, and temperature scaling is a one-parameter family that cannot fix a
head whose miscalibration is not a uniform sharpening.

**Why simulate rather than sweep seeds.** The four-seed sweep can tell you a
rule changed something; it cannot tell you whether the change was right,
because it has no ground truth about whether a given temperature *should* have
been applied. Here the head is constructed, so the correct decision is known
and the two rules can be scored against it — at whatever sample size, as often
as you like, in seconds rather than in four twenty-minute training runs.

The two rules:

* **bare** — decline if held-out ECE is not lower. A threshold on a noisy
  estimate with nothing under it. This shipped for one commit and its cost
  showed up immediately: on one seed it discarded a temperature that was
  helping and moved that run's ECE from 0.0128 to 0.0219.
* **bootstrap** — decline only if the fit is worse in ≥95% of paired
  resamples of the check split.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def _head(rng, n, k, stated, true_rate, *, spread=0.0, tilt=0.0):
    """One head's worth of (logits, label) rows.

    `stated` is the confidence it reports and `true_rate` how often it is
    right. With `spread`, the confidence varies uniformly over
    `stated ± spread` instead of being constant, and `tilt` makes the *error*
    depend on the confidence: positive tilt means overconfident where it is
    confident and underconfident where it is not.

    `tilt` is the regime the whole decision exists for. Temperature scaling is
    a one-parameter family; it sharpens or flattens everywhere at once. A head
    with tilt cannot be fixed by any temperature, so the fitted one is
    guaranteed to be the wrong tool and the only question is whether applying
    it does harm. A study without this shape cannot score the rules on the
    case they were written for -- the first version of this script had no such
    shape and reported that the two rules were interchangeable.
    """
    rows = []
    for _ in range(n):
        confidence = min(max(stated + rng.uniform(-spread, spread), 0.30), 0.97)
        offset = tilt * (confidence - stated) / max(spread, 1e-9) if spread else 0.0
        rate = min(max(true_rate + (confidence - stated) - offset, 0.05), 0.99)
        probs = [(1.0 - confidence) / (k - 1)] * k
        probs[0] = confidence
        label = 0 if rng.random() < rate else rng.randrange(1, k)
        rows.append(([math.log(max(p, 1e-12)) for p in probs], label))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--check-n", type=int, default=500)
    args = parser.parse_args()

    from trigon.calibration.metrics import report
    from trigon.calibration.temperature import fit_temperature
    from trigon.cli import _harm_is_real, _help_is_real, _scaled

    def ece(rows, temperature):
        return report(
            [_scaled(x, temperature) for x, _ in rows], [y for _, y in rows], simulate_floor=False
        ).ece

    # Each row is a head shape. `gap` is how far its stated confidence sits
    # from its true accuracy: 0 is already calibrated (no temperature can help,
    # so applying one is always the wrong call), large is plainly fixable.
    shapes = [
        ("already calibrated", dict(stated=0.75, true_rate=0.75)),
        ("slightly overconfident", dict(stated=0.75, true_rate=0.72)),
        ("clearly overconfident", dict(stated=0.90, true_rate=0.70)),
        ("clearly underconfident", dict(stated=0.60, true_rate=0.80)),
        # Varying confidence: the realistic baseline for the two below.
        ("spread, calibrated", dict(stated=0.75, true_rate=0.75, spread=0.18)),
        # Tilted: over at high confidence, under at low. No temperature fixes
        # it, so the fit is the wrong tool and the rule's job is to notice.
        ("spread, tilted", dict(stated=0.75, true_rate=0.75, spread=0.18, tilt=0.10)),
        ("spread, tilted hard", dict(stated=0.75, true_rate=0.75, spread=0.18, tilt=0.22)),
    ]

    print(f"{args.trials} trials per shape, {args.check_n}-point check split")
    print()
    print(f"{'head':<26} {'rule':<11} {'declined':>9} {'mean ECE':>10} {'worst ECE':>10}")
    print("-" * 70)
    for label, shape in shapes:
        results: dict[str, list[float]] = {"bare": [], "bootstrap": [], "strict": []}
        declines = {"bare": 0, "bootstrap": 0, "strict": 0}
        never = []
        for trial in range(args.trials):
            rng = random.Random(1000 + trial)
            fit_rows = _head(rng, args.check_n, 4, **shape)
            check_rows = _head(rng, args.check_n, 4, **shape)
            # A fresh draw the decision never sees: the rules are scored on
            # what they do to data neither the fit nor the check touched.
            eval_rows = _head(rng, 4000, 4, **shape)
            never.append(eval_rows)

            value = fit_temperature([x for x, _ in fit_rows], [y for _, y in fit_rows])
            unscaled = [_scaled(x, 1.0) for x, _ in check_rows]
            rescaled = [_scaled(x, value) for x, _ in check_rows]
            labels = [y for _, y in check_rows]

            bare = (
                report(rescaled, labels, simulate_floor=False).ece
                >= report(unscaled, labels, simulate_floor=False).ece
            )
            boot = _harm_is_real(unscaled, rescaled, labels, resamples=120)
            # Burden of proof on ACCEPTING rather than on declining: keep
            # the fit only where it clearly helps. The mirror of
            # `bootstrap`, and the rule the `never scale` column argues
            # for -- not scaling wins on every shape a temperature cannot
            # fix, and loses only where the improvement is unmistakable.
            strict = not _help_is_real(unscaled, rescaled, labels, resamples=120)

            for rule, declined in (("bare", bare), ("bootstrap", boot), ("strict", strict)):
                declines[rule] += declined
                results[rule].append(ece(eval_rows, 1.0 if declined else value))

        # The third option: never fit at all. Without it the table compares two
        # rules and cannot say whether either beats doing nothing.
        unscaled_only = [ece(rows, 1.0) for rows in never]
        for rule, values, count in (
            ("bare", results["bare"], f"{declines['bare']}/{args.trials}"),
            ("bootstrap", results["bootstrap"], f"{declines['bootstrap']}/{args.trials}"),
            ("strict", results["strict"], f"{declines['strict']}/{args.trials}"),
            ("never scale", unscaled_only, f"{args.trials}/{args.trials}"),
        ):
            print(
                f"{label:<26} {rule:<11} {count:>9} "
                f"{statistics.mean(values):>10.4f} {max(values):>10.4f}"
            )
        print()

    print("Scored on a third draw the decision never saw. Lower is better, and")
    print("the worst column matters more than the mean: a release gate is")
    print("failed by the bad run, not by the average one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
