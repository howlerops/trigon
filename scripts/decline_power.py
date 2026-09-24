#!/usr/bin/env python
"""Does the calibrator decline rule have the power to act on a 77-way head?

    python scripts/decline_power.py --trials 40

The certified Banking77 model shipped two of four seeds uncalibrated at ECE
0.045-0.049 -- a whisker under the 0.05 gate -- while on the other two the
isotonic map took the same kind of head to 0.01-0.02. The run logs say what
happened: on the declined seeds the held-out check read 0.0486 -> 0.0261 and
0.0420 -> 0.0318, and the rule refused anyway. One seed flipped between
applying and declining across two runs.

The hypothesis is power: the check is 500 answers, a perfectly calibrated
head reads ECE ~0.02-0.03 at that size, and "better in 95% of paired
resamples" cannot separate a 5-point overconfidence from noise often enough.
A seed sweep cannot test that -- it has no ground truth about whether a
calibrator *should* have been applied -- so this constructs heads whose true
calibration is known, at Banking77's shape, and runs the **shipped** decision
(`trigon.cli._choose_calibrator`) on them.

Each trial draws a fit/check split, lets a rule decide, and scores what it
decided on a fresh 5,000-answer draw with the statistic the gates read:
max(ECE, adaptive ECE). The worst column matters more than the mean -- a
release gate is failed by the bad run, not the average one.

Rules compared, all scored the same way:

* **shipped** -- 1,000 calibration answers, apply only if better in >=95% of
  paired resamples of the 500-answer check.
* **more data** -- the same rule on 2,000 calibration answers.
* **looser** -- the same data, accept at >=80% of resamples.
* **unless harm** -- apply unless the fit is worse in >=95% of resamples; the
  mirror rule `scripts/decline_rule.py` measured and rejected at 4 classes.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

K = 77


def _head(rng: random.Random, n: int, gap: float, tilt: float = 0.0):
    """Rows of (log-probabilities, label) from a head with known calibration.

    Confidence is clustered near 1 with a tail, the way a trained 77-way
    head's is: 85% of answers at 0.93-0.999, the rest spread over 0.3-0.93.
    The head is right with probability ``confidence - gap`` -- so ``gap`` is
    its true overconfidence -- plus ``tilt`` times how far it sits below 0.9,
    which makes it underconfident in the tail while overconfident at the top:
    a shape no temperature fixes.
    """
    rows = []
    for _ in range(n):
        c = rng.uniform(0.93, 0.999) if rng.random() < 0.85 else rng.uniform(0.30, 0.93)
        rate = min(max(c - gap + tilt * max(0.0, 0.9 - c), 0.01), 0.999)
        top = rng.randrange(K)
        probs = [(1.0 - c) / (K - 1)] * K
        probs[top] = c
        if rng.random() < rate:
            label = top
        else:
            label = rng.randrange(K - 1)
            label += label >= top
        rows.append(([math.log(max(p, 1e-12)) for p in probs], label))
    return rows


RULES = {
    # name: (calibration answers, accept threshold, which help test)
    "shipped": (1000, None, "help"),
    "more data": (2000, None, "help"),
    "looser": (1000, 0.80, "help"),
    "unless harm": (1000, None, "harm"),
}
SHAPES = [
    ("calibrated", dict(gap=0.0)),
    ("overconfident 0.03", dict(gap=0.03)),
    ("overconfident 0.05", dict(gap=0.05)),
    ("overconfident 0.08", dict(gap=0.08)),
    ("tilted", dict(gap=0.05, tilt=0.4)),
]


def _trial(job):
    """One (shape, rule, trial): the shipped decision on a drawn head, scored fresh."""
    label, rule, trial, eval_n = job
    from trigon import cli
    from trigon.calibration.isotonic import IsotonicCalibrator
    from trigon.calibration.temperature import TemperatureScaler

    shape = dict(SHAPES)[label]
    cal_n, threshold, test = RULES[rule]
    rng = random.Random(7000 + trial)
    data = _head(rng, cal_n, **shape)
    fresh = _head(rng, eval_n, **shape)

    if threshold is not None:
        cli.ACCEPT_CONFIDENCE = threshold
    if test == "harm":
        harm = cli._harm_is_real
        cli._help_is_real = lambda u, r, y, **kw: not harm(u, r, y, **kw)
    scaler, isotonic = TemperatureScaler(), IsotonicCalibrator()
    verdict = cli._choose_calibrator("choice", data, scaler, isotonic)

    def gate_error(sc, iso):
        probs = [
            iso.apply("choice", cli._scaled(x, 1.0))
            if "choice" in iso.knots
            else cli._scaled(x, sc.primitive.get("choice", 1.0))
            for x, _ in fresh
        ]
        return cli._calibration_error(probs, [y for _, y in fresh])

    raw = gate_error(TemperatureScaler(), IsotonicCalibrator()) if rule == "shipped" else None
    return label, rule, not verdict.startswith("none"), gate_error(scaler, isotonic), raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=5000)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    import multiprocessing

    # A fresh process per job, so a rule's monkeypatched threshold never
    # leaks into another rule's trial.
    jobs = [
        (label, rule, trial, args.eval_n)
        for label, _ in SHAPES
        for rule in RULES
        for trial in range(args.trials)
    ]
    with multiprocessing.get_context("spawn").Pool(args.jobs, maxtasksperchild=1) as pool:
        results = pool.map(_trial, jobs, chunksize=1)

    print(f"{K}-way heads, {args.trials} trials per cell, scored on {args.eval_n} fresh answers")
    print()
    print(f"{'head':<20} {'rule':<12} {'applied':>8} {'mean':>8} {'worst':>8}  {'raw':>8}")
    print("-" * 72)
    for label, _ in SHAPES:
        raw = [r[4] for r in results if r[0] == label and r[1] == "shipped"]
        for rule in RULES:
            cell = [r for r in results if r[0] == label and r[1] == rule]
            errors = [r[3] for r in cell]
            print(
                f"{label:<20} {rule:<12} {sum(r[2] for r in cell):>4}/{len(cell):<3} "
                f"{statistics.mean(errors):>8.4f} {max(errors):>8.4f}  "
                + (f"{statistics.mean(raw):>8.4f}" if rule == "shipped" else "")
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
