"""Coverage is the only thing a conformal wrapper promises, so it is gated.

`README.md` tells a user facing calibration transfer to fit a wrapper on their
own labels and "read the coverage number rather than the ECE". That number was
computed, printed to stderr, and checked by nothing — a guarantee a reader was
trusted to notice.

The interesting part is not the comparison, it is the floor. Empirical coverage
on n held-out points is Binomial(n, 1 - alpha) / n even for a perfect
predictor, so a fixed tolerance is vacuous at small n and spuriously red at
large n. This is the same mistake `gate_is_testable` exists to prevent for
ECE — a limit quoted without the noise under it — so the floor is derived from
the target and the count instead.
"""

from __future__ import annotations

import random

import pytest

from trigon.limits import CONFORMAL_COVERAGE_SIGMAS, conformal_coverage_floor


def test_the_floor_tightens_with_the_sample_and_never_reaches_the_target():
    """More evidence, less room — but never zero room."""
    floors = [conformal_coverage_floor(0.9, n) for n in (100, 300, 900, 1500, 6000)]
    assert floors == sorted(floors), "a larger sample must not widen the excuse"
    assert all(f < 0.9 for f in floors), "a perfect predictor must be allowed to be unlucky"
    # The shape is sqrt(n): four times the data halves the gap to the target.
    assert (0.9 - conformal_coverage_floor(0.9, 1500)) / (
        0.9 - conformal_coverage_floor(0.9, 6000)
    ) == pytest.approx(2.0, rel=1e-6)


def test_the_floor_refuses_inputs_it_cannot_judge():
    for target in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="coverage target"):
            conformal_coverage_floor(target, 1000)
    with pytest.raises(ValueError, match="no held-out points"):
        conformal_coverage_floor(0.9, 0)


def test_a_correctly_covering_predictor_almost_never_trips_the_gate():
    """The false-positive half: a gate that cries wolf gets switched off.

    Simulates a predictor that genuinely covers at exactly its target, which is
    what split conformal delivers, and counts how often the empirical rate on a
    finite sample falls under the floor.
    """
    rng = random.Random(20260921)
    target, n, trials = 0.9, 1500, 2000
    floor = conformal_coverage_floor(target, n)
    tripped = sum(
        1 for _ in range(trials) if sum(rng.random() < target for _ in range(n)) / n < floor
    )
    # Three sigma one-sided is about 1 in 750; allow generous headroom for the
    # discreteness of a binomial at this n, and still assert it is rare.
    assert tripped / trials < 0.01, f"{tripped} of {trials} correct predictors were failed"


@pytest.mark.parametrize(
    ("true_coverage", "n", "at_least"),
    [
        # The gate's power, measured rather than hoped for. A first draft of
        # this test asserted that 0.87 at n=1,500 is caught 95% of the time; it
        # is caught 78% of the time, and finding that out is why the power
        # table in `limits.py` exists. A gate whose sensitivity nobody has
        # computed is a limit quoted without the noise under it, which is the
        # thing this project refuses everywhere else.
        (0.85, 1500, 0.95),  # five points under: certain
        (0.87, 1500, 0.70),  # three points under: likely, not certain
        (0.87, 3000, 0.95),  # ...and certain once the sample doubles
        (0.80, 300, 0.90),  # a gross breach is caught on almost nothing
    ],
)
def test_a_predictor_that_under_covers_is_caught(true_coverage, n, at_least):
    """The true-positive half, across breach sizes and sample sizes.

    A user reading "90% coverage" and getting 85% is being told something
    untrue, which is the failure this gate exists for. How reliably it catches
    a *smaller* lie depends on how much held-out data there is, and the honest
    summary is that this is a floor against a broken wrapper rather than an
    assurance that one is exact.
    """
    rng = random.Random(20260921)
    trials = 500
    floor = conformal_coverage_floor(0.9, n)
    caught = sum(
        1 for _ in range(trials) if sum(rng.random() < true_coverage for _ in range(n)) / n < floor
    )
    assert caught / trials > at_least, (
        f"true coverage {true_coverage} at n={n}: only {caught} of {trials} were caught"
    )


def test_the_gate_is_one_sided_on_purpose():
    """Over-covering is a utility problem, not a broken promise.

    A predictor returning every option covers perfectly and says nothing, which
    is why `mean_set_size` is reported beside coverage — but it has not lied,
    so it is not this gate's business. Coverage below target is.
    """
    assert conformal_coverage_floor(0.9, 1500) < 0.9
    assert CONFORMAL_COVERAGE_SIGMAS > 0
    # There is deliberately no ceiling function to pair with the floor.
    import trigon.limits as limits

    assert not hasattr(limits, "conformal_coverage_ceiling")
