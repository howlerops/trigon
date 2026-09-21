"""Isotonic calibration: the calibrator that is not one-parameter.

Temperature scaling sharpens or flattens everywhere at once, so a head that is
overconfident where it is confident and underconfident where it is not has no
correct temperature — and the fitter returns its best one anyway. One seed of
the 8,000-case sweep has a Choice head scoring ECE 0.0879 under four different
rules about whether to apply a temperature, because none of them is deciding
about something that can help it.

These tests pin the properties that make this a different tool, not the
question of when to prefer it — that is `scripts/calibrator_choice.py`, which
scores both against heads whose true calibration is known.
"""

from __future__ import annotations

import pytest

from trigon.calibration.isotonic import (
    MIN_ISOTONIC_SAMPLES,
    IsotonicCalibrator,
    _pav,
)


def test_pav_returns_the_monotone_least_squares_fit():
    """Pool adjacent violators, on the textbook cases."""
    # Already non-decreasing: nothing to pool.
    assert [value for _, value in _pav([1, 2, 3], [0.0, 0.5, 1.0])] == [0.0, 0.5, 1.0]

    # Strictly decreasing: everything pools to the overall mean.
    knots = _pav([1, 2, 3], [1.0, 0.0, 0.0])
    assert len(knots) == 1
    assert knots[0][1] == pytest.approx(1 / 3)

    # A single violation pools just its neighbours, and the result never
    # decreases — which is the one property the whole algorithm exists for.
    values = [value for _, value in _pav([1, 2, 3, 4, 5], [0.0, 1.0, 0.0, 1.0, 1.0])]
    assert values == sorted(values)


def _tilted(rng, n):
    """A head no temperature can fix: over at high confidence, under at low."""
    confidences, correct = [], []
    for _ in range(n):
        stated = rng.uniform(0.55, 0.95)
        true_rate = min(max(stated - 0.18 * (stated - 0.75) / 0.20, 0.05), 0.99)
        confidences.append(stated)
        correct.append(1 if rng.random() < true_rate else 0)
    return confidences, correct


def test_it_corrects_a_tilt_that_no_temperature_can():
    """The reason this exists, stated as a test.

    A temperature is monotone *and* one-parameter: it can only move every
    confidence the same way. Isotonic is monotone and unconstrained otherwise,
    so it can pull the high end down while pushing the low end up.
    """
    import random

    rng = random.Random(4)
    confidences, correct = _tilted(rng, 4000)
    calibrator = IsotonicCalibrator()
    calibrator.fit("choice", confidences, correct)

    low = calibrator.confidence("choice", 0.60)
    high = calibrator.confidence("choice", 0.92)
    # Raised at the bottom, lowered at the top: opposite corrections at once,
    # which is precisely what a single temperature cannot do.
    assert low > 0.60
    assert high < 0.92


def test_the_map_is_non_decreasing_everywhere():
    """A calibrator that reorders confidences would change which answers look
    more certain than which, and nothing downstream expects that."""
    import random

    rng = random.Random(9)
    confidences, correct = _tilted(rng, 2000)
    calibrator = IsotonicCalibrator()
    calibrator.fit("choice", confidences, correct)

    probe = [i / 100 for i in range(101)]
    mapped = [calibrator.confidence("choice", value) for value in probe]
    assert mapped == sorted(mapped)


def test_applying_it_preserves_the_ranking_and_the_total():
    """It calibrates the confidence, never the decision.

    The remaining mass is spread over the other options in their existing
    proportions. A calibrator that reordered a Choice's options would be
    changing the answer rather than the confidence in it.
    """
    import random

    rng = random.Random(9)
    confidences, correct = _tilted(rng, 2000)
    calibrator = IsotonicCalibrator()
    calibrator.fit("choice", confidences, correct)

    before = [0.55, 0.25, 0.15, 0.05]
    after = calibrator.apply("choice", before)
    assert sum(after) == pytest.approx(1.0)
    assert sorted(range(4), key=lambda i: -before[i]) == sorted(range(4), key=lambda i: -after[i])
    # The non-top options keep their relative sizes exactly.
    assert after[1] / after[2] == pytest.approx(before[1] / before[2])


def test_an_unfitted_primitive_is_the_identity():
    """Absence of a calibration is not a calibration to uniform."""
    calibrator = IsotonicCalibrator()
    assert calibrator.confidence("score", 0.73) == 0.73
    assert calibrator.apply("score", [0.7, 0.2, 0.1]) == [0.7, 0.2, 0.1]


def test_it_refuses_a_split_too_small_to_fit_on():
    """With k points it can place k steps, so a small split is memorised.

    Refusing is the honest behaviour: temperature scaling fits one number and
    is the right tool there, and a silently overfitted isotonic map would
    report excellent calibration on the split it memorised.
    """
    calibrator = IsotonicCalibrator()
    with pytest.raises(ValueError, match="at least"):
        calibrator.fit("choice", [0.5] * 10, [1] * 10)
    assert MIN_ISOTONIC_SAMPLES >= 100

    with pytest.raises(ValueError, match="differ in length"):
        calibrator.fit("choice", [0.5] * 500, [1] * 499)


def test_it_round_trips_through_a_file(tmp_path):
    import random

    rng = random.Random(1)
    confidences, correct = _tilted(rng, 1000)
    calibrator = IsotonicCalibrator()
    calibrator.fit("choice", confidences, correct)
    path = tmp_path / "isotonic.json"
    calibrator.save(path)

    reloaded = IsotonicCalibrator.load(path)
    assert reloaded.fitted_on == {"choice": 1000}
    for value in (0.0, 0.3, 0.61, 0.9, 1.0):
        assert reloaded.confidence("choice", value) == calibrator.confidence("choice", value)
