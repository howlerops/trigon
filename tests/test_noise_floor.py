"""The noise floor: what makes a published ECE evidence rather than a number."""

from __future__ import annotations

import random

import pytest

from trigon.calibration.metrics import noise_floor, report


def _uniformish(n: int, k: int, seed: int = 0):
    rng = random.Random(seed)
    probs = []
    for _ in range(n):
        row = [rng.random() for _ in range(k)]
        total = sum(row)
        probs.append([x / total for x in row])
    return probs


def test_floor_shrinks_as_the_sample_grows():
    """The whole point: at small n a perfectly calibrated model still scores a
    large ECE, so a small-n ECE cannot support a calibration claim."""
    small = noise_floor(_uniformish(60, 4), trials=60)
    large = noise_floor(_uniformish(2000, 4), trials=60)
    assert small.mean > large.mean * 2
    assert small.p95 >= small.mean


def test_a_calibrated_model_is_reported_as_indistinguishable():
    rng = random.Random(3)
    probs = _uniformish(1500, 4, seed=3)
    labels = [rng.choices(range(4), weights=row)[0] for row in probs]
    result = report(probs, labels, trials=60)
    assert result.distinguishable is False


def test_real_miscalibration_is_reported_as_distinguishable():
    probs = [[0.97, 0.01, 0.01, 0.01]] * 1500
    labels = [0] * 900 + [1] * 600  # 60% accurate at 97% confidence
    result = report(probs, labels, trials=60)
    assert result.distinguishable is True
    assert result.ece > result.floor.p95


def test_no_floor_means_no_verdict():
    result = report([[0.6, 0.4]] * 50, [0] * 50, simulate_floor=False)
    assert result.floor is None
    assert result.distinguishable is None


def test_floor_is_deterministic_for_a_seed():
    probs = _uniformish(200, 3, seed=9)
    assert noise_floor(probs, trials=30, seed=1) == noise_floor(probs, trials=30, seed=1)


def test_floor_needs_enough_trials_to_have_a_quantile():
    with pytest.raises(ValueError, match="at least 2 trials"):
        noise_floor(_uniformish(10, 2), trials=1)
