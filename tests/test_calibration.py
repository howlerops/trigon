"""Metrics against hand-computed cases, and recovery of a known temperature."""

from __future__ import annotations

import math
import random
import warnings

import pytest

from trigon.calibration import (
    ConformalPredictor,
    TemperatureScaler,
    adaptive_calibration_error,
    brier,
    coverage,
    expected_calibration_error,
    fit_conformal,
    fit_temperature,
    maximum_calibration_error,
    mean_set_size,
    negative_log_likelihood,
    report,
)
from trigon.calibration.conformal import ConformalMethod
from trigon.calibration.temperature import CalibrationWarning, fit_binary_temperature


def test_brier_matches_the_definition():
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2
    assert brier([[0.7, 0.2, 0.1]], [0]) == pytest.approx(0.09 + 0.04 + 0.01)


def test_nll_matches_the_definition():
    assert negative_log_likelihood([[0.7, 0.3]], [0]) == pytest.approx(-math.log(0.7))


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    # 80 confident-at-0.8 predictions, 80% of which are right.
    probs = [[0.8, 0.2]] * 100
    labels = [0] * 80 + [1] * 20
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_equals_the_gap_for_a_single_bin():
    probs = [[0.9, 0.1]] * 100
    labels = [0] * 60 + [1] * 40
    assert expected_calibration_error(probs, labels) == pytest.approx(0.3)
    assert maximum_calibration_error(probs, labels) == pytest.approx(0.3)


def test_adaptive_ece_sees_what_equal_width_binning_hides():
    """Everything lands in one equal-width bin, where the over- and
    under-confident halves cancel. Equal-mass binning separates them."""
    probs = [[0.71, 0.29]] * 50 + [[0.79, 0.21]] * 50
    labels = [0] * 50 + [1] * 50  # first half all right, second half all wrong
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.25, abs=0.02)
    assert adaptive_calibration_error(probs, labels, n_bins=10) > 0.5


def test_equal_mass_binning_does_not_split_ties():
    """A constant predictor at the base rate is calibrated by definition. If
    the equal-mass binner cut through its single confidence value, sampling
    noise inside the bins would report it as badly miscalibrated."""
    probs = [[0.49, 0.51]] * 400
    labels = [1] * 204 + [0] * 196
    assert adaptive_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_report_flags_overconfidence_with_a_sign():
    rep = report([[0.99, 0.01]] * 100, [0] * 70 + [1] * 30)
    assert rep.accuracy == pytest.approx(0.7)
    assert rep.overconfidence > 0.25


def test_temperature_recovers_a_known_scaling():
    rng = random.Random(7)
    probs, labels = _calibrated_sample(rng, 1500, 4)
    sharpened = [[math.log(max(p, 1e-9)) * 3.0 for p in row] for row in probs]
    assert fit_temperature(sharpened, labels) == pytest.approx(3.0, rel=0.1)


def test_temperature_leaves_a_calibrated_set_alone():
    rng = random.Random(11)
    probs, labels = _calibrated_sample(rng, 1500, 4)
    logits = [[math.log(max(p, 1e-9)) for p in row] for row in probs]
    assert fit_temperature(logits, labels) == pytest.approx(1.0, rel=0.12)


def test_binary_temperature_recovers_a_known_scaling():
    rng = random.Random(3)
    logits, labels = [], []
    for _ in range(2000):
        logit = rng.uniform(-4, 4)
        logits.append(logit * 2.5)
        labels.append(int(rng.random() < 1 / (1 + math.exp(-logit))))
    assert fit_binary_temperature(logits, labels) == pytest.approx(2.5, rel=0.15)


def test_degenerate_fit_warns_rather_than_returning_a_quiet_number():
    with pytest.warns(CalibrationWarning, match="ceiling"):
        fit_temperature([[0.0, 0.0, 0.0]] * 60, [i % 3 for i in range(60)])


def test_scaler_falls_back_from_domain_to_primitive_to_one():
    scaler = TemperatureScaler(primitive={"choice": 2.0}, domain={"support": {"choice": 3.0}})
    assert scaler.temperature("choice", "support") == 3.0
    assert scaler.temperature("choice", "unknown") == 2.0
    assert scaler.temperature("score") == 1.0


def test_scaler_round_trips_through_disk(tmp_path):
    scaler = TemperatureScaler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        scaler.fit("choice", [[1.0, 0.0], [0.0, 1.0]] * 30, [0, 1] * 30)
    path = tmp_path / "t.json"
    scaler.save(path)
    assert TemperatureScaler.load(path).to_dict() == scaler.to_dict()


@pytest.mark.parametrize("method", [ConformalMethod.LAC, ConformalMethod.APS])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_hits_its_coverage_target(method, alpha):
    rng = random.Random(42)
    probs, labels = _calibrated_sample(rng, 4000, 5)
    predictor = fit_conformal(probs[:2000], labels[:2000], alpha=alpha, method=method)
    sets = [predictor.predict(p).indices for p in probs[2000:]]
    achieved = coverage(sets, labels[2000:])
    # Marginal coverage is guaranteed in expectation; allow finite-sample slack.
    assert achieved >= (1 - alpha) - 0.03
    # A set can never exceed the label count, and the guarantee is worthless if
    # it does not bound anything -- but see the APS test below: on a genuinely
    # uniform problem at a tight alpha, "all labels" is the honest answer.
    assert mean_set_size(sets) <= 5


def test_aps_is_more_conservative_than_lac():
    """The documented trade-off: APS buys difficulty-tracking coverage with
    larger sets, and without randomization it over-covers."""
    rng = random.Random(42)
    probs, labels = _calibrated_sample(rng, 4000, 5)
    lac = fit_conformal(probs[:2000], labels[:2000], alpha=0.1, method=ConformalMethod.LAC)
    aps = fit_conformal(probs[:2000], labels[:2000], alpha=0.1, method=ConformalMethod.APS)
    lac_sets = [lac.predict(p).indices for p in probs[2000:]]
    aps_sets = [aps.predict(p).indices for p in probs[2000:]]
    assert mean_set_size(aps_sets) > mean_set_size(lac_sets)
    assert coverage(aps_sets, labels[2000:]) > coverage(lac_sets, labels[2000:])


def test_conformal_refuses_a_calibration_split_that_is_too_small():
    with pytest.raises(ValueError, match="at least"):
        fit_conformal([[0.5, 0.5]] * 3, [0, 1, 0], alpha=0.01)


def test_conformal_round_trips_through_disk(tmp_path):
    predictor = ConformalPredictor(
        alpha=0.1, method=ConformalMethod.LAC, threshold=0.4, calibration_n=500
    )
    path = tmp_path / "c.json"
    predictor.save(path)
    assert ConformalPredictor.load(path) == predictor


def test_metrics_reject_probabilities_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sums to"):
        report([[0.5, 0.2]], [0])


def _calibrated_sample(rng: random.Random, n: int, k: int):
    """Draw labels from the stated distribution, so the sample is calibrated
    by construction and any measured miscalibration is the estimator's."""
    probs, labels = [], []
    for _ in range(n):
        row = [rng.random() for _ in range(k)]
        total = sum(row)
        row = [x / total for x in row]
        probs.append(row)
        labels.append(rng.choices(range(k), weights=row)[0])
    return probs, labels
