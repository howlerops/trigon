"""Properties of the scoring primitives, not just their happy paths.

Everything above these functions -- confidence, ECE, the gates, the served
probabilities -- is a composition of them, so a silent error here is invisible
at every level that reads it. Coverage found the overflow branches, the
degenerate inputs and the guard clauses untested; those are exactly where a
numerical bug hides without failing anything.
"""

from __future__ import annotations

import math

import pytest

from trigon.numeric import (
    brier_score,
    entropy,
    expectation,
    log_loss,
    normalize,
    normalized_entropy,
    sigmoid,
    softmax,
    standard_deviation,
)

# -- softmax -----------------------------------------------------------------


def test_softmax_is_a_distribution_and_shift_invariant():
    logits = [2.0, -1.0, 0.5, 7.25]
    probs = softmax(logits)
    assert sum(probs) == pytest.approx(1.0)
    assert all(p > 0 for p in probs)
    # Adding a constant to every logit must not move the distribution -- this
    # is what lets the implementation subtract the max for stability.
    assert softmax([x + 1000.0 for x in logits]) == pytest.approx(probs)


def test_softmax_survives_logits_that_would_overflow_exp():
    """exp(800) is inf. The max-subtraction is the only thing preventing nan."""
    for probs in (softmax([800.0, -800.0]), softmax([-800.0, 800.0])):
        assert all(math.isfinite(p) for p in probs)
        assert sum(probs) == pytest.approx(1.0)


def test_temperature_above_one_flattens_and_below_one_sharpens():
    logits = [3.0, 1.0, 0.0]
    sharp, base, flat = softmax(logits, 0.5), softmax(logits), softmax(logits, 4.0)
    assert max(sharp) > max(base) > max(flat)
    # Flattening has a limit: the uniform distribution.
    assert softmax(logits, 1e6) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-5)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_softmax_rejects_a_non_positive_temperature(bad):
    with pytest.raises(ValueError, match="temperature"):
        softmax([1.0, 2.0], bad)


def test_softmax_rejects_an_empty_vector():
    with pytest.raises(ValueError, match="at least one"):
        softmax([])


# -- sigmoid -----------------------------------------------------------------


def test_sigmoid_branches_agree_at_the_seam():
    """The implementation switches formula at x = 0; both sides must meet."""
    for x in (-1e-9, 0.0, 1e-9):
        assert sigmoid(x) == pytest.approx(0.5, abs=1e-9)


def test_sigmoid_is_symmetric_and_never_overflows():
    for x in (0.25, 3.0, 40.0, 800.0):
        assert sigmoid(x) + sigmoid(-x) == pytest.approx(1.0)
        assert math.isfinite(sigmoid(x)) and math.isfinite(sigmoid(-x))
    # The saturated ends stay inside [0, 1] rather than rounding outside it.
    assert 0.0 <= sigmoid(-800.0) < 0.5 < sigmoid(800.0) <= 1.0


def test_sigmoid_agrees_with_a_two_logit_softmax():
    """A Noul is one logit; the equivalent Choice is two. They must match."""
    for logit in (-4.0, -0.5, 0.0, 2.5):
        assert sigmoid(logit) == pytest.approx(softmax([logit, 0.0])[0])


def test_sigmoid_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        sigmoid(1.0, 0.0)


# -- normalize ---------------------------------------------------------------


def test_normalize_returns_uniform_for_an_all_zero_vector():
    """The degenerate case: no evidence for anything is not a divide by zero."""
    assert normalize([0.0, 0.0, 0.0, 0.0]) == pytest.approx([0.25] * 4)


def test_normalize_rejects_negatives_rather_than_producing_a_pseudo_distribution():
    with pytest.raises(ValueError, match="negative"):
        normalize([1.0, -0.5])
    with pytest.raises(ValueError, match="empty"):
        normalize([])


# -- entropy -----------------------------------------------------------------


def test_normalized_entropy_is_comparable_across_option_counts():
    """The whole reason confidence is scaled by log(n): a 2-option and a
    77-option answer have to read the same when both are uninformative."""
    for n in (2, 5, 77, 1000):
        assert normalized_entropy([1.0 / n] * n) == pytest.approx(1.0)
    assert normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert normalized_entropy([1.0]) == 0.0


def test_entropy_ignores_zero_mass_rather_than_returning_nan():
    assert entropy([0.5, 0.5, 0.0]) == pytest.approx(math.log(2))


# -- scoring rules -----------------------------------------------------------


def test_log_loss_is_finite_on_a_confidently_wrong_answer():
    """A proper scoring rule must punish this hard -- but a single infinity
    would swallow an entire eval run's mean."""
    penalty = log_loss([1.0, 0.0], label_index=1)
    assert math.isfinite(penalty) and penalty > 25


def test_cross_entropy_is_minimised_by_reporting_the_true_distribution():
    """Strict propriety, which is why it is the training objective."""
    truth = [0.6, 0.3, 0.1]
    honest = sum(p * log_loss(truth, i) for i, p in enumerate(truth))
    for lie in ([0.9, 0.05, 0.05], [1 / 3] * 3, [0.5, 0.4, 0.1]):
        misreported = sum(p * log_loss(lie, i) for i, p in enumerate(truth))
        assert misreported > honest


def test_brier_is_minimised_by_reporting_the_true_distribution():
    truth = [0.6, 0.3, 0.1]
    honest = sum(p * brier_score(truth, i) for i, p in enumerate(truth))
    for lie in ([0.9, 0.05, 0.05], [1 / 3] * 3):
        assert sum(p * brier_score(lie, i) for i, p in enumerate(truth)) > honest


@pytest.mark.parametrize("scorer", [log_loss, brier_score])
@pytest.mark.parametrize("index", [-1, 3])
def test_scoring_rules_reject_an_out_of_range_label(scorer, index):
    with pytest.raises(IndexError):
        scorer([0.5, 0.3, 0.2], index)


# -- expectation and dispersion ----------------------------------------------


def test_expectation_and_dispersion_on_an_ordered_scale():
    levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert expectation([0.0, 0.0, 1.0, 0.0, 0.0], levels) == pytest.approx(3.0)
    assert standard_deviation([0.0, 0.0, 1.0, 0.0, 0.0], levels) == pytest.approx(0.0)
    # Mass split across the two ends is the widest a 1-5 scale can be.
    assert standard_deviation([0.5, 0.0, 0.0, 0.0, 0.5], levels) == pytest.approx(2.0)


def test_expectation_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        expectation([0.5, 0.5], [1.0, 2.0, 3.0])
