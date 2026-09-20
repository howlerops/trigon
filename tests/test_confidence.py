"""Confidence is a statistic of the distribution's shape, and Score's statistic
knows its levels are ordered."""

from __future__ import annotations

import pytest

from trigon.confidence import ConfidenceMethod, choice_confidence, score_confidence


def test_uniform_is_zero_confidence_and_one_hot_is_full():
    assert choice_confidence([0.25] * 4) == pytest.approx(0.0)
    assert choice_confidence([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_is_comparable_across_cardinalities():
    """A 2-option and a 77-option answer with the same relative concentration
    should read the same, which is the reason for the log(n) scaling."""
    two = choice_confidence([0.5, 0.5])
    many = choice_confidence([1 / 77] * 77)
    assert two == pytest.approx(many, abs=1e-9)


def test_margin_ignores_the_tail_while_entropy_does_not():
    """Same option count, same top-two gap, different tails. Margin cannot
    tell them apart; entropy -- the default -- can."""
    narrow_tail = [0.5, 0.3, 0.2, 0.0]
    wide_tail = [0.5, 0.3, 0.1, 0.1]
    assert choice_confidence(narrow_tail, ConfidenceMethod.MARGIN) == pytest.approx(
        choice_confidence(wide_tail, ConfidenceMethod.MARGIN)
    )
    assert choice_confidence(narrow_tail) > choice_confidence(wide_tail)


def test_score_confidence_respects_level_order():
    """Equal entropy, very different certainty about the score."""
    anchors = [0.0, 1.0, 2.0, 3.0, 4.0]
    adjacent = [0.0, 0.5, 0.5, 0.0, 0.0]
    extremes = [0.5, 0.0, 0.0, 0.0, 0.5]
    assert score_confidence(adjacent, anchors) > 0.7
    assert score_confidence(extremes, anchors) == pytest.approx(0.0)


def test_score_confidence_uses_declared_anchor_values():
    """A 1-5 star scale and a 0-4 index scale are the same question."""
    probs = [0.0, 0.5, 0.5, 0.0, 0.0]
    assert score_confidence(probs, [1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(
        score_confidence(probs, [0.0, 1.0, 2.0, 3.0, 4.0])
    )


def test_confidence_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        score_confidence([0.5, 0.5], [0.0, 1.0, 2.0])


# -- the alternative methods, and the degenerate shapes ----------------------
#
# Coverage found MARGIN, MAX_PROBABILITY and every single-option or flat-scale
# branch untested. They are selectable on the wire, so an untested one is a
# number a caller can ask for and nobody has checked.


def test_every_choice_method_reads_zero_on_a_uniform_distribution():
    """The point of all three rescalings: uninformative means 0, not 1/n."""
    for n in (2, 4, 77):
        uniform = [1.0 / n] * n
        for method in (
            ConfidenceMethod.ENTROPY,
            ConfidenceMethod.MARGIN,
            ConfidenceMethod.MAX_PROBABILITY,
        ):
            assert choice_confidence(uniform, method) == pytest.approx(0.0, abs=1e-12)


def test_every_choice_method_reads_one_on_a_certain_distribution():
    certain = [1.0, 0.0, 0.0, 0.0]
    for method in (
        ConfidenceMethod.ENTROPY,
        ConfidenceMethod.MARGIN,
        ConfidenceMethod.MAX_PROBABILITY,
    ):
        assert choice_confidence(certain, method) == pytest.approx(1.0)


def test_margin_ignores_how_the_losing_mass_is_spread_and_entropy_does_not():
    """Documented as the reason margin is not the default."""
    concentrated = [0.5, 0.4, 0.1, 0.0]
    spread = [0.5, 0.4, 0.05, 0.05]
    assert choice_confidence(concentrated, ConfidenceMethod.MARGIN) == pytest.approx(
        choice_confidence(spread, ConfidenceMethod.MARGIN)
    )
    assert choice_confidence(concentrated, ConfidenceMethod.ENTROPY) != pytest.approx(
        choice_confidence(spread, ConfidenceMethod.ENTROPY)
    )


def test_a_single_option_is_certain_by_construction():
    """Not a real request -- types.py rejects one-option Choices -- but the
    shortlist stage can narrow to one, and it must not divide by log(1)."""
    assert choice_confidence([1.0]) == 1.0
    assert score_confidence([1.0], [3.0]) == 1.0


def test_an_empty_distribution_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="empty"):
        choice_confidence([])


def test_an_unknown_choice_method_is_rejected():
    with pytest.raises(ValueError, match="not a Choice confidence method"):
        choice_confidence([0.5, 0.5], ConfidenceMethod.DISPERSION)


def test_score_falls_back_to_the_choice_methods_when_asked():
    probs, anchors = [0.7, 0.2, 0.1], [1.0, 2.0, 3.0]
    assert score_confidence(probs, anchors, ConfidenceMethod.MARGIN) == pytest.approx(
        choice_confidence(probs, ConfidenceMethod.MARGIN)
    )


def test_a_scale_with_no_span_is_certain_rather_than_a_divide_by_zero():
    """Every level anchored to the same value: dispersion is undefined, and
    the honest answer is that the expectation cannot be anywhere else."""
    assert score_confidence([0.5, 0.5], [2.0, 2.0]) == 1.0


def test_score_rejects_anchors_that_do_not_match_the_distribution():
    with pytest.raises(ValueError, match="length mismatch"):
        score_confidence([0.5, 0.5], [1.0, 2.0, 3.0])
