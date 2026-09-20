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
