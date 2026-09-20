"""Confidence is a serving-layer statistic, not a model output.

The model emits a distribution; confidence summarises how concentrated that
distribution is. Keeping it out of the model matters for two reasons: a learned
"confidence" head would need its own calibration story, and callers who
disagree with our summary can compute their own from the probabilities, which
every answer returns in full.

Two statistics, because Choice and Score are different shapes:

* Choice is unordered, so concentration means low entropy. We report
  ``1 - H(p)/log(n)``, scaled by ``log(n)`` so a confident 2-option answer and a
  confident 77-option answer read the same number.
* Score is *ordered*, and entropy throws that away: mass split between
  "neutral" and "slightly positive" is a near-certain score, while the same
  entropy split between "terrible" and "excellent" is not. So Score confidence
  is ``1 - sd(p)/sd_max``, the dispersion of the score itself.

Noul gets no confidence field at all: for a binary question the probability is
already the whole answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from .numeric import normalized_entropy, standard_deviation

__all__ = ["ConfidenceMethod", "choice_confidence", "margin_confidence", "score_confidence"]


class ConfidenceMethod(str, Enum):
    """Selectable for ablations; ``ENTROPY`` and ``DISPERSION`` are the defaults."""

    ENTROPY = "entropy"
    MARGIN = "margin"
    MAX_PROBABILITY = "max_probability"
    DISPERSION = "dispersion"


def choice_confidence(
    probs: Sequence[float], method: ConfidenceMethod = ConfidenceMethod.ENTROPY
) -> float:
    """Concentration of an unordered distribution, in [0, 1]."""
    if not probs:
        raise ValueError("cannot score confidence for an empty distribution")
    if len(probs) == 1:
        return 1.0
    if method is ConfidenceMethod.ENTROPY:
        return _clamp(1.0 - normalized_entropy(probs))
    if method is ConfidenceMethod.MARGIN:
        return margin_confidence(probs)
    if method is ConfidenceMethod.MAX_PROBABILITY:
        # Rescaled so the uniform distribution reads 0 rather than 1/n.
        n = len(probs)
        return _clamp((max(probs) - 1.0 / n) / (1.0 - 1.0 / n))
    raise ValueError(f"{method} is not a Choice confidence method")


def margin_confidence(probs: Sequence[float]) -> float:
    """Gap between the top two probabilities.

    Reads more naturally than entropy when the question is "is the winner
    safe", and it ignores how the remaining mass is spread -- which is the
    reason it is not the default.
    """
    if len(probs) < 2:
        return 1.0
    top, second = sorted(probs, reverse=True)[:2]
    return _clamp(top - second)


def score_confidence(
    probs: Sequence[float],
    anchors: Sequence[float],
    method: ConfidenceMethod = ConfidenceMethod.DISPERSION,
) -> float:
    """Concentration of an *ordered* distribution, in [0, 1].

    ``sd_max`` is the dispersion of the worst case on this scale: half the mass
    on the lowest level and half on the highest, i.e. ``(max - min) / 2``.
    """
    if len(probs) != len(anchors):
        raise ValueError(f"length mismatch: {len(probs)} probs vs {len(anchors)} anchors")
    if len(probs) < 2:
        return 1.0
    if method is not ConfidenceMethod.DISPERSION:
        return choice_confidence(probs, method)
    span = max(anchors) - min(anchors)
    if span <= 0:
        return 1.0
    return _clamp(1.0 - standard_deviation(probs, anchors) / (span / 2.0))


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))
