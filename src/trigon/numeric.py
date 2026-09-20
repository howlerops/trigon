"""Pure-Python numerics for the typed-output layer.

The calibration layer and the eval metrics run inside the gateway and inside
CI, neither of which should need numpy or torch. Everything here is plain
floats over plain lists, written to be numerically safe rather than fast: the
vectors are over options and levels (tens to thousands of entries), never over
model activations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "brier_score",
    "entropy",
    "expectation",
    "log_loss",
    "normalize",
    "normalized_entropy",
    "sigmoid",
    "softmax",
    "standard_deviation",
]

# Probabilities below this are treated as this value by the scoring rules, so a
# confidently wrong answer gets a large but finite penalty instead of infinity.
EPS = 1e-12


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Temperature-scaled softmax. ``temperature`` > 1 flattens, < 1 sharpens."""
    if not logits:
        raise ValueError("softmax needs at least one logit")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def sigmoid(logit: float, temperature: float = 1.0) -> float:
    """Temperature-scaled logistic, used for the single Noul logit."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    x = logit / temperature
    # Branch to keep exp() away from overflow on large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def normalize(values: Sequence[float]) -> list[float]:
    """Scale non-negative values to sum to 1."""
    if not values:
        raise ValueError("cannot normalize an empty vector")
    if any(v < 0 for v in values):
        raise ValueError("cannot normalize a vector with negative entries")
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in nats."""
    # ``+ 0.0`` normalizes the -0.0 that a one-hot distribution otherwise yields.
    return -sum(p * math.log(p) for p in probs if p > 0) + 0.0


def normalized_entropy(probs: Sequence[float]) -> float:
    """Entropy scaled to [0, 1] by the entropy of the uniform distribution.

    Scaling by ``log(n)`` is what makes the derived confidence comparable
    across questions with different option counts, which a raw entropy is not.
    """
    n = len(probs)
    if n <= 1:
        return 0.0
    return entropy(probs) / math.log(n)


def expectation(probs: Sequence[float], values: Sequence[float]) -> float:
    """Expected value of ``values`` under ``probs``."""
    if len(probs) != len(values):
        raise ValueError(f"length mismatch: {len(probs)} probs vs {len(values)} values")
    return sum(p * v for p, v in zip(probs, values, strict=True))


def standard_deviation(probs: Sequence[float], values: Sequence[float]) -> float:
    """Standard deviation of ``values`` under ``probs``."""
    mean = expectation(probs, values)
    var = sum(p * (v - mean) ** 2 for p, v in zip(probs, values, strict=True))
    return math.sqrt(max(var, 0.0))


def log_loss(probs: Sequence[float], label_index: int) -> float:
    """Negative log probability of the true label -- a proper scoring rule."""
    if not 0 <= label_index < len(probs):
        raise IndexError(f"label_index {label_index} out of range for {len(probs)} classes")
    return -math.log(max(probs[label_index], EPS))


def brier_score(probs: Sequence[float], label_index: int) -> float:
    """Multiclass Brier score: squared error against the one-hot label."""
    if not 0 <= label_index < len(probs):
        raise IndexError(f"label_index {label_index} out of range for {len(probs)} classes")
    return sum((p - (1.0 if i == label_index else 0.0)) ** 2 for i, p in enumerate(probs))
