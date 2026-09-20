"""Post-hoc temperature scaling, fitted per primitive and per domain.

Temperature scaling is the cheap half of the calibration story: one parameter
per slice, fitted on held-out data by minimising NLL, which cannot change any
argmax and so cannot cost accuracy. The expensive half is the outcome-grounded
training in ``docs/training.md``; this layer catches what is left, and catches
the drift that a quantization change introduces.

Re-fit after *any* change to the serving path -- a KV bit-width change moves
probabilities well before it moves argmax, which is exactly why the release
gate is an ECE delta and not an accuracy delta.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..numeric import sigmoid, softmax

__all__ = [
    "CalibrationWarning",
    "TemperatureScaler",
    "fit_binary_temperature",
    "fit_temperature",
]


class CalibrationWarning(UserWarning):
    """Raised when a fit is degenerate rather than merely imperfect."""


# A temperature outside this range means something upstream is broken, not that
# the model needs that much flattening.
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
_TOLERANCE = 1e-4
_MAX_ITERATIONS = 200


def _nll(logits: Sequence[Sequence[float]], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for row, label in zip(logits, labels, strict=True):
        probs = softmax(row, temperature)
        total -= math.log(max(probs[label], 1e-12))
    return total / len(labels)


def _binary_nll(logits: Sequence[float], labels: Sequence[int], temperature: float) -> float:
    total = 0.0
    for logit, label in zip(logits, labels, strict=True):
        p = sigmoid(logit, temperature)
        total -= math.log(max(p if label else 1.0 - p, 1e-12))
    return total / len(labels)


def _minimize(objective, lo: float, hi: float) -> float:
    """Golden-section search over a unimodal objective.

    NLL as a function of temperature is unimodal for a fixed logit set, so a
    derivative-free line search converges in a few dozen evaluations -- which
    keeps this module free of scipy and importable inside the gateway.
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(_MAX_ITERATIONS):
        if b - a < _TOLERANCE:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = objective(d)
    return (a + b) / 2.0


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Temperature minimising NLL for a categorical head (Choice, Score)."""
    if not logits:
        raise ValueError("cannot fit a temperature on an empty set")
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logit rows against {len(labels)} labels")
    return _check_bounds(
        _minimize(lambda t: _nll(logits, labels, t), MIN_TEMPERATURE, MAX_TEMPERATURE)
    )


def fit_binary_temperature(logits: Sequence[float], labels: Sequence[int]) -> float:
    """Temperature minimising NLL for the single-logit Noul head."""
    if not logits:
        raise ValueError("cannot fit a temperature on an empty set")
    if len(logits) != len(labels):
        raise ValueError(f"{len(logits)} logits against {len(labels)} labels")
    return _check_bounds(
        _minimize(lambda t: _binary_nll(logits, labels, t), MIN_TEMPERATURE, MAX_TEMPERATURE)
    )


def _check_bounds(value: float) -> float:
    """Warn when the optimum sits on a bound.

    A temperature pinned at the ceiling means NLL kept improving as the
    distribution flattened -- the logits carry no usable signal and the fit is
    finding the uniform distribution, not calibrating anything. Pinned at the
    floor means the opposite. Either way the number should not be shipped
    without someone looking at the model first.
    """
    margin = 0.01
    if value >= MAX_TEMPERATURE - margin:
        warnings.warn(
            f"fitted temperature hit the ceiling ({MAX_TEMPERATURE}): the logits carry "
            f"little signal and this fit is flattening them toward uniform, not "
            f"calibrating them",
            CalibrationWarning,
            stacklevel=3,
        )
    elif value <= MIN_TEMPERATURE + margin:
        warnings.warn(
            f"fitted temperature hit the floor ({MIN_TEMPERATURE}): the fit is sharpening "
            f"toward a one-hot distribution, which usually means the calibration split "
            f"is too easy or too small",
            CalibrationWarning,
            stacklevel=3,
        )
    return value


@dataclass
class TemperatureScaler:
    """Per-primitive temperatures, optionally overridden per domain.

    Lookup falls back from ``(domain, primitive)`` to ``primitive`` to 1.0, so a
    deployment that has fitted only some domains still serves the rest.
    """

    primitive: dict[str, float] = field(default_factory=dict)
    domain: dict[str, dict[str, float]] = field(default_factory=dict)
    # Recorded so a report can state what the temperatures were fitted against.
    fitted_on: dict[str, int] = field(default_factory=dict)

    def temperature(self, primitive: str, domain: str | None = None) -> float:
        if domain is not None:
            by_domain = self.domain.get(domain)
            if by_domain and primitive in by_domain:
                return by_domain[primitive]
        return self.primitive.get(primitive, 1.0)

    def apply(
        self, logits: Sequence[float], primitive: str, domain: str | None = None
    ) -> list[float]:
        """Scale and softmax a categorical head."""
        return softmax(logits, self.temperature(primitive, domain))

    def apply_binary(self, logit: float, domain: str | None = None) -> float:
        """Scale and squash the Noul head."""
        return sigmoid(logit, self.temperature("noul", domain))

    def fit(
        self,
        primitive: str,
        logits: Sequence[Sequence[float]],
        labels: Sequence[int],
        domain: str | None = None,
    ) -> float:
        """Fit and store one temperature. Returns the fitted value."""
        value = fit_temperature(logits, labels)
        self._store(primitive, value, domain)
        self.fitted_on[_key(primitive, domain)] = len(labels)
        return value

    def fit_binary(
        self, logits: Sequence[float], labels: Sequence[int], domain: str | None = None
    ) -> float:
        """Fit and store the Noul temperature. Returns the fitted value."""
        value = fit_binary_temperature(logits, labels)
        self._store("noul", value, domain)
        self.fitted_on[_key("noul", domain)] = len(labels)
        return value

    def _store(self, primitive: str, value: float, domain: str | None) -> None:
        if domain is None:
            self.primitive[primitive] = value
        else:
            self.domain.setdefault(domain, {})[primitive] = value

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"primitive": self.primitive, "domain": self.domain, "fitted_on": self.fitted_on}

    @classmethod
    def from_dict(cls, payload: dict) -> TemperatureScaler:
        return cls(
            primitive=dict(payload.get("primitive", {})),
            domain={k: dict(v) for k, v in payload.get("domain", {}).items()},
            fitted_on=dict(payload.get("fitted_on", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _key(primitive: str, domain: str | None) -> str:
    return primitive if domain is None else f"{domain}/{primitive}"
