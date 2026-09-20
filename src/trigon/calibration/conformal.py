"""Split-conformal wrappers, so users can get a coverage guarantee on their own
data without retraining anything.

This is the mitigation for the biggest risk in the build plan: outcome-grounded
calibration fitted on public and synthetic data may not transfer to a user's
domain. A conformal wrapper needs only a few hundred labelled examples from
that domain and converts our probabilities into prediction sets whose coverage
is guaranteed at ``1 - alpha`` regardless of whether the underlying model is
well calibrated -- the guarantee is distribution-free and finite-sample, and it
holds under exchangeability alone.

Two scores, both standard:

* ``LAC`` (least-ambiguous set-valued classifier) thresholds ``1 - p[y]``. It
  gives the smallest sets at the target coverage, but its coverage is only
  marginal -- averaged over all inputs, not per class.
* ``APS`` (adaptive prediction sets) accumulates probability mass in descending
  order. Larger sets, but coverage tracks difficulty, which is what you want
  when the question is hard for some inputs and trivial for others. Without
  the randomized correction it is conservative: on a genuinely uncertain
  problem at a tight ``alpha`` the threshold can reach 1.0, at which point
  every set is the full label list. That is the correct answer -- the data does
  not support a smaller set at that coverage -- but it is worth recognising as
  a signal to loosen ``alpha`` rather than as a bug.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "ConformalMethod",
    "ConformalPredictor",
    "PredictionSet",
    "fit_conformal",
]


class ConformalMethod(str, Enum):
    LAC = "lac"
    APS = "aps"


@dataclass(frozen=True)
class PredictionSet:
    """The labels that survive at the wrapper's target coverage."""

    indices: tuple[int, ...]
    labels: tuple[str, ...]
    alpha: float
    method: ConformalMethod

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def is_singleton(self) -> bool:
        """A singleton set is the useful case: one answer, with a guarantee."""
        return self.size == 1

    @property
    def is_empty(self) -> bool:
        """LAC can return nothing when every label scores below threshold. The
        caller should read that as "abstain", not as "no answer exists"."""
        return self.size == 0


def _lac_scores(probs: Sequence[float], label: int) -> float:
    return 1.0 - probs[label]


def _aps_scores(probs: Sequence[float], label: int) -> float:
    """Mass accumulated in descending order up to and including the true label."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    total = 0.0
    for idx in order:
        total += probs[idx]
        if idx == label:
            return total
    raise IndexError(f"label {label} not found among {len(probs)} classes")


def fit_conformal(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    alpha: float = 0.1,
    method: ConformalMethod = ConformalMethod.LAC,
) -> ConformalPredictor:
    """Fit a wrapper on a held-out calibration split.

    The quantile index uses the finite-sample correction ``ceil((n+1)(1-alpha))``
    -- without it coverage is short by roughly ``1/n``, which matters at the few
    hundred examples a user is realistically going to label.
    """
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} predictions against {len(labels)} labels")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(labels)
    if n < math.ceil(1.0 / alpha) - 1:
        raise ValueError(
            f"need at least {math.ceil(1.0 / alpha) - 1} calibration examples for "
            f"alpha={alpha}, got {n}"
        )

    score_fn = _lac_scores if method is ConformalMethod.LAC else _aps_scores
    scores = sorted(score_fn(p, y) for p, y in zip(probs, labels, strict=True))
    rank = math.ceil((n + 1) * (1.0 - alpha))
    threshold = scores[min(rank, n) - 1]
    return ConformalPredictor(alpha=alpha, method=method, threshold=threshold, calibration_n=n)


@dataclass(frozen=True)
class ConformalPredictor:
    """A fitted wrapper. Cheap to store: one threshold per profile."""

    alpha: float
    method: ConformalMethod
    threshold: float
    calibration_n: int

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def predict(self, probs: Sequence[float], labels: Sequence[str] | None = None) -> PredictionSet:
        """Prediction set for one distribution."""
        if self.method is ConformalMethod.LAC:
            keep = [i for i, p in enumerate(probs) if 1.0 - p <= self.threshold]
        else:
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            keep, total = [], 0.0
            for idx in order:
                keep.append(idx)
                total += probs[idx]
                if total >= self.threshold:
                    break
            keep.sort()
        names = tuple(labels[i] for i in keep) if labels is not None else ()
        return PredictionSet(
            indices=tuple(keep), labels=names, alpha=self.alpha, method=self.method
        )

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "method": self.method.value,
            "threshold": self.threshold,
            "calibration_n": self.calibration_n,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ConformalPredictor:
        return cls(
            alpha=float(payload["alpha"]),
            method=ConformalMethod(payload["method"]),
            threshold=float(payload["threshold"]),
            calibration_n=int(payload["calibration_n"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> ConformalPredictor:
        return cls.from_dict(json.loads(Path(path).read_text()))
