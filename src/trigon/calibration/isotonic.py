"""Isotonic calibration: the calibrator that is not one-parameter.

Temperature scaling sharpens or flattens a distribution everywhere at once. A
head whose confidence is too high where it is confident and too low where it is
not — a *tilt* — has no correct temperature, and the fitter still returns its
best one. `reports/sweeps/README.md` has a live example: one seed's Choice head
scores ECE 0.0879 under four different rules about whether to apply a
temperature, because none of them is deciding about something that can help it.

Isotonic regression fits an arbitrary non-decreasing map from stated confidence
to observed accuracy, so a tilt is exactly the thing it can correct. It buys
that with a weaker guarantee: temperature scaling cannot reorder options and
this cannot introduce new confidence values between training points, so it
needs more data and it can overfit a small calibration split. Both calibrators
therefore stay, and `scripts/decline_rule.py` is how you find out which one a
given head wants.

**What it calibrates.** The top-1 confidence, which is what ECE and the
reliability diagram are computed from. The rest of the distribution is
rescaled proportionally so it still sums to one and the ranking is unchanged —
a calibrator that reorders a Choice's options would be changing the answer,
not the confidence in it.

Pure Python and dependency-free, like the rest of `trigon.calibration`, so the
gateway and CI can import it without a tensor library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: Below this many labelled answers, a fitted isotonic map is memorising rather
#: than calibrating: with k points it can place k steps. Temperature scaling
#: fits one number and is the right tool on a small split.
MIN_ISOTONIC_SAMPLES = 400


def _pav(xs: Sequence[float], ys: Sequence[float]) -> list[tuple[float, float]]:
    """Pool adjacent violators: the non-decreasing least-squares fit.

    Returns ``(x, value)`` knots. Blocks are merged while the running mean
    would decrease, which is the whole algorithm — the result is the unique
    non-decreasing step function minimising squared error.
    """
    # (sum of y, count, largest x in the block)
    blocks: list[list[float]] = []
    for x, y in zip(xs, ys, strict=True):
        blocks.append([y, 1.0, x])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count, right = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
            blocks[-1][2] = right
    return [(right, total / count) for total, count, right in blocks]


@dataclass
class IsotonicCalibrator:
    """A monotone confidence map per primitive, fitted on held-out answers."""

    knots: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    fitted_on: dict[str, int] = field(default_factory=dict)

    def fit(self, primitive: str, confidences: Sequence[float], correct: Sequence[int]) -> int:
        """Fit one primitive's map. Returns the number of answers used."""
        if len(confidences) != len(correct):
            raise ValueError("confidences and outcomes differ in length")
        if len(confidences) < MIN_ISOTONIC_SAMPLES:
            raise ValueError(
                f"isotonic calibration needs at least {MIN_ISOTONIC_SAMPLES} answers, "
                f"got {len(confidences)}; fit a temperature instead"
            )
        order = sorted(range(len(confidences)), key=lambda i: confidences[i])
        self.knots[primitive] = _pav(
            [confidences[i] for i in order], [float(correct[i]) for i in order]
        )
        self.fitted_on[primitive] = len(confidences)
        return len(confidences)

    def confidence(self, primitive: str, value: float) -> float:
        """The calibrated confidence for a stated one. Identity if unfitted."""
        knots = self.knots.get(primitive)
        if not knots:
            return value
        # Piecewise constant, clamped at both ends: outside the fitted range
        # there is no evidence, and extrapolating a step function invents some.
        previous = knots[0][1]
        for right, fitted in knots:
            if value <= right:
                return fitted
            previous = fitted
        return previous

    def apply(self, primitive: str, probabilities: Sequence[float]) -> list[float]:
        """Rescale a distribution so its top-1 confidence is the calibrated one.

        The remaining mass is spread over the other options in their existing
        proportions, so the ranking is untouched. A calibrator that reordered a
        Choice's options would be changing the answer rather than the
        confidence in it.
        """
        probabilities = list(probabilities)
        if not probabilities:
            return probabilities
        top = max(range(len(probabilities)), key=lambda i: probabilities[i])
        target = min(max(self.confidence(primitive, probabilities[top]), 1e-6), 1.0 - 1e-6)
        rest = sum(p for i, p in enumerate(probabilities) if i != top)
        if rest <= 0.0:
            # A degenerate distribution has nothing to redistribute into;
            # spread the remainder evenly rather than divide by zero.
            share = (1.0 - target) / max(len(probabilities) - 1, 1)
            return [target if i == top else share for i in range(len(probabilities))]
        scale = (1.0 - target) / rest
        return [target if i == top else p * scale for i, p in enumerate(probabilities)]

    def to_dict(self) -> dict:
        return {
            "knots": {k: [list(pair) for pair in v] for k, v in sorted(self.knots.items())},
            "fitted_on": dict(sorted(self.fitted_on.items())),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibrator:
        payload = json.loads(Path(path).read_text())
        return cls(
            knots={k: [(a, b) for a, b in v] for k, v in payload.get("knots", {}).items()},
            fitted_on=dict(payload.get("fitted_on", {})),
        )
