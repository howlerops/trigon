"""Calibration metrics. These are the release gates, so they are unit-tested
against hand-computed cases rather than against another library.

ECE is reported two ways on purpose. Equal-width binning is what everyone
publishes, so it is what makes our number comparable; it is also the estimator
that flatters a model whose predictions pile up inside one bin. Equal-mass
binning (``adaptive_calibration_error``) is the honest one. Publish both.

Two things about ECE that a release gate has to respect, because neither is
visible in the number itself:

* **Both estimators are biased upward at small n.** A bin of ``m`` samples has
  sampling noise of order ``sqrt(p(1-p)/m)`` in its accuracy, and ECE averages
  the *absolute* gap, so noise cannot cancel -- it accumulates. A perfectly
  calibrated model scored on a few hundred examples will report a non-zero ECE.
  Gate on a fixed, stated sample size and compare like with like.
* **Equal-mass bins must not split ties.** Sorting by confidence and cutting
  into equal-sized groups will slice through a run of identical confidences,
  manufacturing gaps out of nothing; a constant predictor -- calibrated by
  definition when its constant is the base rate -- would score as badly
  miscalibrated. ``_split_evenly`` therefore extends each bin to the end of the
  current run of equal confidences.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from ..numeric import brier_score, log_loss

__all__ = [
    "CalibrationReport",
    "ReliabilityBin",
    "adaptive_calibration_error",
    "brier",
    "coverage",
    "expected_calibration_error",
    "maximum_calibration_error",
    "mean_set_size",
    "negative_log_likelihood",
    "noise_floor",
    "reliability_bins",
    "report",
]

DEFAULT_BINS = 15


@dataclass(frozen=True)
class ReliabilityBin:
    """One point of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed miscalibration. Positive means overconfident."""
        return self.mean_confidence - self.accuracy


@dataclass(frozen=True)
class NoiseFloor:
    """What a perfectly calibrated model would have scored on this data.

    Simulated by drawing labels from the model's own predicted distributions:
    a model whose labels come from its own probabilities is calibrated by
    construction, so any ECE it still reports is estimator bias and sampling
    noise, not miscalibration. Comparing the measured ECE against this is what
    turns the number into evidence.
    """

    mean: float
    p95: float
    trials: int
    n: int


def noise_floor(
    probs: Sequence[Sequence[float]],
    n_bins: int = DEFAULT_BINS,
    trials: int = 200,
    seed: int = 0,
    *,
    equal_mass: bool = False,
) -> NoiseFloor:
    """The ECE a perfectly calibrated model would report on these predictions.

    Labels are resampled from each prediction, so the simulated model is
    calibrated by definition; the spread of its ECE is the floor beneath which
    a measured ECE says nothing. This exists because the failure it prevents
    has already happened in public: an independent re-analysis found published
    ECE figures of 0.0505-0.0712 at n=60 being offered as evidence of good
    calibration, when at that sample size the same figures are also what
    serious miscalibration looks like.
    """
    if not probs:
        raise ValueError("no predictions to simulate")
    if trials < 2:
        raise ValueError(f"need at least 2 trials, got {trials}")
    rng = random.Random(seed)
    indices = [list(range(len(row))) for row in probs]
    scores = []
    for _ in range(trials):
        labels = [
            rng.choices(index, weights=row)[0] for index, row in zip(indices, probs, strict=True)
        ]
        estimator = adaptive_calibration_error if equal_mass else expected_calibration_error
        scores.append(estimator(probs, labels, n_bins))
    scores.sort()
    rank = max(0, min(len(scores) - 1, int(round(0.95 * (len(scores) - 1)))))
    return NoiseFloor(mean=sum(scores) / len(scores), p95=scores[rank], trials=trials, n=len(probs))


@dataclass(frozen=True)
class CalibrationReport:
    """Everything the release gate looks at, for one slice of eval data."""

    n: int
    ece: float
    adaptive_ece: float
    mce: float
    brier: float
    nll: float
    accuracy: float
    mean_confidence: float
    bins: tuple[ReliabilityBin, ...] = field(default=())
    slice_name: str = "all"
    #: What a perfectly calibrated model would have scored here. ``None`` when
    #: the report was built without simulating it.
    floor: NoiseFloor | None = None

    @property
    def distinguishable(self) -> bool | None:
        """Whether the measured ECE is separable from a calibrated model's.

        ``False`` is not a failure -- it is the strongest claim available: this
        model is statistically indistinguishable from perfectly calibrated at
        this sample size. ``True`` means the miscalibration is real. ``None``
        means no floor was simulated, and the ECE should not be published as
        evidence either way.
        """
        if self.floor is None:
            return None
        return self.ece > self.floor.p95

    @property
    def overconfidence(self) -> float:
        """Mean confidence minus accuracy. The sign that matters for RLHF'd
        baselines, which are overconfident rather than merely miscalibrated."""
        return self.mean_confidence - self.accuracy

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bins"] = [asdict(b) for b in self.bins]
        out["overconfidence"] = self.overconfidence
        out["distinguishable"] = self.distinguishable
        return out


def _check(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} predictions against {len(labels)} labels")
    if not probs:
        raise ValueError("no predictions to score")
    for i, (p, y) in enumerate(zip(probs, labels, strict=True)):
        if not 0 <= y < len(p):
            raise IndexError(f"label {y} out of range for {len(p)} classes at index {i}")
        total = sum(p)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"prediction {i} sums to {total}, not 1")


def reliability_bins(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    *,
    equal_mass: bool = False,
) -> list[ReliabilityBin]:
    """Bin top-label confidence and compare it to accuracy within each bin."""
    _check(probs, labels)
    points = []
    for p, y in zip(probs, labels, strict=True):
        top = max(range(len(p)), key=lambda i: p[i])
        points.append((p[top], 1.0 if top == y else 0.0))

    if equal_mass:
        points.sort(key=lambda t: t[0])
        groups = _split_evenly(points, n_bins)
        edges = [(g[0][0], g[-1][0]) for g in groups]
    else:
        width = 1.0 / n_bins
        groups = [[] for _ in range(n_bins)]
        for conf, correct in points:
            idx = min(int(conf / width), n_bins - 1)
            groups[idx].append((conf, correct))
        edges = [(i * width, (i + 1) * width) for i in range(n_bins)]

    out = []
    for (lower, upper), group in zip(edges, groups, strict=True):
        if not group:
            out.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0))
            continue
        out.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(group),
                mean_confidence=sum(c for c, _ in group) / len(group),
                accuracy=sum(a for _, a in group) / len(group),
            )
        )
    return out


def _split_evenly(points: list, n_bins: int) -> list[list]:
    """Split a confidence-sorted list into groups of near-equal size, never
    cutting through a run of identical confidences.

    Without the tie handling a constant predictor scores as miscalibrated: its
    single confidence value gets scattered across bins whose accuracies then
    differ by sampling noise alone.
    """
    n = len(points)
    if n == 0:
        return []
    n_bins = max(1, min(n_bins, n))
    target = n / n_bins
    groups: list[list] = []
    start = 0
    for i in range(n_bins):
        if start >= n:
            break
        end = min(n, int(round((i + 1) * target)))
        end = max(end, start + 1)
        # Extend past the boundary while the next point has the same
        # confidence as the last one inside it.
        while end < n and points[end][0] == points[end - 1][0]:
            end += 1
        groups.append(points[start:end])
        start = end
    if start < n:
        groups[-1].extend(points[start:])
    return groups


def _binned_error(bins: Sequence[ReliabilityBin], total: int, *, worst: bool = False) -> float:
    gaps = [(b.count, abs(b.gap)) for b in bins if b.count]
    if not gaps:
        return 0.0
    if worst:
        return max(g for _, g in gaps)
    return sum(count * gap for count, gap in gaps) / total


def expected_calibration_error(
    probs: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = DEFAULT_BINS
) -> float:
    """Equal-width ECE over top-label confidence."""
    bins = reliability_bins(probs, labels, n_bins)
    return _binned_error(bins, len(labels))


def adaptive_calibration_error(
    probs: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = DEFAULT_BINS
) -> float:
    """Equal-mass ECE. Does not hide miscalibration in an over-full bin."""
    bins = reliability_bins(probs, labels, n_bins, equal_mass=True)
    return _binned_error(bins, len(labels))


def maximum_calibration_error(
    probs: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = DEFAULT_BINS
) -> float:
    """Worst single-bin gap -- the number a risk owner asks for."""
    bins = reliability_bins(probs, labels, n_bins)
    return _binned_error(bins, len(labels), worst=True)


def brier(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean multiclass Brier score. A proper scoring rule, unlike ECE."""
    _check(probs, labels)
    return sum(brier_score(p, y) for p, y in zip(probs, labels, strict=True)) / len(labels)


def negative_log_likelihood(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean negative log probability of the true label."""
    _check(probs, labels)
    return sum(log_loss(p, y) for p, y in zip(probs, labels, strict=True)) / len(labels)


def report(
    probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    n_bins: int = DEFAULT_BINS,
    slice_name: str = "all",
    *,
    simulate_floor: bool = True,
    trials: int = 200,
) -> CalibrationReport:
    """Every gate metric for one slice, plus the reliability diagram points.

    The noise floor is simulated by default. It costs a few hundred resamples
    and it is the difference between publishing a number and publishing
    evidence, so opting out is deliberate rather than the default.
    """
    _check(probs, labels)
    bins = reliability_bins(probs, labels, n_bins)
    correct, confidence = 0, 0.0
    for p, y in zip(probs, labels, strict=True):
        top = max(range(len(p)), key=lambda i: p[i])
        correct += int(top == y)
        confidence += p[top]
    n = len(labels)
    return CalibrationReport(
        n=n,
        ece=_binned_error(bins, n),
        adaptive_ece=adaptive_calibration_error(probs, labels, n_bins),
        mce=_binned_error(bins, n, worst=True),
        brier=brier(probs, labels),
        nll=negative_log_likelihood(probs, labels),
        accuracy=correct / n,
        mean_confidence=confidence / n,
        bins=tuple(bins),
        slice_name=slice_name,
        floor=noise_floor(probs, n_bins, trials) if simulate_floor else None,
    )


def coverage(prediction_sets: Sequence[Sequence[int]], labels: Sequence[int]) -> float:
    """Fraction of conformal prediction sets that contain the true label."""
    if len(prediction_sets) != len(labels):
        raise ValueError("prediction sets and labels differ in length")
    if not labels:
        raise ValueError("no predictions to score")
    return sum(1 for s, y in zip(prediction_sets, labels, strict=True) if y in set(s)) / len(labels)


def mean_set_size(prediction_sets: Sequence[Sequence[int]]) -> float:
    """Average conformal set size. Coverage is free if sets may be huge."""
    if not prediction_sets:
        raise ValueError("no prediction sets")
    return sum(len(s) for s in prediction_sets) / len(prediction_sets)
