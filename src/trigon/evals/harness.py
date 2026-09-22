"""The eval runner. Cases in, aligned distributions out, metrics on top.

Every suite in this package -- calibration, jaggedness, workflow -- runs
through this one path, so a number from one suite means the same thing as a
number from another. The runner is backend-agnostic by construction: it talks
to an ``Engine``, so a prompted LLM baseline, the lexical floor and a trained
readout model are all measured identically. That is the phase-0 exit criterion
and the thing that makes the published comparison reproducible.

The runner records latency per case as well as accuracy, because a Pareto plot
without a load-tested latency axis is the marketing number this project exists
to replace.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..calibration.metrics import CalibrationReport, report
from ..engine import Engine
from ..types import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

__all__ = [
    "Case",
    "CaseOutcome",
    "Expectation",
    "QuestionAccuracy",
    "QuestionOutcome",
    "SuiteResult",
    "run_cases",
    "summarize",
]


@dataclass(frozen=True)
class Expectation:
    """Ground truth for one question of one case.

    ``label`` is the index of the true option or level; ``probability`` is the
    true probability for a Noul. ``distribution`` carries an annotator
    distribution for subjective questions that have no resolvable outcome --
    training and scoring against the distribution rather than the majority vote
    is the whole point of that data stream.
    """

    label: int | None = None
    probability: float | None = None
    distribution: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.label is None and self.probability is None and self.distribution is None:
            raise ValueError("an expectation needs a label, a probability or a distribution")

    @property
    def hard_label(self) -> int | None:
        """The index a hard-accuracy metric should compare against."""
        if self.label is not None:
            return self.label
        if self.distribution is not None:
            return max(range(len(self.distribution)), key=lambda i: self.distribution[i])
        if self.probability is not None:
            return int(self.probability >= 0.5)
        return None


@dataclass(frozen=True)
class Case:
    """One request plus what the right answers were."""

    case_id: str
    request: SystemOneRequest
    expected: dict[str, Expectation] = field(default_factory=dict)
    domain: str = "general"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionOutcome:
    """One question's predicted distribution, aligned to declared label order."""

    question_id: str
    primitive: str
    labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    confidence: float | None
    expected: Expectation | None

    @property
    def predicted_index(self) -> int:
        return max(range(len(self.probabilities)), key=lambda i: self.probabilities[i])

    @property
    def correct(self) -> bool | None:
        if self.expected is None:
            return None
        truth = self.expected.hard_label
        return None if truth is None else self.predicted_index == truth


@dataclass(frozen=True)
class CaseOutcome:
    case: Case
    response: SystemOneResponse
    questions: dict[str, QuestionOutcome]
    latency_ms: float

    @property
    def prefill_tokens(self) -> int:
        return self.response.usage.prefill_tokens


@dataclass(frozen=True)
class QuestionAccuracy:
    """One question's accuracy against its own marginal predictor.

    The pooled lift averages over questions, which lets a model that has
    learned one question of three clear a gate while ignoring the state on the
    other two. Per question, that is visible.
    """

    question_id: str
    n: int
    accuracy: float
    baseline_accuracy: float

    @property
    def lift(self) -> float:
        return self.accuracy - self.baseline_accuracy

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "n": self.n,
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "lift": self.lift,
        }


@dataclass(frozen=True)
class SuiteResult:
    """What a suite run produced, ready to serialize into a release report."""

    suite: str
    model: str
    n_cases: int
    n_questions: int
    accuracy: float | None
    #: Accuracy of the marginal predictor -- the one that ignores the state and
    #: always answers each question with its most frequent label. The number a
    #: model has to beat before its calibration means anything.
    baseline_accuracy: float | None
    calibration: CalibrationReport | None
    latency_p50_ms: float
    latency_p99_ms: float
    mean_prefill_tokens: float
    #: Accuracy and marginal-predictor accuracy for each question id. The
    #: pooled figures above average over questions, so a model that answers one
    #: question of three and ignores the state on the rest still clears a
    #: pooled lift gate -- the reference run does exactly that. This is the
    #: breakdown that shows it.
    per_question: dict[str, QuestionAccuracy] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)
    per_primitive: dict[str, CalibrationReport] = field(default_factory=dict)

    @property
    def worst_question_lift(self) -> float | None:
        """The smallest lift over baseline of any single question.

        The pooled lift is an average and hides a question answered by rote.
        This is the number a per-question gate reads.
        """
        lifts = [q.lift for q in self.per_question.values()]
        return min(lifts) if lifts else None

    @property
    def lift_over_baseline(self) -> float | None:
        """How much the model beats the marginal predictor by."""
        if self.accuracy is None or self.baseline_accuracy is None:
            return None
        return self.accuracy - self.baseline_accuracy

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "model": self.model,
            "n_cases": self.n_cases,
            "n_questions": self.n_questions,
            "accuracy": self.accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "lift_over_baseline": self.lift_over_baseline,
            "worst_question_lift": self.worst_question_lift,
            "per_question": {k: v.to_dict() for k, v in self.per_question.items()},
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "mean_prefill_tokens": self.mean_prefill_tokens,
            "extra": self.extra,
            "per_primitive": {k: v.to_dict() for k, v in self.per_primitive.items()},
        }


def run_cases(engine: Engine, cases: Iterable[Case], batch_size: int = 1) -> list[CaseOutcome]:
    """Run every case through one engine, recording answers and latency.

    **Serial by default, because batching is slower here.** It was written
    expecting the opposite -- `docs/next.md` B.2 called the batching layer
    "where the latency story is won or lost" -- and measurement on an idle
    machine says the forward pass costs 3.43 ms per request alone and 4.14 ms
    per request in a batch of 16. `reports/batching/README.md` has the numbers
    and the reason: batching fills parallel capacity that a single small
    request leaves idle, and on one CPU thread there is none to fill.

    The batched path is kept, tested and correct because it is the right shape
    for a GPU, which this project does not have. `batch_size > 1` opts in.

    **The latency reported under batching is per request, not per batch.** A
    batch's wall clock shared out is what a request cost when the work was
    coalesced; it is not the p50 a caller sees on a gateway answering one
    request, so mixing the two would understate serving latency.
    """
    cases = list(cases)
    if batch_size > 1 and getattr(engine.backend, "infer_many", None) is not None:
        started = time.perf_counter()
        responses = engine.answer_many([case.request for case in cases], batch_size=batch_size)
        shared = (time.perf_counter() - started) * 1000.0 / max(1, len(cases))
        return [
            CaseOutcome(
                case=case,
                response=response,
                questions=_align(case, response),
                latency_ms=shared,
            )
            for case, response in zip(cases, responses, strict=True)
        ]

    outcomes: list[CaseOutcome] = []
    for case in cases:
        started = time.perf_counter()
        response = engine.answer(case.request)
        latency = (time.perf_counter() - started) * 1000.0
        outcomes.append(
            CaseOutcome(
                case=case,
                response=response,
                questions=_align(case, response),
                latency_ms=latency,
            )
        )
    return outcomes


def _align(case: Case, response: SystemOneResponse) -> dict[str, QuestionOutcome]:
    """Put answers back in the caller's declared label order.

    Metrics index into these vectors, so an ordering mistake here would silently
    corrupt every number downstream; the labels travel alongside for that reason.
    """
    aligned: dict[str, QuestionOutcome] = {}
    for qid, question in case.request.questions.items():
        answer = response.answers[qid]
        if isinstance(question, NoulQuestion):
            assert isinstance(answer, NoulAnswer)
            labels = ("no", "yes")
            probs = (1.0 - answer.probability, answer.probability)
            confidence = None
            primitive = "noul"
        else:
            assert isinstance(answer, (ChoiceAnswer, ScoreAnswer))
            labels = tuple(question.names)
            probs = tuple(answer.probabilities[name] for name in labels)
            confidence = answer.confidence
            primitive = "choice" if isinstance(question, ChoiceQuestion) else "score"
        aligned[qid] = QuestionOutcome(
            question_id=qid,
            primitive=primitive,
            labels=labels,
            probabilities=probs,
            confidence=confidence,
            expected=case.expected.get(qid),
        )
    return aligned


def summarize(
    suite: str,
    model: str,
    outcomes: Sequence[CaseOutcome],
    extra: dict[str, float] | None = None,
    *,
    simulate_floor: bool = True,
    floor_trials: int = 200,
) -> SuiteResult:
    """Collapse outcomes into the numbers a release report prints.

    ``simulate_floor`` costs a few hundred resamples and is what makes a
    published ECE evidence rather than a number; leave it on for anything that
    will be quoted. Suites whose headline is not calibration (jaggedness) turn
    it off, and CI turns the trial count down.
    """
    if not outcomes:
        raise ValueError(f"suite {suite!r} produced no outcomes")

    scored: list[tuple[str, tuple[float, ...], int]] = []
    labels_by_question: dict[str, list[int]] = {}
    hits_by_question: dict[str, int] = {}
    n_questions = 0
    for outcome in outcomes:
        for qid, question in outcome.questions.items():
            n_questions += 1
            if question.expected is None:
                continue
            truth = question.expected.hard_label
            if truth is not None:
                scored.append((question.primitive, question.probabilities, truth))
                labels_by_question.setdefault(qid, []).append(truth)
                probabilities = question.probabilities
                predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
                hits_by_question[qid] = hits_by_question.get(qid, 0) + int(predicted == truth)

    latencies = sorted(o.latency_ms for o in outcomes)
    calibration = (
        report(
            [p for _, p, _ in scored],
            [y for _, _, y in scored],
            slice_name=suite,
            simulate_floor=simulate_floor,
            trials=floor_trials,
        )
        if scored
        else None
    )
    per_primitive = {}
    for primitive in {p for p, _, _ in scored}:
        rows = [(p, y) for prim, p, y in scored if prim == primitive]
        per_primitive[primitive] = report(
            [p for p, _ in rows],
            [y for _, y in rows],
            slice_name=primitive,
            # Per-primitive slices are diagnostics, not published gates.
            simulate_floor=False,
        )

    return SuiteResult(
        suite=suite,
        model=model,
        n_cases=len(outcomes),
        n_questions=n_questions,
        accuracy=calibration.accuracy if calibration else None,
        baseline_accuracy=_marginal_accuracy(labels_by_question),
        per_question=_per_question(labels_by_question, hits_by_question),
        calibration=calibration,
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p99_ms=_percentile(latencies, 0.99),
        mean_prefill_tokens=statistics.fmean(o.prefill_tokens for o in outcomes),
        extra=dict(extra or {}),
        per_primitive=per_primitive,
    )


def _per_question(
    labels_by_question: dict[str, list[int]], hits_by_question: dict[str, int]
) -> dict[str, QuestionAccuracy]:
    """Model and marginal-predictor accuracy, question by question."""
    out: dict[str, QuestionAccuracy] = {}
    for qid, labels in sorted(labels_by_question.items()):
        if not labels:
            continue
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        out[qid] = QuestionAccuracy(
            question_id=qid,
            n=len(labels),
            accuracy=hits_by_question.get(qid, 0) / len(labels),
            baseline_accuracy=max(counts.values()) / len(labels),
        )
    return out


def _marginal_accuracy(labels_by_question: dict[str, list[int]]) -> float | None:
    """Accuracy of the predictor that ignores the state entirely.

    Per question, always answer its most frequent label. This is the model that
    a calibration gate alone cannot reject -- reporting true marginals is
    perfectly calibrated -- so it is the number a run has to beat before its
    ECE is worth reading.
    """
    if not labels_by_question:
        return None
    right = total = 0
    for labels in labels_by_question.values():
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        right += max(counts.values())
        total += len(labels)
    return right / total if total else None


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. Exact on small samples, unlike interpolation."""
    if not sorted_values:
        raise ValueError("no values")
    import math

    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]
