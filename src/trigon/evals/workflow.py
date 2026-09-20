"""Workflow evals: fixed compute graphs, scored against outcomes and cost.

A workflow is a small DAG of decision steps over one state, where later steps
depend on earlier answers -- the shape real pipelines actually have. Two things
this module does that the methodology it borrows from does not:

* scoring against **resolved outcomes**, not against a frontier ensemble's
  probabilities. Agreement-with-frontier scoring is vendor-graded: when the
  reference and the candidate share an error, the error is invisible. Reference
  agreement is still reported, as ``reference_kl``, because it is comparable
  with published numbers -- it is just not the headline.
* reporting cost and latency on the same run, so a result is a point on a
  Pareto plot rather than a percentage with no denominator.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..engine import Engine
from ..types import Question, SystemOneRequest, SystemOneResponse

__all__ = ["Workflow", "WorkflowCase", "WorkflowResult", "WorkflowStep", "run_workflow"]


@dataclass(frozen=True)
class WorkflowStep:
    """One node of the graph.

    ``build`` receives the state and the answers so far and returns the
    questions for this step, or ``None`` to skip -- which is how a graph
    expresses "only ask about refunds if this is a billing ticket".
    """

    name: str
    build: Callable[[object, dict[str, object]], dict[str, Question] | None]


@dataclass(frozen=True)
class WorkflowCase:
    """One input, and the decision reality eventually recorded."""

    case_id: str
    state: object
    # The final decision that actually happened, keyed by question id.
    outcome: dict[str, str]
    # Optional frontier-ensemble probabilities per question, for the
    # agreement metric that makes our numbers comparable with published ones.
    reference: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Workflow:
    name: str
    steps: tuple[WorkflowStep, ...]


@dataclass(frozen=True)
class WorkflowResult:
    workflow: str
    model: str
    n_cases: int
    outcome_accuracy: float | None
    reference_kl: float | None
    mean_prefill_tokens: float
    mean_model_calls: float
    latency_p50_ms: float
    latency_p99_ms: float

    def to_dict(self) -> dict:
        return {
            "workflow": self.workflow,
            "model": self.model,
            "n_cases": self.n_cases,
            "outcome_accuracy": self.outcome_accuracy,
            "reference_kl": self.reference_kl,
            "mean_prefill_tokens": self.mean_prefill_tokens,
            "mean_model_calls": self.mean_model_calls,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }


def run_workflow(
    engine: Engine, workflow: Workflow, cases: Sequence[WorkflowCase]
) -> WorkflowResult:
    """Execute the graph per case and score the decisions it produced."""
    if not cases:
        raise ValueError(f"workflow {workflow.name!r} has no cases")

    correct, judged = 0, 0
    kls: list[float] = []
    tokens: list[int] = []
    calls: list[int] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        answers: dict[str, object] = {}
        distributions: dict[str, dict[str, float]] = {}
        case_tokens, case_calls = 0, 0

        for step in workflow.steps:
            questions = step.build(case.state, answers)
            if not questions:
                continue
            response: SystemOneResponse = engine.answer(
                SystemOneRequest(state=case.state, questions=questions)  # type: ignore[arg-type]
            )
            case_calls += 1
            case_tokens += response.usage.prefill_tokens
            for qid, answer in response.answers.items():
                answers[qid] = _decision(answer)
                probs = getattr(answer, "probabilities", None)
                if probs is not None:
                    distributions[qid] = probs
                else:
                    distributions[qid] = {
                        "no": 1.0 - answer.probability,  # type: ignore[union-attr]
                        "yes": answer.probability,  # type: ignore[union-attr]
                    }

        latencies.append((time.perf_counter() - started) * 1000.0)
        tokens.append(case_tokens)
        calls.append(case_calls)

        for qid, truth in case.outcome.items():
            if qid in answers:
                judged += 1
                correct += int(answers[qid] == truth)
        for qid, reference in case.reference.items():
            if qid in distributions:
                kls.append(_kl(reference, distributions[qid]))

    latencies.sort()
    return WorkflowResult(
        workflow=workflow.name,
        model=engine.backend.model_version,
        n_cases=len(cases),
        outcome_accuracy=correct / judged if judged else None,
        reference_kl=statistics.fmean(kls) if kls else None,
        mean_prefill_tokens=statistics.fmean(tokens),
        mean_model_calls=statistics.fmean(calls),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p99_ms=_percentile(latencies, 0.99),
    )


def _decision(answer: object) -> str:
    selected = getattr(answer, "selected", None)
    if selected is not None:
        return selected
    probability = getattr(answer, "probability", None)
    if probability is not None:
        return "yes" if probability >= 0.5 else "no"
    # Score: the decision is the level the mass concentrated on.
    probabilities: dict[str, float] = answer.probabilities  # type: ignore[attr-defined]
    return max(probabilities, key=lambda k: probabilities[k])


def _kl(reference: dict[str, float], candidate: dict[str, float]) -> float:
    """KL(reference || candidate) over the reference's support."""
    total = 0.0
    for label, p in reference.items():
        if p <= 0:
            continue
        q = max(candidate.get(label, 0.0), 1e-9)
        total += p * math.log(p / q)
    return total


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    rank = max(1, math.ceil(q * len(sorted_values)))
    return sorted_values[rank - 1]
