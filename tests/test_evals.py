"""The eval harness: alignment, scoring, gates, and the benchmarks' own logic.

The benchmarks are scored by hand here against stub engines with known
behaviour, because a benchmark that silently mis-scores is worse than no
benchmark -- it produces a number people quote.
"""

from __future__ import annotations

import pytest

from trigon.backends.base import BackendOutput, QuestionOutput
from trigon.backends.lexical import LexicalBackend
from trigon.engine import Engine
from trigon.evals import (
    Case,
    Expectation,
    Workflow,
    WorkflowCase,
    WorkflowStep,
    check_gates,
    render_json,
    render_markdown,
    run_calibration_suite,
    run_cases,
    run_jaggedness,
    run_workflow,
    slice_reports,
    summarize,
    synthetic_outcome_cases,
)
from trigon.evals.jaggedness import (
    NegationCoherenceBenchmark,
    NoulChoiceAgreementBenchmark,
    all_benchmarks,
)
from trigon.types import ChoiceQuestion, NoulQuestion, SystemOneRequest


class FixedNoul:
    """Answers every Noul with the same probability, whatever was asked.

    A constant predictor at the true base rate is calibrated by definition,
    which makes it the right stub for asserting that the gates let a calibrated
    model through. Categorical heads get a uniform distribution.
    """

    model_version = "fixed-1.0.0"

    def __init__(self, probability: float):
        self.probability = probability

    def infer(self, compiled, request):
        import math

        odds = self.probability / (1.0 - self.probability)
        return BackendOutput(
            outputs={
                q.question_id: QuestionOutput(
                    question_id=q.question_id,
                    kind=q.kind,
                    logits=(math.log(odds),) if q.kind == "noul" else (0.0,) * q.cardinality,
                )
                for q in compiled.schema.questions
            },
            model_version=self.model_version,
        )


def _noul_only(cases):
    """Strip the categorical questions, keeping the Noul and its ground truth."""
    return [
        Case(
            case_id=c.case_id,
            request=SystemOneRequest(
                state=c.request.state,
                questions={"at_risk": c.request.questions["at_risk"]},
            ),
            expected={"at_risk": c.expected["at_risk"]},
            domain=c.domain,
            tags=c.tags,
        )
        for c in cases
    ]


def test_outcomes_are_aligned_to_declared_label_order():
    engine = Engine(LexicalBackend())
    case = next(iter(synthetic_outcome_cases(n=1)))
    outcome = run_cases(engine, [case])[0]
    plan = outcome.questions["plan"]
    declared = case.request.questions["plan"].names
    assert plan.labels == tuple(declared)
    assert (
        plan.probabilities[declared.index("pro")]
        == (outcome.response.answers["plan"].probabilities["pro"])
    )


def test_noul_outcomes_are_expanded_to_no_yes():
    engine = Engine(FixedNoul(0.8))
    case = _noul_only(synthetic_outcome_cases(n=1))[0]
    outcome = run_cases(engine, [case])[0]
    assert outcome.questions["at_risk"].labels == ("no", "yes")
    assert outcome.questions["at_risk"].probabilities[1] == pytest.approx(0.8, abs=1e-6)


def test_expectation_requires_some_ground_truth():
    with pytest.raises(ValueError, match="needs a label"):
        Expectation()


def test_expectation_from_a_distribution_uses_its_mode():
    assert Expectation(distribution=(0.1, 0.6, 0.3)).hard_label == 1


def test_gates_fail_an_uncalibrated_model_and_say_by_how_much():
    engine = Engine(LexicalBackend())
    result, _ = run_calibration_suite(engine, synthetic_outcome_cases(n=60, noise=0.1))
    gates = check_gates(result)
    assert any(not g.passed for g in gates)
    failing = next(g for g in gates if not g.passed)
    assert failing.value > failing.limit
    assert "FAIL" in str(failing)


def test_gates_pass_a_perfectly_calibrated_stub():
    """A model that answers every Noul at the true base rate is calibrated by
    construction, so the gates must let it through -- including the equal-mass
    estimator, which needs tie-aware binning not to invent a gap here."""
    cases = _noul_only(synthetic_outcome_cases(n=400, noise=0.5))
    base_rate = sum(1 for c in cases if c.expected["at_risk"].hard_label == 1) / len(cases)
    result, _ = run_calibration_suite(Engine(FixedNoul(base_rate)), cases)
    gates = check_gates(result)
    assert all(g.passed for g in gates), [str(g) for g in gates]


def test_slice_reports_split_by_domain():
    engine = Engine(LexicalBackend())
    outcomes = run_cases(engine, synthetic_outcome_cases(n=20))
    slices = slice_reports(outcomes)
    assert set(slices) == {"accounts"}
    assert slices["accounts"].n > 0


def test_negation_coherence_scores_a_perfectly_incoherent_model():
    """A model answering 0.8 to both a claim and its complement is 0.6 away
    from coherent, and the benchmark must say exactly that."""
    benchmark = NegationCoherenceBenchmark(n=5)
    outcomes = run_cases(Engine(FixedNoul(0.8)), benchmark.cases())
    scores = benchmark.score(outcomes)
    assert scores["mean_incoherence"] == pytest.approx(0.6, abs=1e-6)
    assert scores["coherent_rate"] == 0.0


def test_negation_coherence_scores_a_coherent_model():
    benchmark = NegationCoherenceBenchmark(n=5)
    outcomes = run_cases(Engine(FixedNoul(0.5)), benchmark.cases())
    assert benchmark.score(outcomes)["mean_incoherence"] == pytest.approx(0.0, abs=1e-6)
    assert benchmark.score(outcomes)["coherent_rate"] == 1.0


def test_noul_choice_agreement_detects_a_disagreeing_model():
    """The stub answers Noul at 0.9 but the Choice head is uniform, so the two
    phrasings of the same question disagree by 0.4."""

    class Split:
        model_version = "split-1.0.0"

        def infer(self, compiled, request):
            import math

            return BackendOutput(
                outputs={
                    q.question_id: QuestionOutput(
                        question_id=q.question_id,
                        kind=q.kind,
                        logits=(math.log(0.9 / 0.1),)
                        if q.kind == "noul"
                        else (0.0,) * q.cardinality,
                    )
                    for q in compiled.schema.questions
                },
                model_version=self.model_version,
            )

    benchmark = NoulChoiceAgreementBenchmark(n=5)
    scores = benchmark.score(run_cases(Engine(Split()), benchmark.cases()))
    assert scores["mean_disagreement"] == pytest.approx(0.4, abs=1e-6)


def test_every_benchmark_runs_and_names_its_failure_mode():
    engine = Engine(LexicalBackend())
    results = run_jaggedness(engine, n=4)
    assert len(results) == len(all_benchmarks())
    for benchmark in all_benchmarks():
        assert benchmark.failure_mode
        assert benchmark.name in " ".join(r.suite for r in results)
    for result in results:
        assert result.n_cases > 0


def test_non_goal_benchmarks_are_marked_as_such():
    """Counting and date comparison are contract, not regression."""
    marked = {b.name for b in all_benchmarks() if b.non_goal}
    assert marked == {"counting", "date_comparison"}


def test_injection_benchmark_separates_steering_from_detection():
    benchmark = next(b for b in all_benchmarks(n=4) if b.name == "injection_steering")
    scores = benchmark.score(run_cases(Engine(LexicalBackend()), benchmark.cases()))
    assert "flip_rate" in scores and "guardrail_detection" in scores
    # A model immune to embedded instructions has a zero flip rate; that is a
    # different property from spotting the injection, which it may still fail.
    assert scores["flip_rate"] == 0.0


def test_context_rot_reports_a_slope_not_just_an_average():
    benchmark = next(b for b in all_benchmarks(n=4) if b.name == "context_rot")
    scores = benchmark.score(run_cases(Engine(LexicalBackend()), benchmark.cases()))
    assert "rot" in scores
    assert "accuracy@pad0" in scores


def test_workflow_scores_against_outcomes_and_reports_cost():
    triage = Workflow(
        name="triage",
        steps=(
            WorkflowStep(
                name="route",
                build=lambda state, answers: {
                    "team": ChoiceQuestion(
                        instructions="Route this ticket.",
                        options=[
                            {"name": "billing", "criteria": "a charge or refund"},
                            {"name": "shipping", "criteria": "delivery or tracking"},
                        ],
                    )
                },
            ),
            WorkflowStep(
                name="refund",
                # Conditional: only asked when the first step said billing.
                build=lambda state, answers: (
                    {"refund": NoulQuestion(instructions="Should we issue a refund?")}
                    if answers.get("team") == "billing"
                    else None
                ),
            ),
        ),
    )
    cases = [
        WorkflowCase(
            case_id="w1",
            state="I was charged twice for one order and want my money back.",
            outcome={"team": "billing"},
            reference={"team": {"billing": 0.9, "shipping": 0.1}},
        ),
        WorkflowCase(
            case_id="w2",
            state="My delivery tracking has not updated in a week.",
            outcome={"team": "shipping"},
        ),
    ]
    result = run_workflow(Engine(LexicalBackend()), triage, cases)
    assert result.n_cases == 2
    assert 0.0 <= result.outcome_accuracy <= 1.0
    assert result.reference_kl is not None
    assert result.mean_prefill_tokens > 0
    assert result.mean_model_calls >= 1.0


def test_reports_render_to_markdown_and_json():
    engine = Engine(LexicalBackend())
    result, slices = run_calibration_suite(engine, synthetic_outcome_cases(n=30))
    gates = check_gates(result)
    markdown = render_markdown([result], gates, slices)
    assert "Reliability diagram" in markdown
    assert "Release gates" in markdown

    import json

    payload = json.loads(render_json([result], gates, slices))
    assert payload["results"][0]["suite"] == "calibration"
    assert payload["gates"][0]["name"].endswith("ece")


def test_summarize_refuses_an_empty_run():
    with pytest.raises(ValueError, match="no outcomes"):
        summarize("empty", "model", [])
