"""The eval harness: alignment, scoring, gates, and the benchmarks' own logic.

The benchmarks are scored by hand here against stub engines with known
behaviour, because a benchmark that silently mis-scores is worse than no
benchmark -- it produces a number people quote.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from trigon.backends.base import BackendOutput, QuestionOutput
from trigon.backends.lexical import LexicalBackend
from trigon.calibration.temperature import TemperatureScaler
from trigon.engine import Engine
from trigon.evals import (
    Case,
    Expectation,
    Workflow,
    WorkflowCase,
    WorkflowStep,
    blocking,
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
from trigon.evals.cardinality import CardinalityResult
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
    result, _ = run_calibration_suite(
        engine, synthetic_outcome_cases(n=2000, noise=0.1), floor_trials=20
    )
    gates = {g.name: g for g in check_gates(result)}
    ece = gates["workhorse_ece"]
    assert not ece.passed
    assert ece.value > ece.limit
    assert "FAIL" in str(ece)


def test_a_run_too_small_to_test_cannot_certify_anything():
    """The failure this prevents has happened in public: ECE reported at n=60
    as evidence of calibration, when at that size it is estimator noise."""
    engine = Engine(LexicalBackend())
    result, _ = run_calibration_suite(
        engine, synthetic_outcome_cases(n=60, noise=0.1), floor_trials=20
    )
    gates = {g.name: g for g in check_gates(result)}
    assert not gates["sample_size"].passed
    assert not gates["gate_is_testable"].passed
    # The gate explains itself rather than just failing.
    assert "perfectly calibrated" in gates["gate_is_testable"].note


def test_the_floor_is_reported_alongside_the_number():
    engine = Engine(LexicalBackend())
    result, _ = run_calibration_suite(
        engine, synthetic_outcome_cases(n=300, noise=0.2), floor_trials=30
    )
    floor = result.calibration.floor
    assert floor is not None and floor.p95 >= floor.mean > 0
    assert result.calibration.distinguishable is not None


def test_a_marginal_predictor_passes_every_calibration_gate():
    """The finding that added ``accuracy_over_baseline``.

    A model that answers every Noul at the true base rate is calibrated by
    construction -- it reports the right distribution, it just ignores the
    input entirely. Every ECE-based gate passes it. A calibration-only release
    gate therefore certifies the one model guaranteed to be useless, which is
    worse than no gate because it looks like evidence.
    """
    cases = _noul_only(synthetic_outcome_cases(n=6000, noise=0.5))
    base_rate = sum(1 for c in cases if c.expected["at_risk"].hard_label == 1) / len(cases)
    result, _ = run_calibration_suite(Engine(FixedNoul(base_rate)), cases, floor_trials=40)
    gates = {g.name: g for g in check_gates(result)}

    assert gates["workhorse_ece"].passed
    assert gates["workhorse_adaptive_ece"].passed
    assert gates["sample_size"].passed
    assert gates["gate_is_testable"].passed
    assert result.calibration.distinguishable is False

    # And it is caught by exactly one gate.
    assert not gates["accuracy_over_baseline"].passed
    assert gates["accuracy_over_baseline"].value == pytest.approx(0.0, abs=1e-9)
    assert "ignores the state" in gates["accuracy_over_baseline"].note


def test_baseline_is_the_per_question_majority_label():
    cases = _noul_only(synthetic_outcome_cases(n=400, noise=0.5))
    truths = [c.expected["at_risk"].hard_label for c in cases]
    expected = max(truths.count(0), truths.count(1)) / len(truths)
    result, _ = run_calibration_suite(Engine(FixedNoul(0.5)), cases, floor_trials=10)
    assert result.baseline_accuracy == pytest.approx(expected)


def test_slice_reports_split_by_domain_and_by_primitive():
    """Both cuts, and the keys cannot collide.

    Per-primitive was documented from the first draft of `docs/evals.md` and
    not implemented, which left the suite unable to say *where* a run is
    miscalibrated. A temperature is fitted per primitive, so that is exactly
    the unit at which a fit can go wrong -- and one did, at a fitted Score
    temperature of 0.20 on a head that had learned nothing.
    """
    engine = Engine(LexicalBackend())
    outcomes = run_cases(engine, synthetic_outcome_cases(n=20))
    slices = slice_reports(outcomes)

    assert set(slices) == {
        "domain:accounts",
        "primitive:choice",
        "primitive:noul",
        "primitive:score",
    }
    assert all(s.n > 0 for s in slices.values())
    assert all(name == s.slice_name for name, s in slices.items())

    # The two cuts partition the same answers, so they must agree on the total.
    domain_total = sum(s.n for name, s in slices.items() if name.startswith("domain:"))
    primitive_total = sum(s.n for name, s in slices.items() if name.startswith("primitive:"))
    assert domain_total == primitive_total

    # Prefixed keys, because a domain named `choice` would otherwise overwrite
    # the primitive of that name and silently halve the table.
    assert not any(name in {"choice", "noul", "score", "accounts"} for name in slices)


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
    result, slices = run_calibration_suite(engine, synthetic_outcome_cases(n=200), floor_trials=20)
    gates = check_gates(result)
    markdown = render_markdown([result], gates, slices)
    assert "Reliability diagram" in markdown
    assert "Release gates" in markdown
    # The floor belongs in the published report, not only in the gate list.
    assert "perfectly calibrated" in markdown

    import json

    payload = json.loads(render_json([result], gates, slices))
    assert payload["results"][0]["suite"] == "calibration"
    assert {g["name"] for g in payload["gates"]} >= {"sample_size", "gate_is_testable"}
    assert payload["results"][0]["calibration"]["floor"]["p95"] > 0


def test_summarize_refuses_an_empty_run():
    with pytest.raises(ValueError, match="no outcomes"):
        summarize("empty", "model", [])


def test_json_report_covers_every_suite_the_markdown_does():
    """A report and its ``.json`` sibling must not disagree about what ran.

    ``render_json`` took three arguments while ``render_markdown`` took five,
    so a cardinality or workflow run wrote a JSON file that recorded neither.
    """
    import inspect
    import json

    assert inspect.signature(render_json).parameters.keys() == (
        inspect.signature(render_markdown).parameters.keys()
    )

    failing = CardinalityResult(
        options=10_000, shortlist=256, recall=0.94, limit=0.99, queries=200, drop_slots=2
    )
    payload = json.loads(render_json([], [], {}, [failing], []))
    assert payload["cardinality"][0]["recall"] == 0.94
    assert payload["cardinality"][0]["passed"] is False
    # The CLI exits non-zero on this run, so the file a consumer reads must
    # not report it as a pass.
    assert payload["passed"] is False


# -- the two gates nothing had ever run --------------------------------------
#
# Coverage found the quantization-delta gate and the premium tier's tighter
# limit entirely unexecuted. Both are release gates: an untested gate is a
# claim about what CI would block, not a thing CI blocks.


def test_the_premium_tier_is_gated_more_tightly_than_the_workhorse():
    engine = Engine(LexicalBackend())
    result, _ = run_calibration_suite(engine, synthetic_outcome_cases(n=200), floor_trials=10)
    workhorse = {g.name: g for g in check_gates(result, tier="workhorse")}
    premium = {g.name: g for g in check_gates(result, tier="premium")}
    assert premium["premium_ece"].limit < workhorse["workhorse_ece"].limit
    # Same measurement, different bar -- the value must not move with the tier.
    assert premium["premium_ece"].value == workhorse["workhorse_ece"].value


def test_an_unknown_tier_is_rejected_rather_than_silently_ungated():
    engine = Engine(LexicalBackend())
    result, _ = run_calibration_suite(engine, synthetic_outcome_cases(n=200), floor_trials=10)
    with pytest.raises(KeyError):
        check_gates(result, tier="freemium")


def test_quantization_gate_catches_a_calibration_regression_argmax_would_miss():
    """Probabilities degrade well before argmax does, which is the entire
    reason this gate is separate from an accuracy check."""
    engine = Engine(LexicalBackend())
    cases = synthetic_outcome_cases(n=200)
    baseline, _ = run_calibration_suite(engine, cases, floor_trials=10)

    # A serving path that keeps every argmax and wrecks the probabilities: the
    # same engine at a temperature far from 1.
    scaler = TemperatureScaler(primitive={"choice": 0.2, "noul": 0.2, "score": 0.2})
    degraded, _ = run_calibration_suite(
        Engine(LexicalBackend(), scaler=scaler), cases, floor_trials=10
    )
    assert degraded.accuracy == pytest.approx(baseline.accuracy), (
        "sharpening cannot change which option wins, or this is not the right probe"
    )

    gates = {g.name: g for g in check_gates(baseline, quantized=degraded)}
    delta = gates["quantization_ece_delta"]
    assert delta.value > delta.limit and not delta.passed
    # And it stays quiet when the quantized path did not move calibration.
    same = {g.name: g for g in check_gates(baseline, quantized=baseline)}
    assert same["quantization_ece_delta"].value == 0.0
    assert same["quantization_ece_delta"].passed


def test_gates_refuse_a_run_with_nothing_scored():
    engine = Engine(LexicalBackend())
    result, _ = run_calibration_suite(engine, synthetic_outcome_cases(n=200), floor_trials=10)
    empty = replace(result, calibration=None)
    with pytest.raises(ValueError, match="no scored questions"):
        check_gates(empty)
    with pytest.raises(ValueError, match="quantized run produced no scored questions"):
        check_gates(result, quantized=empty)


# -- the per-question gate ---------------------------------------------------


def _suite_over(engine):
    result, _ = run_calibration_suite(
        engine, synthetic_outcome_cases(n=300, noise=0.2), floor_trials=5
    )
    return result


def test_per_question_accuracy_shows_what_the_pooled_number_averages_away():
    """The reference run clears the pooled gate with two of three questions
    below their own marginal predictor. That has to be visible somewhere."""
    result = _suite_over(Engine(LexicalBackend()))
    assert set(result.per_question) == {"plan", "at_risk", "size"}
    # The floor answers `plan` well above rote and is at or below it elsewhere.
    assert result.per_question["plan"].lift > 0.2
    assert result.worst_question_lift < 0
    # And the pooled figure is positive regardless -- the whole problem.
    assert result.lift_over_baseline > 0
    assert result.worst_question_lift < result.lift_over_baseline


def test_the_per_question_gate_is_advisory_until_asked_for():
    result = _suite_over(Engine(LexicalBackend()))
    gates = {g.name: g for g in check_gates(result)}
    gate = gates["worst_question_over_baseline"]
    assert gate.advisory and not gate.passed
    assert "the pooled gate hides this" in gate.note
    # Advisory means measured and printed, never counted.
    assert gate not in blocking(check_gates(result))
    assert all(g.passed for g in blocking(check_gates(result)) if g.name.endswith("baseline"))


def test_require_per_question_makes_it_block():
    result = _suite_over(Engine(LexicalBackend()))
    gates = check_gates(result, require_per_question=True)
    gate = next(g for g in gates if g.name == "worst_question_over_baseline")
    assert not gate.advisory
    assert gate in blocking(gates)
    assert not all(g.passed for g in blocking(gates))


def test_an_advisory_failure_does_not_read_as_a_clean_run():
    """A report that says 'all gates passed' while a gate failed is worse than
    having no advisory gate at all."""
    result = _suite_over(Engine(LexicalBackend()))
    markdown = render_markdown([result], check_gates(result))
    assert "Advisory, not blocking" in markdown
    assert "worst_question_over_baseline" in markdown
    assert "Accuracy per question" in markdown


def test_the_json_verdict_ignores_advisory_gates_like_the_exit_code_does():
    import json

    result = _suite_over(Engine(LexicalBackend()))
    gates = check_gates(result)
    payload = json.loads(render_json([result], gates))
    assert payload["passed"] == all(g.passed for g in blocking(gates))


def test_the_json_report_says_which_failures_block():
    """A consumer must reach the same verdict from the file as the command did.

    The top-level `passed` is computed over `blocking(gates)`, but the gate
    list carried no `advisory` flag, so anything re-deriving the verdict by
    filtering on `passed` got a stricter answer than `trigon train`'s own exit
    code. `scripts/seed_sweep.py` did exactly that, and reported an 8,000-case
    configuration as certifying on none of four seeds when three of the four
    were blocked only by the advisory per-question gate.

    That is the `honest defaults` rule applied to a machine-readable file: a
    report may say a run failed, and may say the failure does not block, but it
    may not say the first while withholding the second.
    """
    import json as _json

    from trigon.evals.calibration_suite import GateResult
    from trigon.evals.report import render_json

    gates = [
        GateResult("workhorse_ece", 0.01, 0.05, True),
        GateResult("worst_question_over_baseline", -0.04, 0.05, False, advisory=True),
    ]
    payload = _json.loads(render_json([], gates))

    by_name = {g["name"]: g for g in payload["gates"]}
    assert by_name["worst_question_over_baseline"]["advisory"] is True
    assert by_name["workhorse_ece"]["advisory"] is False

    # The two verdicts agree: an advisory failure does not fail the run, and
    # the file says so rather than leaving it to be inferred.
    assert payload["passed"] is True
    rederived = all(g["passed"] for g in payload["gates"] if not g["advisory"])
    assert rederived == payload["passed"]

    # And a blocking failure fails both.
    blocked = _json.loads(render_json([], [GateResult("workhorse_ece", 0.9, 0.05, False)]))
    assert blocked["passed"] is False
    assert all(g["advisory"] is False for g in blocked["gates"])
