"""The serving pipeline, including the guarantees it refuses to break."""

from __future__ import annotations

import pytest

from trigon.backends.base import BackendOutput, QuestionOutput
from trigon.calibration.conformal import ConformalMethod, ConformalPredictor
from trigon.calibration.temperature import TemperatureScaler
from trigon.engine import Engine, EngineConfig
from trigon.types import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion


class StubBackend:
    """Returns fixed logits, so the pipeline's arithmetic is checkable by hand."""

    model_version = "stub-1.0.0"

    def __init__(self, logits: dict[str, tuple[float, ...]], kinds: dict[str, str]):
        self.logits = logits
        self.kinds = kinds

    def infer(self, compiled, request):
        return BackendOutput(
            outputs={
                qid: QuestionOutput(question_id=qid, kind=self.kinds[qid], logits=values)
                for qid, values in self.logits.items()
            },
            model_version=self.model_version,
        )


def test_answers_cover_every_declared_label(engine, support_request):
    response = engine.answer(support_request)
    choice = response.answers["intent"]
    assert set(choice.probabilities) == {"card_declined", "lost_luggage", "password_reset"}
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)
    assert choice.selected in choice.probabilities


def test_score_is_the_expectation_over_declared_anchors():
    backend = StubBackend({"s": (0.0, 0.0)}, {"s": "score"})
    engine = Engine(backend)
    response = engine.answer(
        DecisionRequest(
            state="x",
            questions={
                "s": ScoreQuestion(
                    instructions="rate",
                    levels=[{"name": "lo", "value": 1.0}, {"name": "hi", "value": 5.0}],
                )
            },
        )
    )
    # Equal logits -> 50/50 -> expectation exactly halfway.
    assert response.answers["s"].score == pytest.approx(3.0)


def test_noul_carries_no_confidence_field(engine, support_request):
    answer = engine.answer(support_request).answers["urgent"]
    assert not hasattr(answer, "confidence")
    assert 0.0 <= answer.probability <= 1.0


def test_temperature_is_applied_before_confidence_is_derived():
    backend = StubBackend({"c": (3.0, 0.0, 0.0)}, {"c": "choice"})
    question = ChoiceQuestion(
        instructions="pick", options=[{"name": "a"}, {"name": "b"}, {"name": "c"}]
    )
    request = DecisionRequest(state="x", questions={"c": question})

    sharp = Engine(backend).answer(request).answers["c"]
    flat = (
        Engine(backend, scaler=TemperatureScaler(primitive={"choice": 5.0}))
        .answer(request)
        .answers["c"]
    )
    assert flat.probabilities["a"] < sharp.probabilities["a"]
    # Confidence must follow the calibrated distribution, not the raw logits.
    assert flat.confidence < sharp.confidence


def test_raw_probabilities_are_returned_only_when_asked():
    backend = StubBackend({"c": (2.0, 0.0)}, {"c": "choice"})
    question = ChoiceQuestion(instructions="pick", options=[{"name": "a"}, {"name": "b"}])
    engine = Engine(backend, scaler=TemperatureScaler(primitive={"choice": 4.0}))

    plain = engine.answer(DecisionRequest(state="x", questions={"c": question}))
    assert plain.answers["c"].raw_probabilities is None

    verbose = engine.answer(
        DecisionRequest(
            state="x", questions={"c": question}, options={"include_raw_probabilities": True}
        )
    )
    answer = verbose.answers["c"]
    assert answer.raw_probabilities is not None
    assert answer.raw_probabilities["a"] > answer.probabilities["a"]


def test_conformal_profile_is_applied_when_requested():
    backend = StubBackend({"c": (3.0, 0.0, 0.0)}, {"c": "choice"})
    engine = Engine(
        backend,
        conformal={
            "support": ConformalPredictor(
                alpha=0.1, method=ConformalMethod.LAC, threshold=0.95, calibration_n=500
            )
        },
    )
    response = engine.answer(
        DecisionRequest(
            state="x",
            questions={
                "c": ChoiceQuestion(
                    instructions="pick",
                    options=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
                )
            },
            options={"conformal_profile": "support"},
        )
    )
    answer = response.answers["c"]
    assert answer.coverage_target == pytest.approx(0.9)
    assert set(answer.prediction_set) <= set(answer.probabilities)


def test_unknown_conformal_profile_is_an_explicit_error(engine, support_request):
    request = support_request.model_copy(
        update={"options": support_request.options.model_copy(update={"conformal_profile": "nope"})}
    )
    with pytest.raises(KeyError, match="no conformal profile"):
        engine.answer(request)


def test_a_backend_that_breaks_the_contract_is_caught():
    backend = StubBackend({"c": (1.0, 2.0)}, {"c": "choice"})  # 2 logits, 3 options
    engine = Engine(backend)
    with pytest.raises(ValueError, match="expected 3 logits"):
        engine.answer(
            DecisionRequest(
                state="x",
                questions={
                    "c": ChoiceQuestion(
                        instructions="pick",
                        options=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
                    )
                },
            )
        )


def test_a_backend_returning_the_wrong_head_is_caught():
    backend = StubBackend({"n": (1.0,)}, {"n": "choice"})
    with pytest.raises(ValueError, match="choice head for a noul question"):
        Engine(backend).answer(
            DecisionRequest(state="x", questions={"n": NoulQuestion(instructions="ok?")})
        )


def test_large_option_sets_are_shortlisted_and_still_answered_in_full(engine):
    # Above the 2,048-option shortlist, so the prefilter actually narrows.
    options = [{"name": f"intent_{i}"} for i in range(6000)]
    options[900] = {"name": "card_payment_declined", "criteria": "a card transaction was refused"}
    response = engine.answer(
        DecisionRequest(
            state="my card payment was declined at the store",
            questions={"intent": ChoiceQuestion(instructions="route", options=options)},
        )
    )
    answer = response.answers["intent"]
    # Every declared option is accounted for, including the ones the prefilter
    # dropped -- they carry probability 0 rather than disappearing.
    assert len(answer.probabilities) == 6000
    assert answer.shortlisted_from == 6000
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.probabilities["card_payment_declined"] > 0.0


def test_usage_reports_the_prefill_split(engine, support_request):
    usage = engine.answer(support_request).usage
    assert usage.prefill_tokens == usage.state_tokens + usage.schema_tokens + usage.readout_tokens
    assert usage.readout_tokens > 0


def test_domain_selects_a_per_domain_temperature():
    backend = StubBackend({"c": (3.0, 0.0)}, {"c": "choice"})
    scaler = TemperatureScaler(primitive={"choice": 1.0}, domain={"support": {"choice": 8.0}})
    question = ChoiceQuestion(instructions="pick", options=[{"name": "a"}, {"name": "b"}])
    request = DecisionRequest(state="x", questions={"c": question})

    general = Engine(backend, scaler=scaler).answer(request).answers["c"]
    support = (
        Engine(backend, scaler=scaler, config=EngineConfig(domain="support"))
        .answer(request)
        .answers["c"]
    )
    assert support.probabilities["a"] < general.probabilities["a"]


def test_an_option_set_inside_the_shortlist_is_not_narrowed(engine):
    """Raising the shortlist to 2,048 means mid-sized questions now skip the
    prefilter entirely rather than paying for a stage they do not need."""
    options = [{"name": f"intent_{i}"} for i in range(1500)]
    options[900] = {"name": "card_payment_declined", "criteria": "a card was refused"}
    response = engine.answer(
        DecisionRequest(
            state="my card payment was declined",
            questions={"intent": ChoiceQuestion(instructions="route", options=options)},
        )
    )
    assert response.answers["intent"].shortlisted_from is None


def test_top_probabilities_trims_the_response_without_renormalising(engine):
    """A truncated map that summed to 1 would misrepresent how much of the
    distribution the caller is seeing."""
    options = [{"name": f"intent_{i}", "criteria": f"about topic {i}"} for i in range(400)]
    options[7] = {"name": "card_declined", "criteria": "a card transaction was refused"}
    request = DecisionRequest(
        state="my card transaction was refused",
        questions={"intent": ChoiceQuestion(instructions="route", options=options)},
        options={"top_probabilities": 10},
    )
    answer = engine.answer(request).answers["intent"]

    assert answer.truncated is True
    assert len(answer.probabilities) == 10
    assert answer.selected in answer.probabilities  # the pick is always present
    assert sum(answer.probabilities.values()) < 1.0
    assert answer.probability_mass == pytest.approx(sum(answer.probabilities.values()))


def test_confidence_is_computed_before_truncation(engine):
    """Confidence derived from a trimmed vector would read high simply because
    the tail was dropped."""
    options = [{"name": f"intent_{i}", "criteria": f"about topic {i}"} for i in range(400)]
    base = DecisionRequest(
        state="a message with no particular signal",
        questions={"intent": ChoiceQuestion(instructions="route", options=options)},
    )
    full = engine.answer(base).answers["intent"]
    trimmed = engine.answer(
        base.model_copy(
            update={"options": base.options.model_copy(update={"top_probabilities": 5})}
        )
    ).answers["intent"]
    assert trimmed.confidence == pytest.approx(full.confidence)
