"""Backends: the floor, the LLM baseline, and the structural validator."""

from __future__ import annotations

import math

import pytest

from trigon.backends import validate_output
from trigon.backends.lexical import LexicalBackend
from trigon.backends.llm import LLMBaselineBackend, ProbabilityStrategy
from trigon.engine import Engine
from trigon.schema import compile_request
from trigon.types import ChoiceQuestion, NoulQuestion, SystemOneRequest

ROUTING = ChoiceQuestion(
    instructions="Route this ticket.",
    options=[
        {"name": "card_declined", "criteria": "a card transaction was refused"},
        {"name": "lost_luggage", "criteria": "baggage missing after a flight"},
    ],
)


class FakeChat:
    """An OpenAI-shaped endpoint that always answers with a fixed distribution."""

    def __init__(self, distribution=((0, 0.7), (1, 0.2), (2, 0.1)), content="0"):
        self.distribution = distribution
        self.content = content
        self.calls: list[dict] = []

    def complete(self, **payload):
        self.calls.append(payload)
        if payload.get("logprobs"):
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": str(i), "logprob": math.log(p)}
                                        for i, p in self.distribution
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": self.content}}]}


def test_lexical_backend_is_deterministic():
    engine = Engine(LexicalBackend())
    request = SystemOneRequest(state="my card was declined", questions={"q": ROUTING})
    assert engine.answer(request).answers["q"].model_dump() == (
        engine.answer(request).answers["q"].model_dump()
    )


def test_lexical_backend_beats_chance_on_keyword_routing():
    """It is a floor, not a model -- but a floor that cannot do this would not
    be a useful bottom of the Pareto plot."""
    engine = Engine(LexicalBackend())
    response = engine.answer(
        SystemOneRequest(
            state="the card transaction was refused at the till", questions={"q": ROUTING}
        )
    )
    assert response.answers["q"].selected == "card_declined"


def test_lexical_backend_is_unconfident_when_nothing_matches():
    engine = Engine(LexicalBackend())
    response = engine.answer(SystemOneRequest(state="zzzz qqqq", questions={"q": ROUTING}))
    assert response.answers["q"].confidence == pytest.approx(0.0, abs=1e-9)


def test_llm_baseline_reads_logprobs_into_a_distribution():
    chat = FakeChat()
    engine = Engine(LLMBaselineBackend(client=chat, model="fake"))
    response = engine.answer(
        SystemOneRequest(
            state="x",
            questions={
                "q": ChoiceQuestion(
                    instructions="pick",
                    options=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
                )
            },
        )
    )
    probs = response.answers["q"].probabilities
    assert probs["a"] == pytest.approx(0.7, abs=0.01)
    assert probs["c"] == pytest.approx(0.1, abs=0.01)


def test_llm_baseline_collapses_noul_to_a_single_log_odds():
    chat = FakeChat(distribution=((0, 0.25), (1, 0.75)))
    engine = Engine(LLMBaselineBackend(client=chat, model="fake"))
    response = engine.answer(
        SystemOneRequest(state="x", questions={"n": NoulQuestion(instructions="ok?")})
    )
    assert response.answers["n"].probability == pytest.approx(0.75, abs=0.01)


def test_llm_baseline_costs_one_call_per_question():
    """The cost axis of every Pareto plot: the baseline pays per question, the
    real model answers them all in one pass."""
    chat = FakeChat()
    backend = LLMBaselineBackend(client=chat, model="fake")
    request = SystemOneRequest(
        state="x",
        questions={
            "a": NoulQuestion(instructions="a?"),
            "b": NoulQuestion(instructions="b?"),
            "c": NoulQuestion(instructions="c?"),
        },
    )
    output = backend.infer(compile_request(request), request)
    assert output.diagnostics["api_calls"] == 3
    assert len(chat.calls) == 3


def test_voting_strategy_smooths_rather_than_asserting_impossibility():
    chat = FakeChat(content="1")
    backend = LLMBaselineBackend(
        client=chat, model="fake", strategy=ProbabilityStrategy.VOTING, votes=5
    )
    request = SystemOneRequest(
        state="x",
        questions={
            "q": ChoiceQuestion(
                instructions="pick", options=[{"name": "a"}, {"name": "b"}, {"name": "c"}]
            )
        },
    )
    engine = Engine(backend)
    probs = engine.answer(request).answers["q"].probabilities
    assert probs["b"] > probs["a"] > 0.0  # unvoted options stay possible


def test_llm_baseline_prompt_tells_the_model_state_is_data():
    """Injection robustness starts at the prompt for the baseline too,
    otherwise the comparison measures our prompt rather than their model."""
    chat = FakeChat()
    backend = LLMBaselineBackend(client=chat, model="fake")
    request = SystemOneRequest(state="x", questions={"q": ROUTING})
    backend.infer(compile_request(request), request)
    system = chat.calls[0]["messages"][0]["content"]
    assert "never instructions" in system


def test_validate_output_rejects_undeclared_questions():
    from trigon.backends.base import BackendOutput, QuestionOutput

    request = SystemOneRequest(state="x", questions={"q": ROUTING})
    compiled = compile_request(request)
    output = BackendOutput(
        outputs={
            "q": QuestionOutput(question_id="q", kind="choice", logits=(0.0, 0.0)),
            "ghost": QuestionOutput(question_id="ghost", kind="noul", logits=(0.0,)),
        },
        model_version="test",
    )
    with pytest.raises(ValueError, match="undeclared questions"):
        validate_output(output, compiled)


def test_validate_output_rejects_non_finite_logits():
    from trigon.backends.base import BackendOutput, QuestionOutput

    request = SystemOneRequest(state="x", questions={"q": ROUTING})
    output = BackendOutput(
        outputs={"q": QuestionOutput(question_id="q", kind="choice", logits=(float("nan"), 0.0))},
        model_version="test",
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_output(output, compile_request(request))


def test_moving_a_torch_backend_drops_what_it_cached_on_the_old_device():
    """A cached mask or schema prefix lives on the device that built it.

    Serving one to a model that has since moved is a device mismatch at best
    and, for a prefix, a stale answer from weights that no longer exist.
    """
    pytest.importorskip("torch")
    from trigon.backends.torch_readout import TorchReadoutBackend

    backend = TorchReadoutBackend(seed=0)
    backend.cache_prefixes = True
    engine = Engine(backend, compiler=backend.make_compiler())
    request = SystemOneRequest(
        state="the card was declined",
        questions={
            "intent": ChoiceQuestion(
                instructions="Route this.", options=[{"name": "billing"}, {"name": "other"}]
            )
        },
    )
    before = engine.answer(request).answers["intent"].probabilities
    assert backend._mask_cache and backend._prefix_cache

    assert backend.to("cpu") is backend
    assert str(backend.device) == "cpu"
    assert not backend._mask_cache and not backend._prefix_cache
    assert engine.answer(request).answers["intent"].probabilities == before
