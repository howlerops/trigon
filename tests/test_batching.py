"""One forward pass for several requests, and the guarantee that makes it safe.

`docs/next.md` B.2. A prefill-only model makes batching the easy case: every
request is exactly one pass, so a batch is a pad and a stack — no decode loop,
no ragged generation, no per-step scheduling.

**The property worth testing is not the speed, it is that an answer does not
depend on what else was in the batch.** That is the block mask's guarantee
extended across requests. A batcher that quietly mixes two callers' states
produces well-formed, well-calibrated answers to questions nobody asked, which
is the failure this whole project is built against and the one a throughput
benchmark cannot see.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")

from trigon.backends.torch_readout import TorchReadoutBackend  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals.datasets import synthetic_outcome_cases  # noqa: E402
from trigon.evals.harness import run_cases  # noqa: E402
from trigon.types import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

SHORT = SystemOneRequest(
    state="short", questions={"q": NoulQuestion(instructions="Is it present?")}
)
LONG = SystemOneRequest(
    state="a considerably longer piece of state " * 20,
    questions={
        "r": ChoiceQuestion(
            instructions="Pick one.", options=[{"name": n} for n in ("a", "b", "c")]
        )
    },
)
MIXED = SystemOneRequest(
    state="third, different again",
    questions={
        "q": NoulQuestion(instructions="Is it present?"),
        "s": ScoreQuestion(
            instructions="How much?", levels=[{"name": n} for n in ("low", "mid", "high")]
        ),
    },
)


@pytest.fixture(scope="module")
def engine() -> Engine:
    backend = TorchReadoutBackend(seed=0)
    return Engine(backend, compiler=backend.make_compiler())


def _probabilities(response) -> dict:
    out = {}
    for qid, answer in response.answers.items():
        out[qid] = getattr(answer, "probabilities", None) or {"p": answer.probability}
    return out


def test_a_batched_answer_equals_the_one_it_would_have_got_alone(engine):
    """Different lengths and different schemas in one batch, which is the
    case padding and per-sample masks exist for."""
    requests = [SHORT, LONG, MIXED]
    alone = [engine.answer(r) for r in requests]
    together = engine.answer_many(requests)
    for one, many in zip(alone, together, strict=True):
        assert _probabilities(one) == _probabilities(many)


def test_the_batch_a_request_lands_in_does_not_change_its_answer(engine):
    """The same request, batched with different neighbours each time."""
    baseline = _probabilities(engine.answer(SHORT))
    for neighbours in ([LONG], [MIXED], [LONG, MIXED], [MIXED, LONG, LONG]):
        batched = engine.answer_many([SHORT, *neighbours])
        assert _probabilities(batched[0]) == baseline, "an answer moved with the company it kept"


def test_padding_attends_to_nothing_real_and_stays_finite(engine):
    """A padding row that may attend to nothing is a softmax over an empty set,
    which is NaN — and one NaN in a batched attention poisons every sample
    sharing the tensor. Padding attends to itself instead."""
    batched = engine.answer_many([SHORT, LONG])
    for response in batched:
        for values in _probabilities(response).values():
            for p in values.values():
                assert p == p, "NaN reached an answer"
                assert 0.0 <= p <= 1.0


def test_order_is_preserved(engine):
    """A caller reads `responses[i]` as the answer to `requests[i]`. Sorting by
    length would pad less and is not worth breaking that."""
    requests = [SHORT, LONG, MIXED, SHORT, LONG]
    together = engine.answer_many(requests)
    assert len(together) == len(requests)
    for request, response in zip(requests, together, strict=True):
        assert set(response.answers) == set(request.questions)


def test_a_batch_of_one_and_an_empty_batch_are_not_special_cases_for_the_caller(engine):
    assert engine.answer_many([]) == []
    solo = engine.answer_many([SHORT])
    assert _probabilities(solo[0]) == _probabilities(engine.answer(SHORT))


def test_a_backend_without_a_batched_path_still_answers():
    """`answer_many` is always safe to call; it is faster only where the
    backend implements one."""
    from trigon.backends.lexical import LexicalBackend

    engine = Engine(LexicalBackend())
    assert not hasattr(engine.backend, "infer_many")
    responses = engine.answer_many([SHORT, MIXED])
    assert len(responses) == 2
    assert _probabilities(responses[0]) == _probabilities(engine.answer(SHORT))


def test_the_eval_harness_scores_identically_at_any_batch_size(engine):
    """Every eval run in this repository goes through `run_cases`. If batching
    moved a single answer, every published ECE would change with a tuning
    parameter."""
    cases = synthetic_outcome_cases(n=24, seed=3, noise=0.2)
    serial = run_cases(engine, cases, batch_size=1)
    batched = run_cases(engine, cases, batch_size=8)
    assert len(serial) == len(batched)
    for a, b in zip(serial, batched, strict=True):
        assert a.case.case_id == b.case.case_id
        assert _probabilities(a.response) == _probabilities(b.response)


def test_the_batched_latency_is_per_request_not_per_batch(engine):
    """A batch's wall clock shared out is what a request cost when the work was
    coalesced. Reporting the whole batch's time against each request would make
    a batch of eight look eight times more expensive than the same work
    unbatched."""
    cases = synthetic_outcome_cases(n=16, seed=4, noise=0.2)
    batched = run_cases(engine, cases, batch_size=16)
    serial = run_cases(engine, cases, batch_size=1)
    assert sum(o.latency_ms for o in batched) < sum(o.latency_ms for o in serial)
