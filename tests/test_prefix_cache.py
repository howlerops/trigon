"""The KV prefix cache: the claim this layout exists to make possible.

`tests/test_independence.py` asserts that the schema half of the sequence
encodes identically regardless of state. That property was asserted and never
used — `Usage.cached_schema_tokens` reported 0 on every response and the spec
said to read it as "not cached, never not cacheable". This is the code that
uses it.

The tests that matter here are the ones about *when the cache is wrong*: a
cache that returns the right answer is worth nothing if it also returns a
stale one after the weights move.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")

from trigon.backends.torch_readout import (  # noqa: E402
    ReadoutConfig,
    TorchReadoutBackend,
)
from trigon.engine import Engine  # noqa: E402
from trigon.types import SystemOneRequest  # noqa: E402

SHAPE = dict(d_model=64, n_layers=2, n_heads=2, d_ff=64)
REQUEST = {
    "state": {"plan": "pro", "seats": 142, "open_tickets": 5},
    "questions": {
        "plan": {
            "type": "choice",
            "instructions": "Which plan is this account on?",
            "options": [{"name": n, "criteria": f"the {n} tier"} for n in ("free", "pro", "max")],
        },
        "at_risk": {"type": "noul", "instructions": "Is this account at risk?"},
    },
}


def _answer(backend, body=REQUEST):
    return Engine(backend, compiler=backend.make_compiler()).answer(
        SystemOneRequest.model_validate(body)
    )


def test_the_cached_path_agrees_with_the_uncached_one():
    """Not bit-identical, and the reason is worth knowing.

    The uncached path computes attention with every position as a query; the
    cached path queries only the non-schema positions against a concatenated
    key tensor. Same sum over the same terms, different GEMM shape, so float32
    rounds differently — measured at ~7e-7 on hidden states and ~3e-8 on
    probabilities.

    That is float epsilon, and it is *serving* tolerance rather than training
    tolerance. `scripts/seed_sweep.py` records that a 1e-6 perturbation can
    move a training run from one outcome to another, which is exactly why the
    cache is off during training rather than merely discouraged.
    """
    plain = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0)
    cached = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)

    a, b = _answer(plain), _answer(cached)
    assert a.answers["plan"].selected == b.answers["plan"].selected
    for name, probability in a.answers["plan"].probabilities.items():
        assert b.answers["plan"].probabilities[name] == pytest.approx(probability, abs=1e-6)
    assert b.answers["at_risk"].probability == pytest.approx(
        a.answers["at_risk"].probability, abs=1e-6
    )


def test_a_second_request_hits_the_cache_and_still_agrees():
    """The first call fills it; the second is the one that could be stale."""
    cached = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    first = _answer(cached)
    assert len(cached._prefix_cache) == 1
    second = _answer(cached)
    assert len(cached._prefix_cache) == 1
    assert first.answers["plan"].probabilities == second.answers["plan"].probabilities


def test_a_different_state_reuses_the_prefix_and_still_answers_differently():
    """Which is the whole claim: the schema is shared, the answer is not."""
    cached = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    first = _answer(cached)
    other = dict(REQUEST, state={"plan": "free", "seats": 3, "open_tickets": 0})
    second = _answer(cached, other)

    assert len(cached._prefix_cache) == 1, "a different state must not be a different prefix"
    assert first.answers["plan"].probabilities != second.answers["plan"].probabilities


def test_a_different_schema_gets_its_own_prefix():
    """Keyed on the schema hash, which is what a schema block depends on."""
    cached = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    _answer(cached)
    wider = dict(
        REQUEST,
        questions={
            **REQUEST["questions"],
            "size": {
                "type": "score",
                "instructions": "How large?",
                "levels": [{"name": "small", "value": 0.0}, {"name": "big", "value": 1.0}],
            },
        },
    )
    _answer(cached, wider)
    assert len(cached._prefix_cache) == 2


def test_training_never_uses_a_cached_prefix():
    """The failure this guards is silent and would corrupt a gradient.

    A prefix belongs to the weights that produced it, and training mutates the
    weights every step. A cache left on during training serves a schema block
    from an earlier epoch into a later one, and nothing raises — the shapes
    all match.

    Driven through `logits` rather than through the engine, because that is
    the path the trainer takes. `infer` forces eval mode around its call, so
    going through it would test the guard against a condition that can never
    occur there — a test passing for the wrong reason.
    """
    backend = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    compiler = backend.make_compiler()
    request = SystemOneRequest.model_validate(REQUEST)
    compiled = compiler.compile_request(request)

    backend.model.train()
    backend.logits(compiled, request)
    assert backend._prefix_cache == {}, "a training-mode forward must not populate the cache"

    backend.model.eval()
    backend.logits(compiled, request)
    assert len(backend._prefix_cache) == 1


def test_the_batched_training_pass_does_not_touch_the_cache():
    """`logits_batch` is the trainer's real hot path and bypasses it entirely.

    Belt and braces: the training-mode guard already covers it, but a batched
    pass reading a prefix built from one request's mask would be wrong in a
    way that is very hard to see.
    """
    backend = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    compiler = backend.make_compiler()
    request = SystemOneRequest.model_validate(REQUEST)
    items = [(compiler.compile_request(request), request)] * 3

    backend.model.train()
    backend.logits_batch(items)
    assert backend._prefix_cache == {}


def test_it_is_off_by_default():
    """A gateway opts in. Everything else gets the path it already had."""
    backend = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0)
    assert backend.cache_prefixes is False
    _answer(backend)
    assert backend._prefix_cache == {}


def test_the_prefix_really_holds_the_schema_and_nothing_else():
    """It must cover the schema block exactly — not a token more or fewer.

    One token short and a schema position gets recomputed under the wrong
    mask; one token long and a *state* token is cached across requests, which
    would make the answer depend on whichever state arrived first.
    """
    from trigon.schema import SegmentKind

    backend = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    compiler = backend.make_compiler()
    compiled = compiler.compile_request(SystemOneRequest.model_validate(REQUEST))
    _answer(backend)

    expected = sum(
        segment.tokens
        for segment in compiled.segments
        if segment.kind
        in {SegmentKind.SCHEMA_QUESTION, SegmentKind.SCHEMA_OPTION, SegmentKind.SCHEMA_LEVEL}
    )
    prefix = next(iter(backend._prefix_cache.values()))
    assert prefix.tokens == expected
    assert prefix.outputs.shape[1] == expected
    assert len(prefix.layers) == SHAPE["n_layers"]


def test_the_cache_is_bounded():
    """A schema prefix is tens of kilobytes per layer.

    An unbounded cache on a gateway serving many distinct schemas is a slow
    memory leak, which is the kind of bug that only appears in production.
    """
    backend = TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=0, cache_prefixes=True)
    for i in range(70):
        _answer(
            backend,
            dict(
                REQUEST,
                questions={f"q{i}": {"type": "noul", "instructions": f"Is fact {i} true of this?"}},
            ),
        )
    assert len(backend._prefix_cache) <= 64


def test_the_cache_adds_error_on_top_of_whatever_the_hardware_already_does():
    """The trade, measured, on the path a caller actually uses.

    Adding questions must not move the answers to the others. That is a
    property of the block mask: there is no path from one question's tokens to
    another's, and `tests/test_independence.py` asserts the consequence to
    exact equality on the direct engine path.

    **Exact equality on the *served* path is not portable, and this test used
    to claim it was.** It asserted `spread(False) == 0.0`, which holds on every
    sequence length tried on the development machine — 1 to 40 extra questions,
    all exactly zero — and fails on GitHub's runners at 2.4e-08. Same code,
    same mask, different CPU: sequence length changes which GEMM kernel torch
    selects, and a different reduction order rounds differently. The structural
    claim is untouched. The numerical restatement of it was stronger than the
    evidence, and CI is where that showed up, because CI was the second machine
    this had ever run on.

    So what is asserted here is what is true anywhere: the cache's error is
    *additional*, both are far below anything a caller could act on, and the
    honest reason to default the cache off is that it adds error on every
    machine while its wall-clock saving is still unmeasured — not that it
    turns an exact answer into an approximate one, which on some hardware it
    does not.
    """
    from fastapi.testclient import TestClient

    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    crowded = dict(
        REQUEST,
        questions={
            **REQUEST["questions"],
            **{
                f"filler_{i}": {"type": "noul", "instructions": f"Is fact {i} here?"}
                for i in range(6)
            },
        },
    )

    def spread(cache_prefixes: bool) -> float:
        client = TestClient(build_app(ServerConfig(backend="torch", cache_prefixes=cache_prefixes)))
        alone = client.post("/v1/systemone", json=REQUEST).json()["answers"]["plan"]
        among = client.post("/v1/systemone", json=crowded).json()["answers"]["plan"]
        return max(
            abs(alone["probabilities"][k] - among["probabilities"][k])
            for k in alone["probabilities"]
        )

    without, with_cache = spread(False), spread(True)
    # float32 has ~1.2e-07 of relative precision, so anything at or below that
    # scale is the arithmetic rather than a leak. A real failure of the mask
    # would move a probability by a visible amount, not by an ulp.
    assert without < 1e-6, f"without the cache the drift is {without:.3e}, far above rounding"
    assert 0.0 < with_cache < 1e-6, f"with the cache the drift is {with_cache:.3e}"
    assert with_cache > without, (
        "the cache is supposed to cost accuracy, not save it; if it ever stops "
        "costing any, the reason it is off by default has gone away"
    )


def test_the_gateway_defaults_the_cache_off():
    """Nobody gets a weaker promise than the one they read about by accident."""
    from trigon.server.config import ServerConfig

    assert ServerConfig().cache_prefixes is False
    assert ServerConfig.from_env({}).cache_prefixes is False
    assert ServerConfig.from_env({"TRIGON_CACHE_PREFIXES": "1"}).cache_prefixes is True
