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


def test_neither_path_lets_a_question_reach_another_and_neither_is_exact():
    """The trade, measured, on the path a caller actually uses — and the
    measurement that removed the reason the cache is off by default.

    This test has now been wrong twice in the same direction, which is worth
    stating before the assertions.

    It first asserted `spread(False) == 0.0`: that without the cache, adding
    questions moves the others by exactly nothing. True here at every length
    tried, 1 to 40 extra questions. On GitHub's runners it is 2.4e-08.

    It was then rewritten to assert that the cache's drift is *larger* —
    keeping the story that the cache trades exactness for compute. On
    GitHub's runners the cache's drift is **0.0** and the uncached path's is
    not. The ordering is not stable either; it is whichever shape happens to
    land on a kernel that reduces in the same order.

    So the honest statement is the one below and nothing more: neither path
    lets a question reach another, and neither is exactly reproducible across
    shapes. `docs/decisions.md` used to justify defaulting the cache off on
    the grounds that it "turns a guarantee into a tolerance". That was never
    the difference between the two paths, and on some hardware it is backwards.
    The cache is still off by default for a reason that survives measurement:
    **its wall-clock saving has never been measured** (`docs/next.md` B.3),
    and an optimisation nobody has timed should not be on by default.
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
    # float32 carries ~1.2e-07 of relative precision, so anything at or below
    # this scale is the arithmetic. A question actually reading another
    # question's tokens would move a probability by a visible fraction.
    assert without < 1e-6, f"without the cache the drift is {without:.3e}, not rounding"
    assert with_cache < 1e-6, f"with the cache the drift is {with_cache:.3e}, not rounding"


def test_the_gateway_defaults_the_cache_on_and_says_so():
    """Default flipped on measurement, and the flip is visible.

    It was off for two reasons. The first -- that the cache traded exact
    independence for compute -- turned out to be wrong: on GitHub's runners
    the cached path is the exact one. The second -- that nobody had timed it
    -- was honest, and `reports/cache/README.md` retired it: 6x at the shape
    the certified Banking77 model serves, 23x at 256 options.

    `/healthz` reports which way it is set for the same reason it reports
    whether the deployment is calibrated. The cache moves answers by ~5e-08
    and wall clock by 6x, and an operator comparing two deployments should not
    have to guess which of them is running it.
    """
    from trigon.server.config import ServerConfig

    assert ServerConfig().cache_prefixes is True
    assert ServerConfig.from_env({}).cache_prefixes is True
    assert ServerConfig.from_env({"TRIGON_CACHE_PREFIXES": "0"}).cache_prefixes is False
    assert ServerConfig.from_env({"TRIGON_CACHE_PREFIXES": "false"}).cache_prefixes is False


def test_healthz_reports_whether_the_cache_is_running():
    from fastapi.testclient import TestClient

    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    for enabled in (True, False):
        with TestClient(build_app(ServerConfig(backend="lexical", cache_prefixes=enabled))) as http:
            assert http.get("/healthz").json()["schema_cache"] is enabled
