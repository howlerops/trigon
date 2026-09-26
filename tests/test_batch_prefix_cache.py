"""The batched serving path reads the schema KV cache too.

For a long time it did not. `Engine.answer_many` padded whole sequences and
recomputed the schema block for every request in the batch, so the preliminary
L4 burn-in (`reports/burn-in/modal-l4/`) computed every billed token at batch
8 and 32 where batch 1 computed 3% of them -- and the certified model served
fewer requests a second batched than one at a time would have at batch 1's
latency.

`TorchReadoutBackend.infer_many` now groups a batch by schema and runs each
group as one padded pass over the state and readout tokens only, against one
cached prefix broadcast across the group. These tests are its specification,
on both the spike and the backbone's forward (a tiny random Qwen2, as in
`tests/test_qwen_backend.py`):

- **Batched cached answers equal one-at-a-time cached answers** -- to the
  float32 bound, because a padded batch is a different shape from a single
  request and a different shape can reduce in a different order
  (`conftest.assert_answer_unmoved`). Where the shapes do match -- the prefix
  itself, and one request batched beside two different neighbours of the same
  length -- the comparison is exact.
- **A mixed-schema batch is correct**, one prefix per schema.
- **`cached_schema_tokens` is per request**, and matches what one-at-a-time
  serving of the same requests reports.
- **Independence still holds** across requests and across questions.
- **Training never reads or fills the cache**, on this path either.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")

from conftest import assert_answer_unmoved  # noqa: E402
from trigon.backends.qwen_readout import (  # noqa: E402
    QwenPrefillModel,
    QwenReadoutBackend,
    QwenShape,
)
from trigon.backends.tokenizer import default_tokenizer  # noqa: E402
from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.types import (  # noqa: E402
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)

ROUTE = {
    "intent": ChoiceQuestion(
        instructions="Which banking intent does this message express?",
        options=[{"name": n} for n in ("card declined", "refund", "login", "double charge")],
    ),
    "urgent": NoulQuestion(instructions="Does this need a human within the hour?"),
}
SIZE = {
    "size": ScoreQuestion(
        instructions="How large is this account?",
        levels=[{"name": n} for n in ("small", "medium", "large")],
    ),
}
STATES = [
    "my card payment was declined at the till and I do not know why",
    "where is the refund",
    "the app will not let me log in after I changed my phone, " * 3,
    "I was charged twice for the same transaction this morning",
]


def _spike(cache: bool) -> TorchReadoutBackend:
    return TorchReadoutBackend(
        ReadoutConfig(d_model=64, n_layers=2, n_heads=2, d_ff=64), seed=0, cache_prefixes=cache
    )


def _qwen(cache: bool) -> QwenReadoutBackend:
    tokenizer = default_tokenizer()
    torch.manual_seed(0)
    shape = QwenShape(
        vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=96
    )
    model = QwenPrefillModel(shape, lora_rank=4, lora_alpha=8.0)
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad and p.dim() > 1:
                p.normal_(std=0.05)
    return QwenReadoutBackend(model, tokenizer, backbone=None, cache_prefixes=cache)


BACKENDS = {"spike": _spike, "qwen": _qwen}


def _engine(kind: str, cache: bool = True) -> Engine:
    backend = BACKENDS[kind](cache)
    return Engine(backend, compiler=backend.make_compiler())


def _dump(response) -> dict:
    return {qid: answer.model_dump() for qid, answer in response.answers.items()}


def _schema_tokens(engine: Engine, request: DecisionRequest) -> int:
    return engine.backend._schema_tokens(engine.compiler.compile_request(request))


@pytest.fixture(params=sorted(BACKENDS))
def kind(request) -> str:
    return request.param


def test_a_batch_on_one_schema_reads_the_prefix_and_answers_as_one_at_a_time_does(kind):
    """The burn-in's shape: one schema, several states, different lengths.

    Float32 bound: the batch pads every request to the longest, so each is
    computed at a different shape from its single-request pass.
    """
    requests = [DecisionRequest(state=s, questions=ROUTE) for s in STATES]
    single, batched = _engine(kind), _engine(kind)

    alone = [single.answer(r) for r in requests]
    together = batched.answer_many(requests)

    for one, many in zip(alone, together, strict=True):
        before, after = _dump(one), _dump(many)
        for qid in ROUTE:
            assert_answer_unmoved(before[qid], after[qid], qid)
    assert len(batched.backend._prefix_cache) == 1


def test_the_prefix_a_batch_fills_is_the_prefix_the_single_path_fills_exactly(kind):
    """Same inputs, same shape, so exact -- and it is what makes the batched
    path's positions and mask semantics the single path's by construction:
    the prefix is filled from one request's own unpadded tensors, never from
    a padded batch row."""
    requests = [DecisionRequest(state=s, questions=ROUTE) for s in STATES]
    single, batched = _engine(kind), _engine(kind)
    single.answer(requests[0])
    batched.answer_many(requests)

    (a,) = single.backend._prefix_cache.values()
    (b,) = batched.backend._prefix_cache.values()
    assert a.schema_hash == b.schema_hash and a.tokens == b.tokens
    assert torch.equal(a.outputs, b.outputs)
    for x, y in zip(a.layers, b.layers, strict=True):
        pairs = zip(x, y, strict=True) if isinstance(x, tuple) else [(x, y)]
        for left, right in pairs:
            assert torch.equal(left, right)


def test_the_cached_batch_agrees_with_the_uncached_batch(kind):
    """The two batched arms: the cache moves no answer beyond rounding."""
    requests = [DecisionRequest(state=s, questions=ROUTE) for s in STATES]
    plain = _engine(kind, cache=False).answer_many(requests)
    cached = _engine(kind).answer_many(requests)
    for one, many in zip(plain, cached, strict=True):
        before, after = _dump(one), _dump(many)
        for qid in ROUTE:
            assert_answer_unmoved(before[qid], after[qid], qid)
    assert all(r.usage.cached_schema_tokens == 0 for r in plain)


def test_a_mixed_schema_batch_is_one_pass_per_schema_and_still_correct(kind):
    requests = [
        DecisionRequest(state=STATES[0], questions=ROUTE),
        DecisionRequest(state=STATES[1], questions=SIZE),
        DecisionRequest(state=STATES[2], questions=ROUTE),
        DecisionRequest(state=STATES[3], questions=SIZE),
        DecisionRequest(state=STATES[1], questions={**ROUTE, **SIZE}),
    ]
    single, batched = _engine(kind), _engine(kind)
    alone = [single.answer(r) for r in requests]

    compiled = [batched.compiler.compile_request(r) for r in requests]
    outputs = batched.backend.infer_many(list(zip(compiled, requests, strict=True)))
    assert {o.diagnostics["passes"] for o in outputs} == {3}
    assert len(batched.backend._prefix_cache) == 3

    together = batched.answer_many(requests)
    for request, one, many in zip(requests, alone, together, strict=True):
        before, after = _dump(one), _dump(many)
        assert before.keys() == after.keys() == request.questions.keys()
        for qid in request.questions:
            assert_answer_unmoved(before[qid], after[qid], qid)


def test_cached_schema_tokens_is_reported_per_request_as_one_at_a_time_would(kind):
    """Only a hit counts. On a miss the group's first request fills the
    prefix and paid for it; every other request on that schema read it."""
    requests = [
        DecisionRequest(state=STATES[0], questions=ROUTE),
        DecisionRequest(state=STATES[1], questions=ROUTE),
        DecisionRequest(state=STATES[2], questions=SIZE),
        DecisionRequest(state=STATES[3], questions=ROUTE),
        DecisionRequest(state=STATES[0], questions=SIZE),
    ]
    single, batched = _engine(kind), _engine(kind)
    route, size = _schema_tokens(batched, requests[0]), _schema_tokens(batched, requests[2])
    assert route > 0 and size > 0 and route != size

    first = [r.usage.cached_schema_tokens for r in batched.answer_many(requests)]
    assert first == [0, route, 0, route, size]
    assert first == [single.answer(r).usage.cached_schema_tokens for r in requests]

    again = batched.answer_many(requests)
    assert [r.usage.cached_schema_tokens for r in again] == [route, route, size, route, size]
    for response in again:
        # What the burn-in's "computed tok/s" column reads: at most the state
        # and the readouts, never the schema a second time.
        computed = response.usage.prefill_tokens - response.usage.cached_schema_tokens
        assert 0 < computed == response.usage.state_tokens + response.usage.readout_tokens


def test_a_request_does_not_hear_its_neighbours_exactly_at_a_fixed_shape(kind):
    """Independence across requests, at the one comparison that can be exact.

    The same request batched beside two different neighbours of the same
    length: every tensor in both batches has the same shape, so the same
    kernel reduces the same way and any difference at all is a leak.
    """
    engine = _engine(kind)
    target = DecisionRequest(state=STATES[0], questions=ROUTE)
    one = DecisionRequest(state="the card was declined", questions=ROUTE)
    other = DecisionRequest(state="the card was blocked", questions=ROUTE)
    lengths = {engine.compiler.compile_request(r).total_tokens for r in (one, other)}
    assert len(lengths) == 1, "the neighbours must be the same length for this to be exact"

    engine.answer(target)  # fill the prefix, so both batches read it
    a = engine.answer_many([target, one])
    b = engine.answer_many([target, other])
    assert _dump(a[0]) == _dump(b[0]), "an answer moved with the company it kept"
    # And the neighbours themselves differ, so the batch is not serving one
    # state's answer to every request that shares its schema.
    assert _dump(a[1]) != _dump(b[1])


def test_adding_questions_moves_no_other_answer_on_the_batched_cached_path(kind):
    """Per-question independence, re-asserted on the path that now serves.

    A different schema is a different prefix and a different length, so this
    is the float32 bound, as in `tests/test_independence.py`.
    """
    engine = _engine(kind)
    crowd = {f"filler_{i}": NoulQuestion(instructions=f"Is fact {i} present?") for i in range(8)}
    small = [DecisionRequest(state=s, questions=ROUTE) for s in STATES]
    large = [DecisionRequest(state=s, questions={**crowd, **ROUTE}) for s in STATES]

    before = engine.answer_many(small)
    after = engine.answer_many(small[:2] + large + small[2:])
    after = after[:2] + after[6:]
    crowded = engine.answer_many(large)
    for one, two, three in zip(before, after, crowded, strict=True):
        for qid in ROUTE:
            assert_answer_unmoved(_dump(one)[qid], _dump(two)[qid], qid)
            assert_answer_unmoved(_dump(one)[qid], _dump(three)[qid], qid)


def test_a_training_mode_batch_neither_reads_nor_fills_the_cache(kind):
    """`infer_many` switches to eval for its own pass. The guard is decided
    before that switch, because what it guards is the weights: a model in
    training mode has weights that move every step, and nothing would raise
    if a prefix from one step were served into the next."""
    engine = _engine(kind)
    backend = engine.backend
    requests = [DecisionRequest(state=s, questions=ROUTE) for s in STATES]
    items = [(engine.compiler.compile_request(r), r) for r in requests]

    backend.model.train()
    outputs = backend.infer_many(items)
    assert backend.model.training, "the caller's mode must be restored"
    assert backend._prefix_cache == {}
    assert all(o.cached_schema_tokens == 0 for o in outputs)

    backend.model.eval()
    backend.infer_many(items)
    assert len(backend._prefix_cache) == 1


def test_a_batch_with_the_cache_off_computes_everything_and_says_so(kind):
    engine = _engine(kind, cache=False)
    responses = engine.answer_many([DecisionRequest(state=s, questions=ROUTE) for s in STATES])
    assert engine.backend._prefix_cache == {}
    assert all(r.usage.cached_schema_tokens == 0 for r in responses)
