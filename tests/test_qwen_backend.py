"""The pretrained-backbone backend, held to the reference model's specification.

`tests/test_independence.py` proves the product's two architectural claims
on the spike. The Qwen2 forward in `trigon.backends.qwen_readout` is new code
that receives the same block mask, so the claims are re-asserted on it here --
on a tiny randomly initialised Qwen2, because the claims are properties of
the layout and the forward, not of trained weights, and a 3 GB download has
no place in a unit test. Arithmetic parity with `transformers` on the real
weights is `scripts/backbone_parity.py`'s job.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the backbone backend needs the 'train' extra")

from conftest import assert_answer_unmoved, assert_evidence_unmoved  # noqa: E402
from trigon.backends.qwen_readout import (  # noqa: E402
    LoRALinear,
    QwenPrefillModel,
    QwenReadoutBackend,
    QwenShape,
)
from trigon.backends.tokenizer import default_tokenizer  # noqa: E402
from trigon.backends.torch_readout import IG_STEPS, TorchReadoutBackend  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals import synthetic_outcome_cases  # noqa: E402
from trigon.schema import SegmentKind  # noqa: E402
from trigon.training import TrainingConfig, train  # noqa: E402
from trigon.types import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

STATE = "The customer writes: my card payment was declined at the store."
BASE = {
    "intent": ChoiceQuestion(
        instructions="Route this ticket.",
        options=[{"name": "card"}, {"name": "luggage"}, {"name": "login"}],
    ),
    "severity": ScoreQuestion(
        instructions="How severe is this?",
        levels=[{"name": "low"}, {"name": "medium"}, {"name": "high"}],
    ),
    "urgent": NoulQuestion(instructions="Does this need a human within the hour?"),
}


def _tiny(seed: int = 0, rank: int = 4, cache: bool = False) -> QwenReadoutBackend:
    tokenizer = default_tokenizer()
    torch.manual_seed(seed)
    shape = QwenShape(
        vocab_size=tokenizer.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=96
    )
    model = QwenPrefillModel(shape, lora_rank=rank, lora_alpha=8.0)
    # The frozen half is "pretrained" here only in the sense of being fixed;
    # drawn once per seed so two backends built alike are alike.
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad and p.dim() > 1:
                p.normal_(std=0.05)
    return QwenReadoutBackend(model, tokenizer, backbone=None, cache_prefixes=cache)


@pytest.fixture(scope="module")
def engine() -> Engine:
    backend = _tiny()
    return Engine(backend, compiler=backend.make_compiler())


def _answers(engine: Engine, questions: dict, state: str = STATE) -> dict:
    response = engine.answer(SystemOneRequest(state=state, questions=questions))
    return {k: v.model_dump() for k, v in response.answers.items()}


def test_adding_a_question_moves_nothing_else(engine):
    before = _answers(engine, BASE)
    after = _answers(
        engine,
        {
            **BASE,
            "unrelated": ChoiceQuestion(
                instructions="Something entirely different about weather.",
                options=[{"name": "rain"}, {"name": "sun"}],
            ),
        },
    )
    for qid in BASE:
        assert_answer_unmoved(before[qid], after[qid], qid)


def test_twenty_extra_questions_and_any_order_move_nothing(engine):
    before = _answers(engine, BASE)
    crowd = {f"filler_{i}": NoulQuestion(instructions=f"Is fact {i} present?") for i in range(20)}
    after = _answers(engine, {**dict(reversed(list(BASE.items()))), **crowd})
    for qid in BASE:
        assert_answer_unmoved(before[qid], after[qid], qid)


def test_schema_states_do_not_depend_on_state(engine):
    """Exact: both sides compile to one shape, so one kernel reduces them."""
    backend, compiler = engine.backend, engine.compiler

    def schema_hidden(state: str) -> torch.Tensor:
        compiled = compiler.compile_request(SystemOneRequest(state=state, questions=BASE))
        embeddings, spans = backend._embed(compiled)
        mask = backend._mask_tensor(compiled)
        with torch.no_grad():
            hidden = backend.model(embeddings, mask, spans.positions(), spans.segment_types())[0]
        schema = sum(
            s.tokens
            for s in compiled.segments
            if s.kind
            in {SegmentKind.SCHEMA_QUESTION, SegmentKind.SCHEMA_OPTION, SegmentKind.SCHEMA_LEVEL}
        )
        return hidden[:schema]

    # Same length on purpose: the claim is about values, not shapes.
    a = schema_hidden("Delayed shipment to Berlin, please advise.")
    b = schema_hidden("Card declined at the till, please advise!!")
    assert a.shape == b.shape
    assert torch.equal(a, b)


def test_the_cached_prefix_path_agrees_with_the_uncached_one():
    plain, cached = _tiny(), _tiny(cache=True)
    request = SystemOneRequest(state=STATE, questions=BASE)
    first = Engine(plain, compiler=plain.make_compiler()).answer(request)
    engine = Engine(cached, compiler=cached.make_compiler())
    engine.answer(request)  # fills the prefix
    second = engine.answer(request)
    assert second.usage.cached_schema_tokens > 0
    for qid in BASE:
        assert_answer_unmoved(
            first.answers[qid].model_dump(), second.answers[qid].model_dump(), qid
        )


def test_the_batched_pass_matches_one_at_a_time():
    backend = _tiny()
    compiler = backend.make_compiler()
    requests = [
        SystemOneRequest(state=s, questions=BASE)
        for s in (STATE, "short", "a much longer message about a refund that never arrived")
    ]
    items = [(compiler.compile_request(r), r) for r in requests]
    with torch.no_grad():
        batched = backend.logits_batch(items)
        alone = [backend.logits(c, r)[0] for c, r in items]
    for one, other in zip(batched, alone, strict=True):
        for qid in one:
            assert torch.allclose(one[qid], other[qid], atol=1e-5)


def test_zero_initialised_adapters_leave_the_backbone_untouched():
    """Before the first step the converted model computes the backbone's forward."""
    with_adapters, without = _tiny(rank=4), _tiny(rank=0)
    # Drawn separately, the two consume the RNG differently; give them one
    # backbone, and the same heads, so only the adapters differ.
    with_adapters.model.load_state_dict(without.model.state_dict(), strict=False)
    compiled = with_adapters.make_compiler().compile_request(
        SystemOneRequest(state=STATE, questions=BASE)
    )
    outputs = []
    for backend in (with_adapters, without):
        embeddings, spans = backend._embed(compiled)
        with torch.no_grad():
            outputs.append(
                backend.model(
                    embeddings,
                    backend._mask_tensor(compiled),
                    spans.positions(),
                    spans.segment_types(),
                )
            )
    assert torch.equal(outputs[0], outputs[1])


def test_training_moves_only_what_is_meant_to_train():
    backend = _tiny()
    frozen = {
        k: v.clone()
        for k, v in backend.model.state_dict().items()
        if k not in backend.model.trainable_state()
    }
    trainable_before = {k: v.clone() for k, v in backend.model.trainable_state().items()}
    report = train(
        backend,
        synthetic_outcome_cases(n=40, seed=0, noise=0.2),
        TrainingConfig(epochs=2, accumulate=8, learning_rate=1e-2, validation_fraction=0),
    )
    assert report.final_loss < report.first_loss
    for key, value in backend.model.state_dict().items():
        if key in frozen:
            assert torch.equal(value, frozen[key]), f"{key} is frozen and moved"
    moved = [
        k
        for k, v in backend.model.trainable_state().items()
        if not torch.equal(v, trainable_before[k])
    ]
    assert any("lora_b" in k for k in moved)
    assert backend.model_version.startswith("trigon-qwen2-custom-0.1.0+")


def test_an_adapter_checkpoint_round_trips_through_the_shared_loader(tmp_path):
    backend = _tiny()
    train(
        backend,
        synthetic_outcome_cases(n=20, seed=1, noise=0.2),
        TrainingConfig(epochs=1, accumulate=8, validation_fraction=0),
    )
    path = tmp_path / "adapter.pt"
    backend.save(path)
    loaded = TorchReadoutBackend.load(path)
    assert isinstance(loaded, QwenReadoutBackend)
    assert loaded.model_version == backend.model_version
    request = SystemOneRequest(state=STATE, questions=BASE)
    a = Engine(backend, compiler=backend.make_compiler()).answer(request)
    b = Engine(loaded, compiler=loaded.make_compiler()).answer(request)
    for qid in BASE:
        assert a.answers[qid].model_dump() == b.answers[qid].model_dump()


# -- evidence, on the backbone's forward -------------------------------------


@pytest.mark.parametrize("method", ["gradient_x_input", "integrated_gradients", "span_head"])
def test_evidence_does_not_move_when_a_question_is_added(method):
    """The spike's evidence claim, re-asserted on the Qwen2 forward, in
    float64 for the reason `tests/test_independence.py` gives."""
    backend = _tiny()
    backend.model.double()
    backend.evidence_mode = method
    compiler = backend.make_compiler()

    def evidence(questions):
        request = SystemOneRequest(
            state=STATE, questions=questions, options={"include_evidence": True}
        )
        output = backend.infer(compiler.compile_request(request), request)
        return {qid: out.evidence for qid, out in output.outputs.items()}

    before = evidence(BASE)
    after = evidence({**BASE, "extra": NoulQuestion(instructions="Is the weather nice?")})
    for qid in BASE:
        assert_evidence_unmoved(before[qid], after[qid], qid, bound=1e-12)


def test_integrated_gradients_through_the_cached_prefix_is_complete_and_unbatched():
    """The backbone's cached forward broadcasts one schema prefix over a batch
    of path steps -- new code on this forward. The cached path and the
    uncached one agree, a step per pass agrees with every step in one pass,
    and the attributions sum to the log-probability difference at the served
    step count -- which on a pre-norm forward needs the points crowded towards
    the baseline: evenly spaced, the same 32 miss by more than a tenth."""
    request = SystemOneRequest(state=STATE, questions=BASE)
    results = {}
    for cache, chunk in ((False, 32), (True, 1), (True, 32)):
        backend = _tiny(cache=cache)
        backend.model.double()
        backend.ig_chunk = chunk
        compiled = backend.make_compiler().compile_request(request)
        results[cache, chunk] = backend.integrated_gradients(compiled, request)
    served = results[False, 32]
    for key, got in results.items():
        for qid in BASE:
            assert torch.allclose(got[qid], served[qid], rtol=0, atol=1e-12), (key, qid)

    backend = _tiny()
    backend.model.double()
    compiled = backend.make_compiler().compile_request(request)
    backend.ig_power = 1
    uniform = backend.integrated_gradients(compiled, request)
    for qid, delta in backend.path_difference(compiled, request).items():
        assert abs(float(served[qid].sum()) - delta) <= 1e-2 * abs(delta), qid
        assert abs(float(uniform[qid].sum()) - delta) > 0.1 * abs(delta), qid


# -- integrated gradients' precision ------------------------------------------


def _as_on_the_gpu(backend: QwenReadoutBackend) -> QwenReadoutBackend:
    """The tiny Qwen2 run the way the GPU runs the real one: frozen projections
    stored in bf16, and every layer under bf16 autocast."""
    for module in backend.model.modules():
        if isinstance(module, LoRALinear):
            module.base.to(torch.bfloat16)
    backend.model.autocast_cpu = True
    backend.model.eval()
    return backend


def _completeness(backend, precision: str, steps: int | None = None) -> dict[str, float]:
    request = SystemOneRequest(state=STATE, questions=BASE)
    compiled = backend.make_compiler().compile_request(request)
    backend.ig_precision = precision
    backend.ig_steps = steps or IG_STEPS
    attributions = backend.integrated_gradients(compiled, request)
    return {
        qid: abs(float(attributions[qid].sum()) - delta) / abs(delta)
        for qid, delta in backend.path_difference(compiled, request).items()
    }


def test_integrated_gradients_is_complete_in_float32_and_not_under_bf16_autocast():
    """bf16's share of the GPU's completeness failure, reproduced on a CPU and
    removed. (It is not the whole failure on the real backbone, whose path is
    rough: `docs/architecture.md`, *Evidence*.)

    A token's attribution is the input dotted with an integrated gradient, and
    the log-probability difference they must add up to is what is left when
    large terms of both signs cancel. A gradient carried through bf16 is off by
    a relative error per term that the remainder does not share: here the
    attributions' absolute sum is ~120x the difference, and bf16 misses it by
    15% where float32 misses by 0.04%. Four times the points do not help,
    which is what separates rounding from quadrature."""
    backend = _as_on_the_gpu(_tiny(seed=4))
    served = _completeness(backend, "float32")
    assert max(served.values()) < 1e-3, served
    bf16 = _completeness(backend, "autocast")
    assert max(bf16.values()) > 0.05, bf16
    finer = _completeness(backend, "autocast", steps=4 * IG_STEPS)
    assert max(finer.values()) > 0.05, finer


def test_the_float32_path_is_the_float32_model_on_the_same_weights():
    """Upcasting on the fly (`_UpcastLinear`) computes exactly what a float32
    copy of the rounded weights would, forward and backward, at a fixed shape
    -- without holding that copy."""
    request = SystemOneRequest(state=STATE, questions=BASE)
    upcast = _as_on_the_gpu(_tiny(seed=4))
    copied = _tiny(seed=4)
    copied.model.eval()
    with torch.no_grad():
        for mine, theirs in zip(copied.model.modules(), upcast.model.modules(), strict=True):
            if isinstance(mine, LoRALinear):
                mine.base.weight.copy_(theirs.base.weight.float())
                if mine.base.bias is not None:
                    mine.base.bias.copy_(theirs.base.bias.float())
    compiled = upcast.make_compiler().compile_request(request)
    got = upcast.integrated_gradients(compiled, request)
    want = copied.integrated_gradients(compiled, request)
    for qid in BASE:
        assert torch.equal(got[qid], want[qid]), qid
    assert upcast.path_difference(compiled, request) == copied.path_difference(compiled, request)


def test_the_float32_path_leaves_the_served_prefix_and_answer_alone():
    """The served schema prefix is bf16 and stays in the cache, untouched;
    the float32 path builds its own and drops it. The answer, and gradient x
    input, keep the served precision."""
    backend = _as_on_the_gpu(_tiny(seed=4, cache=True))
    engine = Engine(backend, compiler=backend.make_compiler())
    plain = _answers(engine, BASE)
    cached = dict(backend._prefix_cache)
    assert cached
    request = SystemOneRequest(state=STATE, questions=BASE)
    compiled = backend.make_compiler().compile_request(request)
    backend.integrated_gradients(compiled, request)
    backend.path_difference(compiled, request)
    assert backend._prefix_cache.keys() == cached.keys()
    assert all(backend._prefix_cache[k] is v for k, v in cached.items())
    assert _answers(engine, BASE) == plain
    assert all(k.dtype == torch.bfloat16 for k, _ in next(iter(cached.values())).layers)


def test_integrated_gradients_on_the_float32_path_does_not_move_when_a_question_is_added():
    """Independence on the new path: bf16 frozen weights upcast on the fly, in
    float64 for the cross-shape bound, as the other evidence tests are."""
    backend = _tiny()
    backend.model.double()
    for module in backend.model.modules():
        if isinstance(module, LoRALinear):
            module.base.to(torch.bfloat16)
    backend.evidence_mode = "integrated_gradients"
    assert backend.ig_precision == "float32"
    compiler = backend.make_compiler()

    def evidence(questions):
        request = SystemOneRequest(
            state=STATE, questions=questions, options={"include_evidence": True}
        )
        output = backend.infer(compiler.compile_request(request), request)
        return {qid: out.evidence for qid, out in output.outputs.items()}

    before = evidence(BASE)
    after = evidence({**BASE, "extra": NoulQuestion(instructions="Is the weather nice?")})
    for qid in BASE:
        assert_evidence_unmoved(before[qid], after[qid], qid, bound=1e-12)


def test_the_evidence_head_trains_and_round_trips_on_the_backbone(tmp_path):
    """A rationale trains the adapter checkpoint's evidence head, and the
    checkpoint remembers it did -- the flag decides which evidence it serves."""
    import dataclasses

    from trigon.schema import render_state

    cases = []
    for case in synthetic_outcome_cases(n=12, seed=2, noise=0.0):
        text = render_state(case.request.state)
        where = text.index('"plan"')
        expected = dict(case.expected)
        expected["plan"] = dataclasses.replace(expected["plan"], rationale=((where, where + 6),))
        cases.append(dataclasses.replace(case, expected=expected))

    backend = _tiny()
    report = train(backend, cases, TrainingConfig(epochs=1, accumulate=4, validation_fraction=0))
    assert report.n_rationales == len(cases)
    assert backend.config.evidence_supervised
    path = tmp_path / "adapter.pt"
    backend.save(path)
    loaded = TorchReadoutBackend.load(path)
    assert loaded.config.evidence_supervised
    request = SystemOneRequest(state=STATE, questions=BASE, options={"include_evidence": True})
    answer = Engine(loaded, compiler=loaded.make_compiler()).answer(request)
    assert {a.evidence_method for a in answer.answers.values()} == {"span_head"}
