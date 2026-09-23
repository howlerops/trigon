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

from conftest import assert_answer_unmoved  # noqa: E402
from trigon.backends.qwen_readout import (  # noqa: E402
    QwenPrefillModel,
    QwenReadoutBackend,
    QwenShape,
)
from trigon.backends.tokenizer import default_tokenizer  # noqa: E402
from trigon.backends.torch_readout import TorchReadoutBackend  # noqa: E402
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
