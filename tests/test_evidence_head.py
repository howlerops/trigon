"""The evidence head: trained on rationales, or never used.

An evidence head nobody trained is a random projection, and its spans would
look as confident as a trained one's. So the method a checkpoint serves is
decided by whether it was fitted to human rationales -- recorded by the
trainer, saved with the checkpoint -- and these tests hold that switch to its
word, along with the claim that supervision actually teaches the head.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")

from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals.harness import Case, Expectation  # noqa: E402
from trigon.training import TrainingConfig, train  # noqa: E402
from trigon.types import NoulQuestion, SystemOneRequest  # noqa: E402

SHAPE = dict(d_model=64, n_layers=2, n_heads=2, d_ff=64)
FRUIT = ["apple", "banana", "cherry", "mango", "peach", "plum"]
FILLER = ["table", "window", "river", "pencil", "cloud", "engine", "garden", "ladder"]


def _cases(n: int, seed: int, *, rationale: bool = True) -> list[Case]:
    """A fruit is mentioned, or it is not; the rationale is the fruit."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        words = rng.sample(FILLER, 5)
        has = i % 2 == 0
        if has:
            words.insert(rng.randrange(6), rng.choice(FRUIT))
        state = "the " + " the ".join(words)
        spans = ()
        if has:
            fruit = next(w for w in words if w in FRUIT)
            at = state.index(fruit)
            spans = ((at, at + len(fruit)),)
        out.append(
            Case(
                case_id=f"fruit/{i}",
                request=SystemOneRequest(
                    state=state,
                    questions={"fruit": NoulQuestion(instructions="Is a fruit mentioned?")},
                ),
                expected={
                    "fruit": Expectation(
                        probability=float(has), rationale=spans if rationale else None
                    )
                },
            )
        )
    return out


def _backend(seed: int = 0) -> TorchReadoutBackend:
    return TorchReadoutBackend(ReadoutConfig(**SHAPE), seed=seed)


def _methods(backend) -> set[str]:
    request = SystemOneRequest.model_validate(
        {**_cases(1, 99)[0].request.model_dump(), "options": {"include_evidence": True}}
    )
    answers = Engine(backend, compiler=backend.make_compiler()).answer(request).answers
    return {a.evidence_method for a in answers.values()}


def test_a_model_never_shown_a_rationale_serves_gradient_x_input():
    backend = _backend()
    report = train(
        backend,
        _cases(16, 1, rationale=False),
        TrainingConfig(epochs=1, accumulate=8, validation_fraction=0),
    )
    assert report.n_rationales == 0
    assert not backend.config.evidence_supervised
    assert _methods(backend) == {"gradient_x_input"}


def test_a_rationale_weight_of_zero_trains_no_head_even_with_rationales():
    backend = _backend()
    report = train(
        backend,
        _cases(16, 1),
        TrainingConfig(epochs=1, accumulate=8, validation_fraction=0, rationale_weight=0.0),
    )
    assert report.n_rationales == 0
    assert _methods(backend) == {"gradient_x_input"}


def test_rationales_switch_the_checkpoint_to_the_span_head_and_it_stays_switched(tmp_path):
    backend = _backend()
    report = train(
        backend, _cases(16, 1), TrainingConfig(epochs=1, accumulate=8, validation_fraction=0)
    )
    assert report.n_rationales == 16
    assert _methods(backend) == {"span_head"}
    path = tmp_path / "run.pt"
    backend.save(path)
    loaded = TorchReadoutBackend.load(path)
    assert loaded.config.evidence_supervised
    assert _methods(loaded) == {"span_head"}


def test_a_checkpoint_from_before_the_head_existed_loads_and_attributes(tmp_path):
    """The certified runs were saved without the head or the flag: they must
    still load, and serve gradient x input, never an untrained head."""
    backend = _backend()
    path = tmp_path / "old.pt"
    backend.save(path)
    payload = torch.load(path, weights_only=False)
    payload["config"].pop("evidence_supervised")
    for name in ("evidence_query.weight", "evidence_key.weight", "evidence_bias"):
        payload["state_dict"].pop(name)
    torch.save(payload, path)
    loaded = TorchReadoutBackend.load(path)
    assert not loaded.config.evidence_supervised
    assert _methods(loaded) == {"gradient_x_input"}


def test_supervision_teaches_the_head_where_the_fruit_is():
    """Not a benchmark, a mechanism check: on held-out states, the fruit's
    tokens score above every other token far more often than chance."""
    backend = _backend(seed=3)
    train(
        backend,
        _cases(96, 1),
        TrainingConfig(epochs=12, accumulate=8, learning_rate=3e-3, validation_fraction=0, seed=3),
    )
    engine = Engine(backend, compiler=backend.make_compiler())
    held_out = [c for c in _cases(40, 7) if c.expected["fruit"].rationale]
    top_is_fruit = 0
    for case in held_out:
        request = SystemOneRequest.model_validate(
            {**case.request.model_dump(), "options": {"include_evidence": True}}
        )
        compiled = engine.compiler.compile_request(request)
        tokens = backend.infer(compiled, request).outputs["fruit"].evidence
        best = max(tokens, key=lambda t: t[2])
        ((start, end),) = case.expected["fruit"].rationale
        top_is_fruit += int(best[0] < end and start < best[1])
    # Five other words, each after a "the": chance is about 1 in 11 tokens,
    # and this measured 1.0 when it was written.
    assert top_is_fruit / len(held_out) > 0.8


def test_evidence_requests_in_a_batch_are_answered_as_they_would_be_alone():
    backend = _backend()
    compiler = backend.make_compiler()
    requests = [
        SystemOneRequest.model_validate(
            {**c.request.model_dump(), "options": {"include_evidence": True}}
        )
        for c in _cases(3, 5)
    ]
    items = [(compiler.compile_request(r), r) for r in requests]
    batched = backend.infer_many(items)
    for (compiled, request), output in zip(items, batched, strict=True):
        alone = backend.infer(compiled, request)
        assert output.outputs["fruit"].evidence == alone.outputs["fruit"].evidence


def test_the_span_head_leaves_the_answer_exactly_as_it_was():
    """It needs no gradients, so it stays on the no-grad path the plain answer
    takes: same shape, same kernels, so exact -- unlike gradient x input,
    which leaves the encoder's fast path and agrees only to rounding."""
    backend = _backend()
    backend.config.evidence_supervised = True
    engine = Engine(backend, compiler=backend.make_compiler())
    request = _cases(1, 4)[0].request
    plain = engine.answer(request).answers["fruit"]
    explained = engine.answer(
        SystemOneRequest.model_validate(
            {**request.model_dump(), "options": {"include_evidence": True}}
        )
    ).answers["fruit"]
    assert explained.evidence_method == "span_head"
    assert explained.probability == plain.probability
