"""The architectural claims, proved on the reference model.

These are the two properties the product rests on, and both are properties of
the layout rather than of training, so they hold on an untrained model:

1. adding, removing or reordering questions does not move any other question's
   answer (no context rot from batching, and Noul independence);
2. the schema half of the sequence encodes identically regardless of state,
   which is what makes it a cacheable cross-request KV prefix.

If either of these ever fails, the serving story and the "added questions are
free" claim both fail with it.

**Exact equality is asserted where the tensors have the same shape, and a
float32 bound where they do not, and that distinction was learned the hard
way.** Every one of these asserted exact equality and passed on the machine
they were written on. The first time CI ran them on different hardware,
`test_twenty_extra_questions_still_move_nothing` failed by 1.4e-08 -- because
adding questions lengthens the sequence, a different sequence length selects a
different GEMM kernel, and a different reduction order rounds differently.

The mask is what guarantees independence: there is no path from one question's
tokens to another's, and that is exact and provable from the layout. What is
*not* portable is the arithmetic's reproduction of it across shapes. A real
leak would move a probability by orders of magnitude more than an ulp, so the
bound below is still a regression test with teeth -- it is simply a claim the
evidence supports on more than one machine.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")

from conftest import assert_answer_unmoved, assert_evidence_unmoved  # noqa: E402
from trigon.backends.torch_readout import TorchReadoutBackend  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.schema import SegmentKind, materialize_mask  # noqa: E402
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


@pytest.fixture(scope="module")
def engine() -> Engine:
    backend = TorchReadoutBackend(seed=0)
    return Engine(backend, compiler=backend.make_compiler())


def _answers(engine: Engine, questions: dict) -> dict:
    response = engine.answer(SystemOneRequest(state=STATE, questions=questions))
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


def test_removing_every_other_question_moves_nothing(engine):
    before = _answers(engine, BASE)
    for qid in BASE:
        alone = _answers(engine, {qid: BASE[qid]})
        assert_answer_unmoved(before[qid], alone[qid], qid)


def test_question_map_order_does_not_matter(engine):
    before = _answers(engine, BASE)
    reordered = _answers(engine, dict(reversed(list(BASE.items()))))
    for qid in BASE:
        assert_answer_unmoved(before[qid], reordered[qid], qid)


def test_twenty_extra_questions_still_move_nothing(engine):
    """The batching claim at the scale the cookbooks use it."""
    before = _answers(engine, {"urgent": BASE["urgent"]})
    crowd = {
        f"filler_{i}": NoulQuestion(instructions=f"Is fact number {i} present?") for i in range(20)
    }
    after = _answers(engine, {"urgent": BASE["urgent"], **crowd})
    assert_answer_unmoved(before["urgent"], after["urgent"], "urgent")


def test_schema_states_do_not_depend_on_state(engine):
    """The cacheability claim, checked on hidden states rather than answers.

    Exact, and it stays exact: both sides compile to the same shape, so the
    same kernel reduces in the same order on any hardware. The only thing that
    differs is the *values* in the state tokens, and the claim is precisely
    that those do not reach the schema half.
    """
    backend = engine.backend
    compiler = engine.compiler

    def schema_hidden(state: str) -> torch.Tensor:
        compiled = compiler.compile_request(SystemOneRequest(state=state, questions=BASE))
        embeddings, spans = backend._embed(compiled)
        mask = torch.tensor(materialize_mask(compiled), dtype=torch.bool)
        with torch.no_grad():
            hidden = backend.model(embeddings, mask, spans.positions(), spans.segment_types())[0]
        schema_tokens = sum(
            s.tokens
            for s in compiled.segments
            if s.kind
            in {
                SegmentKind.SCHEMA_QUESTION,
                SegmentKind.SCHEMA_OPTION,
                SegmentKind.SCHEMA_LEVEL,
            }
        )
        return hidden[:schema_tokens]

    a = schema_hidden(STATE)
    b = schema_hidden("An entirely different message about a delayed shipment to Berlin.")
    assert torch.equal(a, b)


def test_output_shapes_always_match_the_declared_label_sets(engine):
    questions = {
        "many": ChoiceQuestion(
            instructions="Pick one.",
            options=[{"name": f"option_{i}"} for i in range(120)],
        ),
        "few": ChoiceQuestion(instructions="Pick one.", options=[{"name": "a"}, {"name": "b"}]),
        "levels": ScoreQuestion(instructions="Rate.", levels=[{"name": str(i)} for i in range(7)]),
    }
    answers = _answers(engine, questions)
    assert len(answers["many"]["probabilities"]) == 120
    assert len(answers["few"]["probabilities"]) == 2
    assert len(answers["levels"]["probabilities"]) == 7
    for answer in answers.values():
        assert sum(answer["probabilities"].values()) == pytest.approx(1.0)


# -- the dot-product head's repairs must not cost the guarantees -------------


@pytest.mark.parametrize(
    "flags",
    [
        {"match_residual": True},
        {"match_normalize": True},
        {"match_residual": True, "match_normalize": True},
    ],
    ids=["residual", "normalize", "both"],
)
def test_dot_product_repairs_keep_option_keys_independent_of_state(flags):
    """The keys are what the schema prefix cache would hold.

    `match_residual` adds each option's input embedding to its key, which is a
    new path from the sequence into the head — and a head that reached the
    state through it would silently make the schema prefix uncacheable while
    every other test still passed. The keys must be a function of the schema
    alone, exactly.
    """
    from trigon.backends.torch_readout import ReadoutConfig
    from trigon.schema import OptionScoring
    from trigon.schema.compiler import materialize_mask

    backend = TorchReadoutBackend(ReadoutConfig(**flags), seed=0)
    compiler = backend.make_compiler(option_scoring=OptionScoring.DOT_PRODUCT)
    states = [
        "the card payment failed twice this week",
        "everything is fine and the customer is delighted",
        "a parcel went missing somewhere near Leeds",
    ]

    keys = []
    with torch.no_grad():
        for state in states:
            compiled = compiler.compile_request(
                SystemOneRequest(state=state, questions={"intent": BASE["intent"]})
            )
            embeddings, spans = backend._embed(compiled)
            mask = torch.tensor(materialize_mask(compiled), dtype=torch.bool)
            hidden = backend.model(embeddings, mask, spans.positions(), spans.segment_types())[0]
            members = torch.stack([hidden[i].mean(dim=0) for i in spans.members["intent"]])
            if flags.get("match_residual"):
                members = members + torch.stack(spans.member_seed_rows["intent"])
            keys.append(backend.model.match_key(members))

    for other in keys[1:]:
        assert torch.equal(keys[0], other), "option keys moved with the state"


@pytest.mark.parametrize("flags", [{"match_residual": True}, {"match_normalize": True}])
def test_dot_product_repairs_survive_a_checkpoint_round_trip(flags, tmp_path):
    """The flags change the head's arithmetic, so a checkpoint that forgets
    them reloads as a different model under the same weights."""
    from trigon.backends.torch_readout import ReadoutConfig

    original = TorchReadoutBackend(ReadoutConfig(**flags), seed=0)
    path = tmp_path / "arm.pt"
    original.save(path)
    reloaded = TorchReadoutBackend.load(path)
    for key, value in flags.items():
        assert getattr(reloaded.config, key) == value


def test_the_vectorized_mask_matches_the_specification_exactly():
    """Two builders for one mask, and the Python one is the specification.

    `materialize_mask` is pure Python because the compiler is imported by the
    gateway and the drift tests without torch, and it is what the isolation
    tests above read. It is also O(n^2) in the interpreter: 208 ms per request
    at HelpSteer2's sequence lengths, against 399 ms for the entire forward
    pass.

    Banking77 hid that entirely — its requests land on a handful of distinct
    lengths, so the shape cache hits and the cost never appears. A corpus of
    free text misses every time. The backend therefore builds the mask with
    tensor indexing instead, and this asserts the two agree **exactly**,
    because the moment they do not, every isolation guarantee in this file is
    being proved about a mask the model does not use.
    """
    from trigon.schema.compiler import materialize_mask

    backend = TorchReadoutBackend(seed=0)
    compiler = backend.make_compiler()

    states = [
        "short",
        "a medium length state with a few more words in it than the first",
        "word " * 200,
        "word " * 511,
    ]
    questions = [
        {"q": BASE["urgent"]},
        BASE,
        {**BASE, "extra": NoulQuestion(instructions="Another one?")},
    ]
    for state in states:
        for question_set in questions:
            compiled = compiler.compile_request(
                SystemOneRequest(state=state, questions=question_set)
            )
            expected = torch.tensor(materialize_mask(compiled), dtype=torch.bool)
            assert torch.equal(backend._mask_tensor(compiled), expected), (
                f"the vectorized mask differs at {compiled.total_tokens} tokens"
            )


# -- evidence inherits the answer's independence -----------------------------
#
# A question's evidence may depend only on what its answer is allowed to see:
# the state's hidden states and the question's own schema and readout. Both
# sources read nothing else -- the span head is a bilinear product of exactly
# those, and gradient x input differentiates the question's own log-probability,
# which no other question's tokens reach -- so the mask that makes the answer
# independent makes the evidence independent too.
#
# **Across shapes these run in float64, and that is a finding, not a
# convenience.** In float32, adding one question moved gradient x input by
# 1.7e-06 on this model while the logits it differentiates moved 1.2e-07: a
# backward pass amplifies the forward's rounding about fourteenfold, past the
# bound the answers are held to. In float64 the same comparison reads exactly
# 0.0 on the evidence and 4e-16 on the logits. So the rounding is rounding,
# and the claim is tested where rounding cannot hide a leak: a real leak moves
# a score by orders of magnitude more than 1e-12. Comparisons at a fixed shape
# stay in float32, the served dtype, and stay exact.

EVIDENCE_METHODS = ["gradient_x_input", "span_head"]
#: Across shapes, in float64. See above for why not float32's `ROUNDING`.
EVIDENCE_ROUNDING_F64 = 1e-12


def _token_evidence(backend, compiler, questions: dict, state: str = STATE) -> dict:
    request = SystemOneRequest(state=state, questions=questions, options={"include_evidence": True})
    output = backend.infer(compiler.compile_request(request), request)
    return {qid: out.evidence for qid, out in output.outputs.items()}


def _backend(method: str, *, double: bool = False, cache: bool = False) -> TorchReadoutBackend:
    """The spike under one evidence method, its span head untrained.

    Untrained is the harder case for this claim, not the easier one: a random
    projection reads every direction of its input, so a leak into the state or
    the readout would show in it at full strength.
    """
    backend = TorchReadoutBackend(seed=0, cache_prefixes=cache)
    backend.evidence_mode = method
    if double:
        backend.model.double()
    return backend


def _unmoved(before, after, qid):
    assert_evidence_unmoved(before, after, qid, bound=EVIDENCE_ROUNDING_F64)


@pytest.fixture(scope="module", params=EVIDENCE_METHODS)
def evidence_backend(request):
    backend = _backend(request.param, double=True)
    return backend, backend.make_compiler()


def test_evidence_does_not_move_when_a_question_is_added(evidence_backend):
    backend, compiler = evidence_backend
    before = _token_evidence(backend, compiler, BASE)
    after = _token_evidence(
        backend,
        compiler,
        {
            **BASE,
            "unrelated": ChoiceQuestion(
                instructions="Something entirely different about weather.",
                options=[{"name": "rain"}, {"name": "sun"}],
            ),
        },
    )
    for qid in BASE:
        _unmoved(before[qid], after[qid], qid)


def test_evidence_does_not_move_when_every_other_question_is_removed(evidence_backend):
    backend, compiler = evidence_backend
    before = _token_evidence(backend, compiler, BASE)
    for qid in BASE:
        alone = _token_evidence(backend, compiler, {qid: BASE[qid]})
        _unmoved(before[qid], alone[qid], qid)


def test_evidence_survives_twenty_extra_questions(evidence_backend):
    backend, compiler = evidence_backend
    before = _token_evidence(backend, compiler, {"urgent": BASE["urgent"]})
    crowd = {
        f"filler_{i}": NoulQuestion(instructions=f"Is fact number {i} present?") for i in range(20)
    }
    after = _token_evidence(backend, compiler, {"urgent": BASE["urgent"], **crowd})
    _unmoved(before["urgent"], after["urgent"], "urgent")


@pytest.mark.parametrize("method", EVIDENCE_METHODS)
def test_evidence_is_exact_when_another_question_changes_at_the_same_shape(method):
    """The fixed-shape case, in float32, and exact: another question's *words*
    change and its token count does not, so every tensor has the same shape,
    the same kernels reduce in the same order, and any difference is real.
    """
    backend = _backend(method)
    compiler = backend.make_compiler()
    original = BASE["urgent"].instructions
    count = backend.tokenizer.count(original)
    candidates = [
        "Does this need a person within the hour?",
        "Does this need a human within the day?",
        "Does this want a human within the hour?",
        "Could this need a human within the hour?",
    ]
    rewrite = next((c for c in candidates if backend.tokenizer.count(c) == count), None)
    assert rewrite is not None, "no same-length rewrite; add a candidate"

    before = _token_evidence(backend, compiler, BASE)
    after = _token_evidence(
        backend, compiler, {**BASE, "urgent": NoulQuestion(instructions=rewrite)}
    )
    for qid in ("intent", "severity"):
        assert_evidence_unmoved(before[qid], after[qid], qid, bound=0.0)
    # And the question that did change is not trivially constant.
    assert before["urgent"] != after["urgent"]


@pytest.mark.parametrize("method", EVIDENCE_METHODS)
def test_evidence_agrees_with_the_schema_prefix_cache_on_and_off(method):
    """On against off is two kernel paths, so float64 and the bound; a hit
    against the miss that filled it is one path at one shape, so float32 and
    exact."""
    off = _backend(method, double=True)
    on = _backend(method, double=True, cache=True)
    compiler = off.make_compiler()
    plain, cached = _token_evidence(off, compiler, BASE), _token_evidence(on, compiler, BASE)
    for qid in BASE:
        _unmoved(plain[qid], cached[qid], qid)

    served = _backend(method, cache=True)
    miss = _token_evidence(served, compiler, BASE)
    assert served._last_cached_tokens == 0
    hit = _token_evidence(served, compiler, BASE)
    assert served._last_cached_tokens > 0, "the second request should hit the cache"
    for qid in BASE:
        assert_evidence_unmoved(miss[qid], hit[qid], qid, bound=0.0)


def test_asking_for_evidence_does_not_move_the_answer(engine):
    """Evidence builds the forward graph with autograd on, which takes
    `nn.TransformerEncoder` off its no-grad fast path: same shape, different
    kernel, so the answers agree to the float32 bound rather than exactly --
    measured at 2e-08. What must not change is anything a caller acts on."""
    plain = engine.answer(SystemOneRequest(state=STATE, questions=BASE))
    explained = engine.answer(
        SystemOneRequest(state=STATE, questions=BASE, options={"include_evidence": True})
    )
    for qid, answer in plain.answers.items():
        with_evidence = explained.answers[qid].model_dump()
        assert with_evidence.pop("evidence") is not None
        assert with_evidence.pop("evidence_method") == "gradient_x_input"
        without = answer.model_dump()
        without.pop("evidence"), without.pop("evidence_method")
        assert_answer_unmoved(without, with_evidence, qid)


def test_answers_with_evidence_do_not_move_when_a_question_is_added():
    """The same claim at the contract, spans and all, through the engine."""
    backend = _backend("gradient_x_input", double=True)
    engine = Engine(backend, compiler=backend.make_compiler())
    options = {"include_evidence": True}
    before = engine.answer(SystemOneRequest(state=STATE, questions=BASE, options=options))
    after = engine.answer(
        SystemOneRequest(
            state=STATE,
            questions={**BASE, "extra": NoulQuestion(instructions="Is the weather nice?")},
            options=options,
        )
    )
    for qid in BASE:
        assert before.answers[qid].evidence, f"{qid}: no spans to compare"
        assert_answer_unmoved(
            before.answers[qid].model_dump(), after.answers[qid].model_dump(), qid
        )


@pytest.mark.parametrize("method", EVIDENCE_METHODS)
def test_the_evidence_tests_can_see_a_leak(method):
    """The positive control. With `state_attends_to_schema` on, adding a
    question changes the state's hidden states and so every other question's
    evidence -- the leak the default layout forbids. Measured at 1e-03 to
    7e-03, nine orders of magnitude over the bound the tests above apply, so a
    pass above is the mask's doing and not a comparison too blunt to fail."""
    backend = _backend(method, double=True)
    compiler = backend.make_compiler(state_attends_to_schema=True)
    before = _token_evidence(backend, compiler, BASE)
    after = _token_evidence(
        backend, compiler, {**BASE, "extra": NoulQuestion(instructions="Is the weather nice?")}
    )
    moved = max(abs(a[2] - b[2]) for q in BASE for a, b in zip(before[q], after[q], strict=True))
    assert moved > 1e6 * EVIDENCE_ROUNDING_F64
