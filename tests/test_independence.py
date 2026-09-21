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


# float32 carries ~1.2e-07 of relative precision. Anything at or below this is
# the arithmetic; a question actually reading another question's tokens would
# move an answer by a visible fraction, not by an ulp.
ROUNDING = 1e-6


def _same(before: dict, after: dict, qid: str) -> None:
    """Two answers for one question, across sequences of different lengths."""
    assert before.keys() == after.keys(), f"{qid}: the answer's shape changed"
    for field, value in before.items():
        other = after[field]
        if isinstance(value, dict):
            assert value.keys() == other.keys(), f"{qid}.{field}: labels changed"
            for label, probability in value.items():
                assert abs(probability - other[label]) < ROUNDING, (
                    f"{qid}.{field}[{label}] moved by "
                    f"{abs(probability - other[label]):.3e}, which is not rounding"
                )
        elif isinstance(value, float):
            assert abs(value - other) < ROUNDING, (
                f"{qid}.{field} moved by {abs(value - other):.3e}, which is not rounding"
            )
        else:
            # Labels, selected options, primitive names: these are decisions,
            # and a decision that moves is a failure at any magnitude.
            assert value == other, f"{qid}.{field} changed from {value!r} to {other!r}"


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
        _same(before[qid], after[qid], qid)


def test_removing_every_other_question_moves_nothing(engine):
    before = _answers(engine, BASE)
    for qid in BASE:
        alone = _answers(engine, {qid: BASE[qid]})
        _same(before[qid], alone[qid], qid)


def test_question_map_order_does_not_matter(engine):
    before = _answers(engine, BASE)
    reordered = _answers(engine, dict(reversed(list(BASE.items()))))
    for qid in BASE:
        _same(before[qid], reordered[qid], qid)


def test_twenty_extra_questions_still_move_nothing(engine):
    """The batching claim at the scale the cookbooks use it."""
    before = _answers(engine, {"urgent": BASE["urgent"]})
    crowd = {
        f"filler_{i}": NoulQuestion(instructions=f"Is fact number {i} present?") for i in range(20)
    }
    after = _answers(engine, {"urgent": BASE["urgent"], **crowd})
    _same(before["urgent"], after["urgent"], "urgent")


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
