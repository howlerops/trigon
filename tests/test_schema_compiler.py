"""Layout, cache keys, budgets and the block mask."""

from __future__ import annotations

import pytest

from trigon.limits import Budget
from trigon.schema import (
    OptionScoring,
    SchemaCompiler,
    SchemaTooLarge,
    SegmentKind,
    compile_request,
    compile_schema,
    materialize_mask,
)
from trigon.types import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion


def _request(**questions) -> DecisionRequest:
    return DecisionRequest(state="a customer message", questions=questions)


def test_schema_hash_ignores_question_map_order():
    a = ChoiceQuestion(instructions="pick", options=[{"name": "x"}, {"name": "y"}])
    b = NoulQuestion(instructions="ok?")
    assert (
        compile_schema({"a": a, "b": b}).schema_hash == compile_schema({"b": b, "a": a}).schema_hash
    )


def test_schema_hash_changes_with_content():
    a = ChoiceQuestion(instructions="pick", options=[{"name": "x"}, {"name": "y"}])
    b = ChoiceQuestion(instructions="pick", options=[{"name": "x"}, {"name": "z"}])
    assert compile_schema({"q": a}).schema_hash != compile_schema({"q": b}).schema_hash


def test_per_question_hash_is_reusable_across_requests():
    """The point of per-question hashes: the same question in a different
    request must hit the same cache entry."""
    shared = ChoiceQuestion(instructions="pick", options=[{"name": "x"}, {"name": "y"}])
    alone = compile_schema({"q": shared}).question("q")
    with_others = compile_schema({"q": shared, "other": NoulQuestion(instructions="ok?")}).question(
        "q"
    )
    assert alone.schema_hash == with_others.schema_hash


def test_layout_is_schema_then_state_then_readouts():
    compiled = compile_request(_request(q=NoulQuestion(instructions="ok?")))
    kinds = [s.kind for s in compiled.segments]
    assert kinds[0] is SegmentKind.SCHEMA_QUESTION
    assert kinds[-1] is SegmentKind.READOUT
    assert kinds.index(SegmentKind.STATE) < kinds.index(SegmentKind.READOUT)


def test_readouts_cannot_see_each_other():
    compiled = compile_request(
        _request(a=NoulQuestion(instructions="a?"), b=NoulQuestion(instructions="b?"))
    )
    plan = compiled.attention
    assert not plan.can_attend("readout:a", "readout:b")
    assert not plan.can_attend("readout:a", "schema:b")
    assert plan.can_attend("readout:a", "schema:a")
    assert plan.can_attend("readout:a", "state")


def test_state_does_not_see_the_schema_by_default():
    """This is what makes per-question independence exact -- see the compiler
    docstring."""
    compiled = compile_request(_request(a=NoulQuestion(instructions="a?")))
    assert not compiled.attention.can_attend("state", "schema:a")


def test_state_can_see_the_schema_when_the_ablation_flag_is_set():
    compiler = SchemaCompiler(state_attends_to_schema=True)
    compiled = compiler.compile_request(_request(a=NoulQuestion(instructions="a?")))
    assert compiled.attention.can_attend("state", "schema:a")


def test_materialized_mask_matches_the_group_plan():
    compiled = compile_request(
        _request(a=NoulQuestion(instructions="a?"), b=NoulQuestion(instructions="b?"))
    )
    mask = materialize_mask(compiled)
    owners = [s.group for s in compiled.segments for _ in range(s.tokens)]
    assert len(mask) == len(owners)
    for i, query in enumerate(owners):
        for j, key in enumerate(owners):
            assert mask[i][j] is compiled.attention.can_attend(query, key)


def test_causal_fallback_masks_the_future():
    compiler = SchemaCompiler(bidirectional=False)
    compiled = compiler.compile_request(_request(a=NoulQuestion(instructions="a?")))
    mask = materialize_mask(compiled)
    for i, row in enumerate(mask):
        assert not any(row[i + 1 :])


def test_option_scoring_switches_to_dot_product_past_the_crossover():
    small = compile_schema(
        {"q": ChoiceQuestion(instructions="pick", options=[{"name": str(i)} for i in range(4)])}
    ).question("q")
    large = compile_schema(
        {"q": ChoiceQuestion(instructions="pick", options=[{"name": str(i)} for i in range(200)])}
    ).question("q")
    assert small.option_scoring is OptionScoring.READOUT_PER_OPTION
    assert small.readout_slots == 4
    assert large.option_scoring is OptionScoring.DOT_PRODUCT
    assert large.readout_slots == 1


def test_oversized_schema_is_rejected_with_an_actionable_message():
    compiler = SchemaCompiler(
        budget=Budget(schema_tokens=32, state_tokens=32, readout_tokens=8, context_tokens=128)
    )
    question = ChoiceQuestion(
        instructions="pick", options=[{"name": f"option_number_{i}"} for i in range(50)]
    )
    with pytest.raises(SchemaTooLarge, match="schema budget"):
        compiler.compile_schema({"q": question})


def test_oversized_state_is_rejected():
    compiler = SchemaCompiler(budget=Budget(state_tokens=8))
    with pytest.raises(SchemaTooLarge, match="state budget"):
        compiler.compile_request(
            DecisionRequest(state="word " * 500, questions={"q": NoulQuestion(instructions="ok?")})
        )


def test_json_state_renders_deterministically():
    a = compile_request(
        DecisionRequest(state={"b": 2, "a": 1}, questions={"q": NoulQuestion(instructions="ok?")})
    )
    b = compile_request(
        DecisionRequest(state={"a": 1, "b": 2}, questions={"q": NoulQuestion(instructions="ok?")})
    )
    assert [s.text for s in a.segments] == [s.text for s in b.segments]


def test_score_always_reads_out_from_one_slot():
    compiled = compile_schema(
        {"q": ScoreQuestion(instructions="rate", levels=[{"name": str(i)} for i in range(10)])}
    ).question("q")
    assert compiled.readout_slots == 1
    assert compiled.cardinality == 10


def test_request_inside_the_total_budget_can_still_fail_the_envelope():
    """The failure mode a single flat context number hides: state plus the
    longest single question is a separate, tighter limit."""
    compiler = SchemaCompiler(
        budget=Budget(
            context_tokens=100_000,
            single_question_envelope=400,
            state_tokens=200,
            schema_tokens=90_000,
            readout_tokens=512,
        )
    )
    big = ChoiceQuestion(
        instructions="pick",
        options=[{"name": f"option_number_{i}", "criteria": "a" * 40} for i in range(60)],
    )
    request = DecisionRequest(state="a short state", questions={"q": big})
    with pytest.raises(SchemaTooLarge, match="per-question"):
        compiler.compile_request(request)


def test_a_non_choice_question_over_budget_is_rejected_not_narrowed():
    """A Score cannot be shortlisted, so an oversized one is a rejection."""
    compiler = SchemaCompiler(
        budget=Budget(
            context_tokens=100_000,
            single_question_envelope=20_000,
            state_tokens=200,
            schema_tokens=90_000,
            readout_tokens=512,
        )
    )
    verbose = ScoreQuestion(
        instructions="rate",
        levels=[{"name": f"level_{i}", "criteria": "x" * 8000} for i in range(16)],
    )
    with pytest.raises(SchemaTooLarge, match="per-question budget"):
        compiler.compile_schema({"q": verbose})


def test_attention_cost_is_block_diagonal_in_the_schema():
    """The property the capacity budgets are built on: a question's schema
    block attends only to itself, so schema cost is the SUM of per-question
    squares rather than the square of their sum."""
    few = compile_request(
        DecisionRequest(
            state="a short state",
            questions={
                "a": ChoiceQuestion(
                    instructions="pick",
                    options=[{"name": f"o{i}", "criteria": "some criterion"} for i in range(40)],
                )
            },
        )
    )
    many = compile_request(
        DecisionRequest(
            state="a short state",
            questions={
                f"q{k}": ChoiceQuestion(
                    instructions="pick",
                    options=[{"name": f"o{i}", "criteria": "some criterion"} for i in range(40)],
                )
                for k in range(16)
            },
        )
    )
    # Sixteen times the schema, far less than sixteen squared times the cost.
    assert many.total_tokens > 10 * few.total_tokens
    assert many.attention_pairs < 40 * few.attention_pairs
    assert many.attention_saving > 0.9


def test_state_is_the_quadratic_term():
    """Doubling state roughly quadruples its contribution; doubling the number
    of questions does not."""

    def cost(state_words: int, n_questions: int) -> int:
        return compile_request(
            DecisionRequest(
                state="word " * state_words,
                questions={
                    f"q{i}": NoulQuestion(instructions=f"is fact {i} present?")
                    for i in range(n_questions)
                },
            )
        ).attention_pairs

    assert cost(2000, 1) / cost(1000, 1) > 3.5
    assert cost(1000, 8) / cost(1000, 4) < 1.2


def test_dense_equivalent_states_the_claim_honestly():
    compiled = compile_request(
        DecisionRequest(
            state="word " * 500,
            questions={
                f"q{k}": ChoiceQuestion(
                    instructions="pick",
                    options=[{"name": f"o{i}", "criteria": "a criterion"} for i in range(50)],
                )
                for k in range(24)
            },
        )
    )
    # Long, but honestly priced: it costs what a much shorter dense request would.
    assert compiled.total_tokens > 8_000
    assert compiled.dense_equivalent_tokens < compiled.total_tokens // 2


def test_compat_budget_requests_are_always_valid_under_the_default():
    """Drop-in compatibility survives raising our limits, because a superset
    never rejects a request the narrower contract would have accepted."""
    from trigon.limits import COMPAT_BUDGET, DEFAULT_BUDGET

    assert DEFAULT_BUDGET.context_tokens >= COMPAT_BUDGET.context_tokens
    assert DEFAULT_BUDGET.single_question_envelope >= COMPAT_BUDGET.single_question_envelope
    assert DEFAULT_BUDGET.state_tokens >= COMPAT_BUDGET.state_tokens
    assert DEFAULT_BUDGET.max_question_tokens >= COMPAT_BUDGET.max_question_tokens
    assert DEFAULT_BUDGET.max_questions >= COMPAT_BUDGET.max_questions

    at_their_limit = DecisionRequest(
        state="word " * (COMPAT_BUDGET.state_tokens // 2),
        questions={
            f"q{i}": NoulQuestion(instructions="ok?") for i in range(COMPAT_BUDGET.max_questions)
        },
    )
    SchemaCompiler(budget=COMPAT_BUDGET).compile_request(at_their_limit)
    compile_request(at_their_limit)  # and under ours


def test_the_mask_cache_is_bounded_by_cells_not_by_entries():
    """A count-based limit does not bound a quadratic cost.

    The limit was 256 *entries*, which is sensible when a mask is the
    synthetic corpus's 70x70 -- 256 of those is about 10 MB. On HelpSteer2 the
    state is an LLM response, sequences run to 1,897 tokens, one mask is 3.6M
    cells, and 256 of them is several gigabytes. Four training processes
    filled 15 GB and the cgroup killed two of them. That is how this was
    found: not by reading the code, by losing two seeds.
    """
    from trigon.schema import compiler as module
    from trigon.schema.compiler import (
        SchemaCompiler,
        mask_cache_cells,
        materialize_mask,
    )

    module._MASK_CACHE.clear()
    module._cells_held = 0
    compiler = SchemaCompiler()

    # Many distinct state lengths, which is exactly what a corpus of free text
    # produces and what the entry-count limit failed to bound.
    for words in range(20, 420, 3):
        compiled = compiler.compile_request(
            DecisionRequest(
                state="word " * words,
                questions={"q": NoulQuestion(instructions="present?")},
            )
        )
        materialize_mask(compiled)
        assert mask_cache_cells() <= module._MASK_CACHE_CELLS, "the cache exceeded its own budget"


def test_a_mask_too_large_for_the_budget_is_returned_but_not_held():
    """One enormous request must not evict a working set it cannot join."""
    from trigon.schema import compiler as module
    from trigon.schema.compiler import SchemaCompiler, mask_cache_cells, materialize_mask

    module._MASK_CACHE.clear()
    module._cells_held = 0
    compiler = SchemaCompiler()

    small = compiler.compile_request(
        DecisionRequest(state="a b c", questions={"q": NoulQuestion(instructions="?")})
    )
    materialize_mask(small)
    held = mask_cache_cells()
    assert held > 0

    original = module._MASK_CACHE_CELLS
    module._MASK_CACHE_CELLS = held + 1  # anything bigger cannot be held
    try:
        big = compiler.compile_request(
            DecisionRequest(state="word " * 200, questions={"q": NoulQuestion(instructions="?")})
        )
        mask = materialize_mask(big)
        assert mask, "the mask must still be built and returned"
        assert mask_cache_cells() == held, "the working set was evicted by a mask that cannot fit"
    finally:
        module._MASK_CACHE_CELLS = original


def test_the_cache_still_hits_for_a_repeated_shape():
    """The saving this exists for: a gateway answering one schema at volume."""
    from trigon.schema import compiler as module
    from trigon.schema.compiler import SchemaCompiler, materialize_mask

    module._MASK_CACHE.clear()
    module._cells_held = 0
    compiler = SchemaCompiler()
    request = DecisionRequest(
        state="the same length every time", questions={"q": NoulQuestion(instructions="?")}
    )
    first = materialize_mask(compiler.compile_request(request))
    second = materialize_mask(compiler.compile_request(request))
    assert first is second, "the same shape must not be rebuilt"
