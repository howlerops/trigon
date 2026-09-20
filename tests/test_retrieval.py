"""The large-cardinality stage and the recall gate that guards it."""

from __future__ import annotations

import pytest

from trigon.limits import DEFAULT_BUDGET
from trigon.retrieval import LexicalShortlister, VectorShortlister, recall_at_k
from trigon.types import ChoiceQuestion


def _big_question(target_index: int = 42, n: int = 500) -> ChoiceQuestion:
    options = [{"name": f"intent_{i}"} for i in range(n)]
    options[target_index] = {
        "name": "card_payment_declined",
        "criteria": "a card transaction was refused by the bank",
    }
    return ChoiceQuestion(instructions="route", options=options)


def test_lexical_shortlister_finds_the_relevant_option():
    question = _big_question()
    keep = LexicalShortlister().shortlist(question, "my card payment was declined", k=10)
    assert 42 in keep
    assert len(keep) == 10


def test_shortlister_returns_everything_when_k_covers_the_set():
    question = ChoiceQuestion(instructions="pick", options=[{"name": "a"}, {"name": "b"}])
    assert LexicalShortlister().shortlist(question, "a", k=50) == [0, 1]


def test_vector_shortlister_uses_the_supplied_embedder():
    question = _big_question(target_index=7, n=20)

    def embed(texts):
        # One dimension per keyword: a toy embedder with obvious behaviour.
        return [[float("card" in t), float("declined" in t), float(len(t))] for t in texts]

    keep = VectorShortlister(embed).shortlist(question, "card declined", k=3)
    assert 7 in keep


def test_recall_at_k_is_the_release_gate():
    assert recall_at_k([[1, 2, 3], [4, 5]], [2, 9]) == pytest.approx(0.5)


def test_recall_at_k_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="against"):
        recall_at_k([[1]], [1, 2])


def test_retrieval_trigger_fires_on_count_and_on_tokens():
    budget = DEFAULT_BUDGET
    assert budget.needs_retrieval(budget.retrieval_option_trigger + 1, 100)
    assert budget.needs_retrieval(10, budget.max_question_tokens + 1)
    assert not budget.needs_retrieval(10, 100)


def test_max_question_tokens_is_derived_from_the_envelope():
    """Not configured: whatever the per-question envelope leaves once state has
    taken its budget, so it cannot drift from the contract it mirrors."""
    budget = DEFAULT_BUDGET
    assert budget.max_question_tokens == (budget.single_question_envelope - budget.state_tokens)


def test_envelope_rejects_a_request_that_fits_the_total_budget():
    """The failure this catches: a request comfortably inside the 64k total
    that is still inadmissible on state-plus-longest-question."""
    from trigon.limits import Budget

    budget = Budget(
        context_tokens=1000,
        single_question_envelope=100,
        state_tokens=60,
        schema_tokens=800,
        readout_tokens=64,
    )
    assert budget.fits_envelope(60, 40)
    assert not budget.fits_envelope(60, 41)


def test_budget_rejects_an_over_subscribed_context():
    from trigon.limits import Budget

    with pytest.raises(ValueError, match="over-subscribed"):
        Budget(context_tokens=100, state_tokens=80, schema_tokens=80, readout_tokens=10)
