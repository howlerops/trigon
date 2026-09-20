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
    assert budget.needs_retrieval(10, budget.retrieval_token_trigger + 1)
    assert not budget.needs_retrieval(10, 100)


def test_budget_rejects_an_over_subscribed_context():
    from trigon.limits import Budget

    with pytest.raises(ValueError, match="over-subscribed"):
        Budget(context_tokens=100, state_tokens=80, schema_tokens=80, readout_tokens=10)
