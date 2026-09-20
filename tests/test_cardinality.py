"""The recall gate on the large-cardinality prefilter -- decision D1's falsifier."""

from __future__ import annotations

import pytest

from trigon.evals.cardinality import build_cardinality_probe, run_cardinality_gate
from trigon.limits import RETRIEVAL_MIN_RECALL_AT_K
from trigon.retrieval import LexicalShortlister


def test_probe_builds_a_confusable_option_set():
    question, queries, labels = build_cardinality_probe(512, n_queries=20, seed=0)
    assert len(question.options) == 512
    assert len(set(question.names)) == 512  # no duplicates, which would fake recall
    assert len(queries) == len(labels) == 20
    assert all(0 <= label < 512 for label in labels)


def test_probe_options_share_vocabulary():
    """A set of unrelated names makes any prefilter look good. Every option
    here shares most of its words with many others."""
    question, _, _ = build_cardinality_probe(256, n_queries=1, seed=0)
    first_tokens = set(question.names[0].split("_"))
    overlapping = sum(1 for name in question.names if first_tokens & set(name.split("_")))
    assert overlapping > 64


def test_gate_reports_recall_against_the_published_limit():
    results = run_cardinality_gate(option_counts=(300,), n_queries=40, seed=0, drop_slots=0)
    assert len(results) == 1
    result = results[0]
    assert result.limit == RETRIEVAL_MIN_RECALL_AT_K
    assert 0.0 <= result.recall <= 1.0
    assert result.passed is (result.recall >= result.limit)
    assert str(result).startswith(("PASS", "FAIL"))


def test_shortlist_is_always_the_budgeted_size():
    """A short shortlist silently returns capacity the caller paid for."""
    question, queries, _ = build_cardinality_probe(300, n_queries=5, seed=1)
    shortlister = LexicalShortlister()
    for query in queries:
        assert len(shortlister.shortlist(question, query, k=64)) == 64


def test_index_is_cached_per_option_set():
    """Rebuilding the BM25 corpus per call scales with QPS rather than with the
    number of distinct schemas, which is the wrong axis entirely."""
    question, queries, _ = build_cardinality_probe(200, n_queries=3, seed=2)
    shortlister = LexicalShortlister()
    shortlister.shortlist(question, queries[0], k=32)
    assert len(shortlister._index) == 1
    shortlister.shortlist(question, queries[1], k=32)
    assert len(shortlister._index) == 1  # same option set, same entry

    other, _, _ = build_cardinality_probe(200, n_queries=1, seed=3)
    shortlister.shortlist(other, queries[2], k=32)
    assert len(shortlister._index) == 2


def test_cached_index_does_not_change_results():
    question, queries, _ = build_cardinality_probe(200, n_queries=4, seed=4)
    warm = LexicalShortlister()
    for query in queries:
        cold = LexicalShortlister()
        assert warm.shortlist(question, query, k=25) == cold.shortlist(question, query, k=25)


def test_generation_terminates_past_the_base_vocabulary():
    """The base vocabulary yields 2,744 combinations. Rejection sampling past
    that point does not terminate, and the first run at 10,000 options duly
    hung. Names are enumerated and shuffled instead."""
    from trigon.evals.cardinality import _vocabulary

    assert len(_vocabulary()) == 2744
    question, _, _ = build_cardinality_probe(10_000, n_queries=1, seed=0)
    assert len(question.options) == 10_000
    assert len(set(question.names)) == 10_000


def test_probe_rejects_a_degenerate_option_count():
    with pytest.raises(ValueError, match="at least 2 options"):
        build_cardinality_probe(1)


def test_queries_paraphrase_rather_than_quote_the_option_name():
    """If the query contained the option name verbatim, recall would measure
    string matching rather than retrieval."""
    question, queries, labels = build_cardinality_probe(300, n_queries=10, seed=5)
    for query, label in zip(queries, labels, strict=True):
        assert question.names[label] not in query


def test_gate_sweeps_difficulty_by_default():
    """A recall number without a difficulty level is not a measurement: this
    prefilter returns 1.0000 at every option count when the query names every
    field."""
    results = run_cardinality_gate(option_counts=(300,), n_queries=20, seed=0)
    assert [r.drop_slots for r in results] == [0, 1, 2]


def test_dropping_slots_makes_the_query_shorter_and_ambiguous():
    _, easy, _ = build_cardinality_probe(300, n_queries=8, seed=6, drop_slots=0)
    _, hard, _ = build_cardinality_probe(300, n_queries=8, seed=6, drop_slots=2)
    assert all(len(h) < len(e) for h, e in zip(hard, easy, strict=True))


def test_drop_slots_is_bounded_so_a_query_always_says_something():
    with pytest.raises(ValueError, match="drop_slots must be 0-3"):
        build_cardinality_probe(50, n_queries=1, drop_slots=4)
