"""The recall gate on the large-cardinality prefilter -- decision D1's falsifier."""

from __future__ import annotations

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
    results = run_cardinality_gate(option_counts=(300,), n_queries=40, seed=0)
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
