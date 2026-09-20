"""The shared tokenizer. It is a measurement instrument -- the lexical floor
and the retrieval prefilter both score with it -- so its behaviour is pinned."""

from __future__ import annotations

import pytest

from trigon.text import STOPWORDS, stem, tokenize


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("charged", "charge"),
        ("charges", "charge"),
        ("declined", "decline"),
        ("shipping", "ship"),
        ("refunds", "refund"),
        ("tracking", "track"),
        ("deliveries", "delivery"),
        ("payments", "payment"),
    ],
)
def test_inflected_forms_share_a_key(a, b):
    """Keys need not be words. They need to agree -- without this the floor
    returns a uniform distribution on tickets it should be able to route."""
    assert stem(a) == stem(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [("billing", "shipping"), ("charge", "refund"), ("account", "hardware")],
)
def test_unrelated_words_do_not_collide(a, b):
    assert stem(a) != stem(b)


def test_short_words_are_left_alone():
    """Stripping suffixes off short words collides far too much."""
    for word in ("is", "as", "ice", "use"):
        assert stem(word) == word


def test_tokenize_drops_stopwords_and_single_characters():
    assert tokenize("I was charged for the order") == ["charg", "order"]
    assert "the" in STOPWORDS


def test_tokenize_can_keep_stopwords():
    assert "the" in tokenize("the order", drop_stopwords=False)


def test_tokenize_is_case_and_punctuation_insensitive():
    assert tokenize("Charged, twice!") == tokenize("charged twice")
