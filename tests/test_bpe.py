"""The byte-level BPE tokenizer.

Two properties carry everything downstream. **Exactness**: the compiler budgets
and the backend embeds from the same count, so a tokenizer that disagrees with
itself is a 500 at serve time. **Injectivity**: the hashing tokenizer it
replaces collides 24.4% of this corpus -- `is` and `report` share an id -- and
that is a ceiling under every number the eval harness reports.
"""

from __future__ import annotations

import json

import pytest

from trigon.bpe import DEFAULT_VOCAB_PATH, PAD_ID, READOUT_ID, RESERVED, SEP_ID, BPETokenizer

SAMPLES = [
    "",
    " ",
    "\n\n",
    "\t\ttabs",
    "pro",
    " pro",
    "the card payment was declined at the till",
    '{"plan": "enterprise", "open_tickets": 7, "payment_failed": true}',
    "snake_case camelCase kebab-case UPPER_SNAKE",
    "don't  they've 12,345.67",
    "café ☕ 日本語 ümlauts",
    "emoji 🎯 then more",
    "a" * 500,
    "_",
    "__x__",
]


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    return BPETokenizer.load()


def test_the_vocabulary_ships_with_the_package():
    """The gateway counts tokens on nodes with no weights, so this cannot be a
    runtime download."""
    assert DEFAULT_VOCAB_PATH.exists()
    payload = json.loads(DEFAULT_VOCAB_PATH.read_text())
    assert payload["vocab_size"] == len(payload["vocab"])
    assert payload["reserved"] == list(RESERVED)


@pytest.mark.parametrize("text", SAMPLES, ids=lambda t: repr(t[:18]))
def test_encoding_round_trips(tokenizer, text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_underscores_survive(tokenizer):
    """A regression with teeth: `_` is a word character to Python, so a
    punctuation class written as `[^\\s\\w]` matches it nowhere and `findall`
    drops it silently. `open_tickets` came out as two tokens with the
    separator gone -- text vanishing, with nothing raised."""
    for text in ("open_tickets", "limit_wire_expired_at_atm", "a_b_c", "_lead", "trail_"):
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert tokenizer.encode(text) != tokenizer.encode(text.replace("_", ""))


def test_every_byte_encodes(tokenizer):
    """Byte-level means no unknown token: there is no input a caller can send
    that the compiler would count wrongly."""
    for byte in range(256):
        text = bytes([byte]).decode("latin-1")
        assert tokenizer.encode(text), f"byte {byte} encoded to nothing"
    assert tokenizer.encode("\x00\x01\x02")


def test_leading_space_is_part_of_the_token(tokenizer):
    """Option names compile without a leading space and the same word inside
    the state has one; conflating them would leak state words into schema
    tokens."""
    assert tokenizer.encode("pro") != tokenizer.encode(" pro")


def test_counts_are_exact_and_that_is_advertised(tokenizer):
    for text in SAMPLES:
        assert tokenizer.count(text) == len(tokenizer.encode(text))
    assert tokenizer.exact is True
    assert tokenizer.kind == "bpe"


def test_encoding_is_deterministic_across_instances(tokenizer):
    other = BPETokenizer.load()
    for text in SAMPLES:
        assert tokenizer.encode(text) == other.encode(text)


def test_reserved_ids_are_fixed(tokenizer):
    """A checkpoint and a vocabulary must not disagree about which id is the
    readout slot."""
    assert (PAD_ID, READOUT_ID, SEP_ID) == (0, 1, 2)
    for expected, token in enumerate(RESERVED):
        assert tokenizer.vocab[token] == expected


def test_a_vocabulary_that_moves_a_reserved_id_is_refused():
    with pytest.raises(ValueError, match="reserved token"):
        BPETokenizer(vocab={"<pad>": 3, "a": 0}, merges=[])


def test_it_beats_the_tokenizer_it_replaces_on_collisions(tokenizer):
    """The reason this exists, asserted rather than described."""
    from trigon.backends.tokenizer import HashingTokenizer

    words = sorted({w for text in SAMPLES for w in text.split() if w.isalpha()})
    words += ["is", "report", "false", "austin", "showed", "yields"]

    hashing = HashingTokenizer(8192)
    hashed = {}
    collisions = 0
    for word in words:
        ident = tuple(hashing.encode(word))
        if ident in hashed and hashed[ident] != word:
            collisions += 1
        hashed[ident] = word

    # BPE is injective by construction: decode inverts encode, so two distinct
    # strings cannot share an encoding.
    encodings = {tuple(tokenizer.encode(w)) for w in words}
    assert len(encodings) == len(set(words))
    assert collisions >= 1, "expected the documented hash collisions in this word list"


def test_matches_the_reference_implementation():
    """The pure-Python encoder exists so the gateway needs no Rust extension.
    It is only worth having if it agrees with the library that trained the
    vocabulary, exactly -- a near-miss is a wrong token count."""
    tokenizers = pytest.importorskip("tokenizers", reason="the trainer library is a dev extra")

    payload = json.loads(DEFAULT_VOCAB_PATH.read_text())
    reference = tokenizers.Tokenizer(
        tokenizers.models.BPE(
            vocab=payload["vocab"],
            merges=[tuple(m.split(" ", 1)) for m in payload["merges"]],
        )
    )
    reference.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)

    mine = BPETokenizer.load()
    for text in SAMPLES:
        assert mine.encode(text) == reference.encode(text).ids, f"diverged on {text!r}"
