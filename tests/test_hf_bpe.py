"""The pretrained-backbone tokenizer: what it refuses, and that it merges by rank.

Exactness against Qwen2.5's real vocabulary is measured by
`scripts/tokenizer_parity.py` over every corpus text (0 of 34,520 differ).
That needs a download; this does not, so it runs everywhere.
"""

from __future__ import annotations

import pytest

pytest.importorskip("regex")

from trigon.backends.hf_bpe import ByteLevelBPE  # noqa: E402
from trigon.bpe import _byte_encoder  # noqa: E402

QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _spec(merges: list[str]) -> dict:
    symbols = list(_byte_encoder().values())
    vocab = {s: i for i, s in enumerate(symbols)}
    for merge in merges:
        vocab.setdefault(merge.replace(" ", ""), len(vocab))
    return {
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN_SPLIT}, "behavior": "Isolated"},
                {"type": "ByteLevel", "use_regex": False},
            ],
        },
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "added_tokens": [{"id": len(vocab) + 5, "content": "<|endoftext|>"}],
    }


def test_merges_apply_in_rank_order_not_left_to_right():
    # "a b" outranks "b c", so "abc" must become [ab, c] and never [a, bc].
    tokenizer = ByteLevelBPE(_spec(["a b", "b c"]))
    assert tokenizer.encode("abc") == [tokenizer.vocab["ab"], tokenizer.vocab["c"]]
    reverse = ByteLevelBPE(_spec(["b c", "a b"]))
    assert reverse.encode("abc") == [reverse.vocab["a"], reverse.vocab["bc"]]


def test_digits_are_single_pieces_and_every_byte_encodes():
    tokenizer = ByteLevelBPE(_spec([]))
    assert tokenizer.count("127") == 3
    assert tokenizer.count("é") == 2  # two UTF-8 bytes, both in the byte alphabet
    assert tokenizer.exact


def test_vocab_size_covers_added_tokens():
    spec = _spec([])
    assert ByteLevelBPE(spec).vocab_size == spec["added_tokens"][0]["id"] + 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("normalizer", {"type": "NFKC"}),
        ("model", {"type": "WordPiece", "vocab": {}}),
    ],
)
def test_a_pipeline_it_does_not_implement_is_refused(field, value):
    spec = _spec([])
    spec[field] = value
    with pytest.raises(ValueError):
        ByteLevelBPE(spec)


# -- offsets ------------------------------------------------------------------

OFFSET_SAMPLES = [
    "Hello world, café 日本語 x",
    "  two  spaces\n\nnew",
    "Ǆ ﬁ Å",
    "emoji 🎯🎯 then",
    "don't they've 12,345",
    "",
]


def test_offsets_are_the_encoders_own_ids_and_cover_their_bytes():
    tokenizer = ByteLevelBPE(_spec(["a b", "c a", "ab c"]))
    decode = {v: k for k, v in _byte_encoder().items()}
    for text in [*OFFSET_SAMPLES, "abcab cab"]:
        triples = tokenizer.encode_with_offsets(text)
        assert [t[0] for t in triples] == tokenizer.encode(text)
        rebuilt = b""
        for token_id, start, end in triples:
            piece = bytes(decode[c] for c in tokenizer._symbols[token_id])
            assert piece in text[start:end].encode("utf-8")
            rebuilt += piece
        assert rebuilt == text.encode("utf-8")


def test_a_character_normalization_composes_maps_back_to_all_of_its_sources():
    """``e`` + combining acute is one character after NFC and two before; the
    span has to cover both, or a highlight would cut an accent off its letter."""
    tokenizer = ByteLevelBPE(_spec([]))
    text = "caf" + "é" + "!"
    triples = tokenizer.encode_with_offsets(text)
    composed = [t for t in triples if (t[1], t[2]) == (3, 5)]
    assert len(composed) == 2  # two UTF-8 bytes of the composed character
    assert triples[-1][1:] == (5, 6)


def test_offsets_match_the_reference_library_on_qwen25():
    """Offset for offset against `tokenizers`, where the vocabulary is cached.

    Skipped rather than downloaded: the parity script is where a download
    belongs. Every sample here is already NFC -- on text normalization
    changes, the reference library maps a composed character to its first
    source character only, and these offsets deliberately cover all of them.
    """
    reference = pytest.importorskip("tokenizers")
    from trigon.backends.hub import BACKBONES, cache_root

    backbone = BACKBONES["qwen2.5-1.5b"]
    path = cache_root() / backbone.repo.replace("/", "--") / backbone.revision / "tokenizer.json"
    if not path.exists():
        pytest.skip("the Qwen2.5 tokenizer is not cached here")
    ours = ByteLevelBPE.from_file(path)
    theirs = reference.Tokenizer.from_file(str(path))
    for text in OFFSET_SAMPLES:
        encoding = theirs.encode(text, add_special_tokens=False)
        triples = ours.encode_with_offsets(text)
        assert [t[0] for t in triples] == encoding.ids
        assert [(t[1], t[2]) for t in triples] == [tuple(o) for o in encoding.offsets], text
