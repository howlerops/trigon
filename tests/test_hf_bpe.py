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
