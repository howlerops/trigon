"""A hashing tokenizer, so the reference model is runnable with no downloads.

Real training swaps in the backbone's tokenizer via
``trigon.schema.CallableEstimator``. This one exists so the architecture spike
-- the block mask, the readout heads, the independence guarantees -- can be
executed and tested in CI on CPU without a checkpoint. It is a real tokenizer
in the only sense that matters here: deterministic, invertible enough for
debugging, and exact, so compiled token counts match the tensors.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

from ..bpe import DEFAULT_VOCAB_PATH, BPETokenizer

__all__ = [
    "PAD_ID",
    "READOUT_ID",
    "SEP_ID",
    "HashingTokenizer",
    "Tokenizer",
    "default_tokenizer",
]

PAD_ID = 0
READOUT_ID = 1
SEP_ID = 2
_RESERVED = 8

_PIECE = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")


class HashingTokenizer:
    """Word/character pieces hashed into a fixed vocabulary."""

    def __init__(self, vocab_size: int = 8192) -> None:
        if vocab_size <= _RESERVED:
            raise ValueError(f"vocab_size must exceed {_RESERVED} reserved ids")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [self._id(piece) for piece in _PIECE.findall(text)]

    def _id(self, piece: str) -> int:
        digest = hashlib.blake2s(piece.lower().encode("utf-8"), digest_size=4).digest()
        return _RESERVED + int.from_bytes(digest, "big") % (self.vocab_size - _RESERVED)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def exact(self) -> bool:
        return True

    @property
    def kind(self) -> str:
        return "hashing"


@runtime_checkable
class Tokenizer(Protocol):
    """What a backend needs from a tokenizer, and nothing more."""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def count(self, text: str) -> int: ...

    @property
    def exact(self) -> bool: ...

    @property
    def kind(self) -> str:
        """Which implementation this is, recorded in checkpoints."""
        ...


def default_tokenizer() -> Tokenizer:
    """The shipped BPE vocabulary, or the hashing fallback if it is missing.

    A checkout always has the vocabulary; a hand-assembled install might not,
    and the architecture spike is still runnable without it -- badly, since the
    hashing tokenizer collides 24.4% of this corpus, but runnable. Falling back
    is better than refusing to start, and a fallback nobody can detect is not:
    ``exact`` is True either way, so the distinction the caller needs is which
    class answered, which ``model_version`` and ``/healthz`` both carry.
    """
    if DEFAULT_VOCAB_PATH.exists():
        return BPETokenizer.load()
    return HashingTokenizer()


def describe(tokenizer: Tokenizer) -> dict[str, object]:
    """What a checkpoint records so it can be rebuilt with the same vocabulary."""
    return {"kind": tokenizer.kind, "vocab_size": tokenizer.vocab_size}


def build_tokenizer(spec: dict[str, object] | None) -> Tokenizer:
    """Rebuild the tokenizer a checkpoint was trained with.

    A model served under a different vocabulary than it was trained on is
    silently wrong -- every id means something else, and nothing raises. So the
    checkpoint names its tokenizer and this rebuilds exactly that one, rather
    than whatever the current default happens to be. ``None`` means a
    checkpoint written before tokenizers were recorded, which was always the
    hashing one.
    """
    if spec is None:
        return HashingTokenizer()
    kind = spec.get("kind")
    size = int(spec.get("vocab_size", 0))
    if kind == "hashing":
        return HashingTokenizer(size or 8192)
    if kind == "bpe":
        tokenizer = BPETokenizer.load()
        if size and tokenizer.vocab_size != size:
            raise ValueError(
                f"checkpoint was trained on a {size}-token BPE vocabulary but the "
                f"installed one has {tokenizer.vocab_size}. Serving it would read "
                "every token id as a different word; retrain, or install the "
                "matching trigon/data/bpe.json."
            )
        return tokenizer
    raise ValueError(f"unknown tokenizer kind {kind!r} in checkpoint")
