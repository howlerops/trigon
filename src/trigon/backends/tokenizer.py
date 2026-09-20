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

__all__ = ["HashingTokenizer", "PAD_ID", "READOUT_ID", "SEP_ID"]

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
