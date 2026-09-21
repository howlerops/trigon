"""A byte-level BPE tokenizer, in pure Python.

Pure Python because of where this runs. The gateway compiles and budgets
schemas on CPU nodes with no model weights and no training stack, and
``tests/test_openapi_drift.py``, the schema compiler and the calibration math
all import without ``torch``. A tokenizer that needed a Rust extension to count
tokens would pull that dependency into every one of those places, so encoding
lives here with no imports beyond the standard library and the vocabulary ships
as one JSON file. Training the vocabulary is a different matter -- that is
``scripts/train_tokenizer.py``, which uses the ``tokenizers`` library and runs
once.

**Why this replaces the hashing tokenizer.** ``HashingTokenizer`` maps each
word to ``blake2s(word) % vocab_size``, which on the corpus this repo actually
trains against collides **24.4%** of the vocabulary: ``is`` and ``report`` are
one token, so are ``false`` and ``austin``. That is the birthday paradox
working exactly as it must -- 2,159 distinct pieces into 8,192 buckets -- and
it is a ceiling on what any model can learn, sitting underneath every number
the eval harness reports. It is still the right choice for what it was for (a
spike that runs anywhere with no artifacts), and it stays available.

**What this is not.** It is not the backbone's tokenizer. Phase 1 swaps in
whatever the chosen base model was pretrained with, through the same
``CallableEstimator`` seam this uses. What it buys now is real subword
structure and an injective vocabulary, so a measurement is not capped by its
text encoding.

Byte-level, in the GPT-2 sense: text is encoded to UTF-8 bytes and each byte is
mapped to a printable code point before merging. That has one property worth
the indirection -- **every possible input encodes**, so there is no unknown
token and no text a caller can send that the compiler counts wrongly.
"""

from __future__ import annotations

import functools
import json
import pathlib
import re
from typing import Any

__all__ = ["BPETokenizer", "DEFAULT_VOCAB_PATH", "PAD_ID", "READOUT_ID", "SEP_ID"]

#: Reserved ids, fixed across every vocabulary so a checkpoint and a vocabulary
#: cannot disagree about which id is the readout slot.
PAD_ID = 0
READOUT_ID = 1
SEP_ID = 2
RESERVED = ("<pad>", "<readout>", "<sep>")

DEFAULT_VOCAB_PATH = pathlib.Path(__file__).resolve().parent / "data" / "bpe.json"

# GPT-2's pre-tokenizer, written for Python's ``re`` (no \p{L} available):
# contractions, then runs of letters / digits / punctuation each optionally
# preceded by one space, then whitespace. Splitting on this before merging is
# what stops a merge running across a word boundary.
# ``[^\W\d_]`` is "letters": word characters that are neither digits nor the
# underscore. The punctuation class has to put the underscore back explicitly,
# because ``_`` counts as a word character to Python and would otherwise match
# no alternative at all and be dropped -- silently, since ``findall`` skips
# what it cannot match. That cost a round of debugging: ``open_tickets`` came
# out as two tokens with the separator gone.
# **Digits are split one per piece**, and that is the single most consequential
# line in this file. BPE merges within a piece and never across one, so ` ?\d+`
# let the trainer fuse whole numbers: `127` and `128` became token 2030 and
# token 2262, two unrelated embedding rows with nothing to say about which is
# larger. A model asked for a threshold on `seats` (1-500) then had to memorise
# five hundred arbitrary id-to-tier mappings from about sixteen examples each,
# and it did not -- that question sat at chance on every seed.
#
# The corpus contains its own control. `open_tickets` ranges over thirteen
# values, `seats` over five hundred, and the same model on the same run learns
# the threshold on the first and not the second. The difficulty was never the
# comparison; it was that the tokenizer destroyed the number.
#
# `\d` -- bare, with no optional leading space -- gives every digit its own
# token, so magnitude is recoverable from position and place value is
# learnable. The space is deliberately *not* attached to the first digit: ` 1`
# and `1` would be different tokens, halving the evidence for each digit and
# making what the model learns about `1` depend on where it sat. It is also
# exactly what `tokenizers.pre_tokenizers.Digits(individual_digits=True)` does,
# and the two have to agree -- `scripts/train_tokenizer.py` trains the
# vocabulary through that library, so a pre-tokenizer here that differs from
# the one there produces a vocabulary full of merges this encoder can never
# emit. `tests/test_bpe.py` asserts the two agree token for token.
#
# See `docs/decisions.md`, "The tokenizer was hiding the numbers".
_PIECE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+",
    re.UNICODE,
)


@functools.lru_cache(maxsize=1)
def _byte_encoder() -> dict[int, str]:
    """Bytes to printable code points, so no byte is a control character.

    The 188 bytes that are already printable map to themselves; the other 68
    are lifted into an unused range. Reversible, which is what makes decoding
    exact rather than best-effort.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {b: chr(b) for b in printable}
    spare = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + spare)
            spare += 1
    return mapping


class BPETokenizer:
    """Byte-level BPE. Deterministic, injective, and exact by construction."""

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        if not vocab:
            raise ValueError("a tokenizer needs a vocabulary")
        for expected_id, token in enumerate(RESERVED):
            if vocab.get(token) != expected_id:
                raise ValueError(
                    f"reserved token {token!r} must have id {expected_id}, got {vocab.get(token)}"
                )
        self.vocab = vocab
        self.decoder = {i: t for t, i in vocab.items()}
        if len(self.decoder) != len(vocab):
            raise ValueError("vocabulary maps two tokens to one id")
        # Rank is merge order: lower merges first, which is the whole algorithm.
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.vocab_size = len(vocab)
        self._byte_encoder = _byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._cache: dict[str, tuple[str, ...]] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | pathlib.Path | None = None) -> BPETokenizer:
        """Read a vocabulary written by ``scripts/train_tokenizer.py``."""
        target = pathlib.Path(path) if path is not None else DEFAULT_VOCAB_PATH
        if not target.exists():
            raise FileNotFoundError(
                f"no tokenizer vocabulary at {target}. Run scripts/train_tokenizer.py, "
                "or use HashingTokenizer for a run that needs no artifacts."
            )
        payload: dict[str, Any] = json.loads(target.read_text())
        return cls(
            vocab=payload["vocab"],
            merges=[tuple(pair.split(" ", 1)) for pair in payload["merges"]],  # type: ignore[misc]
        )

    # -- encoding --------------------------------------------------------

    def _merge(self, piece: str) -> tuple[str, ...]:
        """Apply merges to one pre-token, most-frequent pair first."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        symbols = tuple(piece)
        while len(symbols) > 1:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            best = min(pairs, key=lambda pair: self.ranks.get(pair, len(self.ranks)))
            if best not in self.ranks:
                break
            first, second = best
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = tuple(merged)

        if len(self._cache) >= 65536:
            self._cache.clear()
        self._cache[piece] = symbols
        return symbols

    def encode(self, text: str) -> list[int]:
        """Token ids for ``text``. Never raises on content: every byte encodes."""
        ids: list[int] = []
        for piece in _PIECE.findall(text):
            mapped = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            for symbol in self._merge(mapped):
                token_id = self.vocab.get(symbol)
                if token_id is None:
                    # Unreachable with a vocabulary that contains every single
                    # byte, which the trainer guarantees -- but a hand-edited
                    # file should fail here rather than silently drop text.
                    raise KeyError(f"token {symbol!r} is not in the vocabulary")
                ids.append(token_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Inverse of ``encode`` for any ids it produced."""
        text = "".join(self.decoder[i] for i in ids if i not in {PAD_ID, READOUT_ID, SEP_ID})
        return bytearray(self._byte_decoder[c] for c in text).decode("utf-8", errors="replace")

    # -- the estimator contract ------------------------------------------

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def exact(self) -> bool:
        """True: these are the ids the backend will actually embed."""
        return True

    @property
    def kind(self) -> str:
        return "bpe"
