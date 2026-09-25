"""A pretrained backbone's byte-level BPE, from its `tokenizer.json`.

The vocabulary a backbone was pretrained with is not optional: one id off on
one input and the model reads a sequence it has never seen, and nothing
raises. This encoder is checked id for id against the reference `tokenizers`
library by `scripts/tokenizer_parity.py` -- 0 of 34,520 corpus texts differ
over 10.3M tokens (`reports/tokenizer/README.md`).

**Why Python.** `CLAUDE.md` chooses the language on measured performance and
accuracy. Accuracy is equal. Performance is a wash against a 1.5B forward
pass: Rust is 1.8x ahead in bulk, this is 2x ahead per warm call because a
schema's text repeats on every request and the memo below catches it, and
2.4x behind on text it has not seen. What decides it is that this adds only
`regex` -- for the `\\p{L}` and `\\p{N}` classes Python's `re` cannot express
-- to a path that otherwise has no compiled dependency.

It covers the pipeline Qwen2 declares -- NFC, its split regex, byte-level
mapping, BPE by merge rank -- and refuses a `tokenizer.json` declaring
anything else, rather than encoding a different pipeline and agreeing with
nothing. Added special tokens (`<|im_start|>` and the like) are not parsed out
of text: this project never sends one.
"""

from __future__ import annotations

import json
import pathlib
import unicodedata

from ..bpe import _byte_encoder, char_spans

__all__ = ["ByteLevelBPE"]


class ByteLevelBPE:
    """Exact against the reference for Qwen2-family `tokenizer.json` files."""

    kind = "hf-bpe"

    def __init__(self, spec: dict, *, source: str = "") -> None:
        import regex

        if spec.get("normalizer") != {"type": "NFC"}:
            raise ValueError(f"unsupported normalizer {spec.get('normalizer')}")
        steps = spec["pre_tokenizer"]["pretokenizers"]
        if [s["type"] for s in steps] != ["Split", "ByteLevel"] or steps[1]["use_regex"]:
            raise ValueError("unsupported pre-tokenizer pipeline")
        model = spec["model"]
        if model["type"] != "BPE" or model.get("byte_fallback"):
            raise ValueError("unsupported tokenizer model")
        self._split = regex.compile(steps[0]["pattern"]["Regex"])
        self.vocab: dict[str, int] = model["vocab"]
        merges = [
            tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]
        ]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        # The ids the model can emit or embed, added tokens included, so a
        # checkpoint can record the size its embedding table was built for.
        added = [t["id"] for t in spec.get("added_tokens", [])]
        self.vocab_size = max([*self.vocab.values(), *added]) + 1
        self.source = source
        self._bytes = _byte_encoder()
        self._memo: dict[str, list[int]] = {}

    @classmethod
    def from_file(cls, path: str | pathlib.Path, *, source: str = "") -> ByteLevelBPE:
        return cls(json.loads(pathlib.Path(path).read_text()), source=source)

    @classmethod
    def for_backbone(cls, name: str) -> ByteLevelBPE:
        """The tokenizer of a pinned backbone, fetched once and cached."""
        from .hub import BACKBONES, fetch

        return cls.from_file(fetch(BACKBONES[name], "tokenizer.json"), source=name)

    def _merge(self, piece: str) -> list[int]:
        cached = self._memo.get(piece)
        if cached is not None:
            return cached
        symbols = list(piece)
        ranks = self.ranks
        while len(symbols) > 1:
            best, best_rank = -1, None
            for i in range(len(symbols) - 1):
                rank = ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best_rank is None:
                break
            first, second = symbols[best], symbols[best + 1]
            merged, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        ids = [self.vocab[s] for s in symbols]
        if len(self._memo) >= 262_144:
            self._memo.clear()
        self._memo[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self._split.findall(unicodedata.normalize("NFC", text)):
            out.extend(self._merge("".join(self._bytes[b] for b in piece.encode("utf-8"))))
        return out

    def encode_with_offsets(self, text: str) -> list[tuple[int, int, int]]:
        """``(id, start, end)`` per token, as character offsets into ``text``.

        The ids are :meth:`encode`'s. The offsets index the caller's string,
        not the NFC-normalized one the merges run over: evidence is handed back
        as spans of what the caller sent. Where normalization is the identity
        -- which is almost all text -- the offsets are exactly what the
        reference ``tokenizers`` library reports (``tests/test_hf_bpe.py``).

        Where NFC composes characters, a normalized character is mapped back to
        the whole run of original characters it came from, so ``e`` plus a
        combining acute is one span of two characters. The reference library
        maps it to the ``e`` alone and drops the combining mark from every
        token; that is the one place these offsets deliberately differ from it.
        """
        normalized, origin = _nfc_alignment(text)
        out: list[tuple[int, int, int]] = []
        for match in self._split.finditer(normalized):
            piece = match.group()
            ids = self._merge("".join(self._bytes[b] for b in piece.encode("utf-8")))
            lengths = [len(self._symbols[i]) for i in ids]
            for token_id, (start, end) in zip(
                ids, char_spans(piece, match.start(), lengths), strict=True
            ):
                out.append((token_id, origin[start][0], origin[end - 1][1]))
        return out

    @property
    def _symbols(self) -> dict[int, str]:
        """Id to byte-level symbol, built on first use: only offsets need it."""
        table = self.__dict__.get("_symbol_table")
        if table is None:
            table = {i: s for s, i in self.vocab.items()}
            self.__dict__["_symbol_table"] = table
        return table

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def exact(self) -> bool:
        return True


def _nfc_alignment(text: str) -> tuple[str, list[tuple[int, int]]]:
    """NFC of ``text``, and for each of its characters the original span.

    NFC acts within runs that begin at a starter, so the text is cut into
    runs that normalize independently -- a cut is made before a character
    only when normalizing across it changes nothing -- and each run is mapped
    as a unit. Unchanged runs map character for character; a run NFC changed
    maps every output character to the whole run, which is exact at the only
    granularity normalization preserves.
    """
    normalized = unicodedata.normalize("NFC", text)
    if normalized == text:
        return text, [(i, i + 1) for i in range(len(text))]
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(text)):
        char = text[index]
        if unicodedata.combining(char):
            continue
        head = text[start:index]
        if unicodedata.normalize("NFC", head + char) == unicodedata.normalize(
            "NFC", head
        ) + unicodedata.normalize("NFC", char):
            runs.append((start, index))
            start = index
    runs.append((start, len(text)))

    rebuilt: list[str] = []
    origin: list[tuple[int, int]] = []
    for begin, end in runs:
        run = text[begin:end]
        composed = unicodedata.normalize("NFC", run)
        rebuilt.append(composed)
        if composed == run:
            origin.extend((begin + i, begin + i + 1) for i in range(len(run)))
        else:
            origin.extend((begin, end) for _ in composed)
    if "".join(rebuilt) != normalized:
        # Runs that do not normalize independently: fall back to one run, which
        # is still correct, only coarse.
        return normalized, [(0, len(text))] * len(normalized)
    return normalized, origin
