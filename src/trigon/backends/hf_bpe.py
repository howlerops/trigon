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

from ..bpe import _byte_encoder

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

    def count(self, text: str) -> int:
        return len(self.encode(text))

    @property
    def exact(self) -> bool:
        return True
