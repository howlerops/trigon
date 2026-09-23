#!/usr/bin/env python
"""A backbone's tokenizer in Python against the Rust reference: exactness and speed.

    python scripts/tokenizer_parity.py --tokenizer path/to/tokenizer.json

`docs/next.md` A.3. A pretrained backbone must see exactly the ids it was
trained on -- an encoder that differs by one token on one input hands the
model a sequence it has never seen, and nothing raises. So the port is
checked id-for-id against the reference `tokenizers` library over the text
this project actually serves: every Banking77 and HelpSteer2 row, not a
sample of convenient strings.

Speed is measured beside it because `CLAUDE.md` now picks the language on
performance and accuracy together. Both encoders are timed cold (a fresh
instance, no memo) and warm (the same texts again), because a gateway sees
the same schema text on every request and a memo is part of the real cost.

The port covers what Qwen2.5's `tokenizer.json` declares: NFC normalization,
its split regex, byte-level mapping, and BPE by merge rank. Added special
tokens are not handled -- nothing this project sends contains one -- and the
script refuses a tokenizer.json that declares a different pipeline rather
than reporting agreement on a pipeline it did not implement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import unicodedata

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from trigon.bpe import _byte_encoder  # noqa: E402


class PortedBPE:
    """Qwen2-style byte-level BPE in Python, from a `tokenizer.json`."""

    def __init__(self, spec: dict) -> None:
        import regex

        if spec["normalizer"] != {"type": "NFC"}:
            raise SystemExit(f"unsupported normalizer {spec['normalizer']}")
        steps = spec["pre_tokenizer"]["pretokenizers"]
        if [s["type"] for s in steps] != ["Split", "ByteLevel"] or steps[1]["use_regex"]:
            raise SystemExit("unsupported pre-tokenizer pipeline")
        model = spec["model"]
        if model["type"] != "BPE" or model.get("byte_fallback"):
            raise SystemExit("unsupported model")
        self._split = regex.compile(steps[0]["pattern"]["Regex"])
        self.vocab: dict[str, int] = model["vocab"]
        merges = [
            tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]
        ]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._bytes = _byte_encoder()
        self._memo: dict[str, list[int]] = {}

    def _merge(self, piece: str) -> list[int]:
        cached = self._memo.get(piece)
        if cached is not None:
            return cached
        symbols = list(piece)
        ranks = self.ranks
        while len(symbols) > 1:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                rank = ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
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
        self._memo[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        out: list[int] = []
        for piece in self._split.findall(text):
            out.extend(self._merge("".join(self._bytes[b] for b in piece.encode("utf-8"))))
        return out


def corpus_texts(limit: int) -> list[str]:
    """Every distinct string the compiler would tokenize, from both corpora."""
    from trigon.evals.corpora import load

    texts: dict[str, None] = {}
    for name in ("banking77", "helpsteer2"):
        for split, purpose in (("train", "train"), ("test", "eval")):
            for case in load(name, split, purpose=purpose):
                request = case.request
                texts[request.state] = None
                for question in request.questions.values():
                    texts[question.instructions] = None
                    for member in getattr(question, "options", None) or getattr(
                        question, "levels", []
                    ):
                        texts[member.name] = None
    ordered = list(texts)
    return ordered[:limit] if limit else ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="a tokenizer.json")
    parser.add_argument("--limit", type=int, default=0, help="texts to check; 0 = all")
    parser.add_argument("--out", default=None, help="write the result as JSON here")
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer

    spec = json.loads(pathlib.Path(args.tokenizer).read_text())
    texts = corpus_texts(args.limit)
    chars = sum(len(t) for t in texts)
    print(f"{len(texts):,} distinct texts, {chars:,} characters", flush=True)

    def timed(encode) -> tuple[list[list[int]], float]:
        started = time.perf_counter()
        out = [encode(t) for t in texts]
        return out, time.perf_counter() - started

    reference = Tokenizer.from_file(args.tokenizer)
    rust_cold, rust_cold_s = timed(lambda t: reference.encode(t, add_special_tokens=False).ids)
    _, rust_warm_s = timed(lambda t: reference.encode(t, add_special_tokens=False).ids)
    # The batch API as well. One text per call charges the binding's per-call
    # overhead to Rust, and on that alone the port looked 1.6-1.9x faster --
    # a comparison that flattered our side until this line was added.
    started = time.perf_counter()
    reference.encode_batch(texts, add_special_tokens=False)
    rust_batch_s = time.perf_counter() - started

    build_started = time.perf_counter()
    port = PortedBPE(spec)
    build_s = time.perf_counter() - build_started
    py_cold, py_cold_s = timed(port.encode)
    _, py_warm_s = timed(port.encode)

    mismatches = [i for i, (a, b) in enumerate(zip(rust_cold, py_cold, strict=True)) if a != b]
    tokens = sum(len(ids) for ids in rust_cold)
    result = {
        "texts": len(texts),
        "characters": chars,
        "tokens": tokens,
        "mismatched_texts": len(mismatches),
        "rust_cold_s": rust_cold_s,
        "rust_warm_s": rust_warm_s,
        "rust_batch_s": rust_batch_s,
        "python_build_s": build_s,
        "python_cold_s": py_cold_s,
        "python_warm_s": py_warm_s,
    }
    print(f"tokens            : {tokens:,}")
    print(f"mismatched texts  : {len(mismatches):,} of {len(texts):,}")
    for label, seconds in (
        ("rust, cold", rust_cold_s),
        ("rust, warm", rust_warm_s),
        ("rust, batch", rust_batch_s),
        ("python, cold", py_cold_s),
        ("python, warm", py_warm_s),
    ):
        print(f"{label:<18}: {seconds:8.2f} s  {tokens / seconds:>12,.0f} tok/s")
    print(f"python load       : {build_s:8.2f} s")
    for i in mismatches[:5]:
        print(f"\nMISMATCH on {texts[i][:80]!r}")
        print(f"  rust   {rust_cold[i][:20]}\n  python {py_cold[i][:20]}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
