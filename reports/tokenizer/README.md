# Qwen2.5's tokenizer: a Python port against the Rust reference

`docs/next.md` A.3, and the first thing the new tokenizer rule in `CLAUDE.md`
asks for: choose the language on measured performance and accuracy. Measured
2026-09-23 on this session's VM (4-core Xeon, no GPU) with
`scripts/tokenizer_parity.py` against the Qwen2.5-1.5B `tokenizer.json`
(151,643 tokens, 151,387 merges, NFC, byte-level BPE).

## Accuracy: exact

Every distinct string the compiler would tokenize from Banking77 and
HelpSteer2 — states, instructions, option and level names, train and test —
encoded by both and compared id for id.

| Texts | Characters | Tokens | Mismatched texts |
| ---: | ---: | ---: | ---: |
| 34,520 | 48,293,312 | 10,329,234 | **0** |

## Performance: it depends on the call, and none of it is large

| Workload | Rust `tokenizers` | Python port |
| --- | ---: | ---: |
| Bulk, whole corpus, one text per call | 402k tok/s | 637k tok/s cold, 765k warm |
| Bulk, `encode_batch` on 4 cores | **1.41M tok/s** | — (single-threaded) |
| 77 option names, warm (a Banking77 schema) | 0.82 ms | **0.37 ms** |
| Median text, 1,065 chars, warm | 0.58 ms | **0.28 ms** |
| Median text, never seen before | **0.58 ms** | 1.41 ms |
| Longest text, 10,152 chars, never seen before | **4.63 ms** | 7.39 ms |

**The first result in this table was misleading, and it flattered the port.**
One-text-per-call bulk had Python 1.6–1.9× ahead. That is the Rust binding
building offsets and masks on every call, not the encoder; given its batch
API, Rust is 1.8× ahead of Python's best. The port wins where it can memoise
— a schema's text repeats on every request — and loses on text it has not
seen, which is every request's state.

**Against the forward pass it does not matter either way.** The worst gap is
0.8 ms on a median state, against a 150 ms p50 budget and a 1.5B-parameter
prefill. In training, tokenizing the whole corpus costs 7 to 16 seconds once.

## What that decides

Neither language wins on performance by enough to matter, and accuracy is
equal. Taking the port for the serving path follows from the one difference
that remains: it adds no compiled dependency to a path that has none, except
`regex` for the Unicode classes (`\p{L}`, `\p{N}`) that Python's `re` cannot
express — a single C extension against a Rust toolchain wheel.
`tokenizers` stays in `scripts/`, where it is the reference the port is
tested against.

Not covered: Qwen's 22 added special tokens (`<|im_start|>` and the like).
Nothing this project sends contains one, and the port does not claim to
handle them.
