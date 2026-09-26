# What the schema KV cache saves

`docs/next.md` B.3. Measured on an idle machine with `scripts/cache_bench.py`,
one torch thread, 12 repeats over 5 states per arm, against the certified
Banking77 checkpoint (`reports/banking77/b77-seed3.pt`).

Hardware: 4-core Intel Xeon @ 2.80GHz, 15 GB, no GPU.

| Options | Tokens | Of which schema | Cache off (p50) | Cache on (p50) | Saving |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 77 | 56 | 3.46 ms | 3.19 ms | +0.27 ms (+7.8%) |
| 16 | 191 | 158 | 8.66 ms | 4.88 ms | +3.78 ms (+43.6%) |
| 77 | 725 | 707 | 91.79 ms | 15.31 ms | **+76.48 ms (+83.3%)** |
| 256 | 2,492 | 2,474 | 999.45 ms | 43.71 ms | **+955.74 ms (+95.6%)** |

At Banking77's own shape — 77 options, which is what the certified model
actually serves — the cache is a **6× speedup**. At 256 options it is 23×.

## Why it scales like that

The saving is not linear in the schema's share of the sequence, because
attention is quadratic in sequence length and the cache removes the schema
block from the *query* set. What is recomputed without it is a 2,474-token
self-attention on every request whose answer cannot change, since the schema
half encodes identically regardless of state — which is the property
`tests/test_independence.py` asserts exactly.

The tail is worse than the median on the uncached arm and roughly flat on the
cached one: at 256 options, p90 is 1,915 ms against a 999 ms p50 without the
cache, and 53 ms against 44 ms with it. A deployment sized on p99 pays for the
recomputation twice.

## The arms answer the same question

Checked before the saving is reported, and the script exits non-zero rather
than printing a number if they disagree: the two arms agree to between
1.9e-08 and 5.2e-08 across the four shapes. That is the same float32
cross-shape rounding documented in `docs/ledger.md` — it is not specific to
the cache, and on GitHub's runners the *cached* path is the exact one.

## What this changes

**The cache is now on by default.** It was off because nobody had timed it,
which was the honest reason once the earlier justification — that the cache
traded exactness for compute — turned out to be wrong on other hardware. That
reason is now spent: an optimisation with no number under it does not get to
be the default, and one with a 6× number does.

The sign of the saving is algorithmic rather than a kernel artifact: the cache
skips recomputing 97% of the sequence, and no hardware makes that slower. The
*magnitude* is this machine's and will vary.

`TRIGON_CACHE_PREFIXES=0` turns it off. `/healthz` reports which way it is
set, for the same reason it reports whether the deployment is calibrated: an
operator should not have to guess which numbers their gateway is producing.

## Batches read it too

Until 2026-09-26 only the single-request path did. `Engine.answer_many`
padded whole sequences and recomputed the schema block for every request in
the batch, so the preliminary L4 burn-in (`reports/burn-in/modal-l4/`) computed
every billed token at batch 8 and 32 where batch 1 computed 3% of them, and
every batched row was slower per request than batch 1.

`TorchReadoutBackend.infer_many` now groups a batch by `schema_hash` and runs
each group as one padded pass over the state and readout tokens only, against
one cached prefix broadcast across the group — the spike and the Qwen2 forward
alike. The prefix is looked up or filled from one request's own unpadded
tensors, the same call the single path makes, so it is bit-identical whichever
path filled it. Mixed schemas are one pass per schema. `cached_schema_tokens`
is per request: on a miss the group's first request pays for the prefix and
the rest read it, which is what one-at-a-time serving reports.
`tests/test_batch_prefix_cache.py` is the specification: batched answers
within the float32 bound of one-at-a-time ones (different shapes), exact
where the shapes match.

Timed on CPU only, and on a **shared, heavily loaded** 4-core Xeon @ 2.80GHz
(load average 15–21 from other jobs), so read the ratios rather than the
absolute numbers. `scripts/burn_in.py --device cpu --shapes spike --requests
256 --warmup 2`, two threads, the pre-change tree and this one interleaved
twice; the quieter of the two rounds:

| Batch | Before req/s | Before billed / computed tok/s | After req/s | After billed / computed tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 45.8 | 33,279 / 889 | 42.9 | 31,133 / 831 |
| 8 | 12.3 | 8,950 / **8,950** | **47.8** | 34,741 / **928** |
| 32 | 13.0 | 9,445 / **9,445** | **53.6** | 38,908 / **1,039** |

Batched throughput goes up 3.9× at batch 8 and 4.1× at batch 32, and
computed tokens now sit at 2.7% of billed at every batch size instead of
100% above batch 1. The spike is too small for batching itself to buy much
over batch 1 — per-request Python (embedding, heads) dominates a 0.5M-parameter
pass — which is why the gain is the recovery of the cache and not more. The
noisier round agrees in direction (19.1 against 10.1 req/s at batch 8).

At the Qwen2.5-0.5B shape (random weights, 32 requests a row), after the
change: 1.1, 1.4 and 1.4 req/s at batch 1, 8 and 32, computed 21–28 tok/s
against 791–1,042 billed. Before it, batch 8 managed 0.4 req/s, and batch 32
had not finished one row after 25 minutes and was stopped. What this
changes on an L4 is not measured here: the re-time is
`modal run scripts/modal_burn_in.py` from a clean tree, and until it runs
the batched rows in `reports/burn-in/modal-l4/` describe the old path.

## Reproduce

```bash
python scripts/cache_bench.py --options 77 --repeats 12 \
    --weights reports/banking77/b77-seed3.pt
```
