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

## Reproduce

```bash
python scripts/cache_bench.py --options 77 --repeats 12 \
    --weights reports/banking77/b77-seed3.pt
```
