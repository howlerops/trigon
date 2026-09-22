# Batching across requests, measured — and it is slower here

`docs/next.md` B.2 said continuous batching over a prefill-only model is the
easy case and that "the batching layer is where the latency story is won or
lost". The first half is true. The second is wrong on this hardware, and the
measurement says so clearly enough that the default is serial.

Idle machine, one torch thread, 4-core Intel Xeon @ 2.80GHz, no GPU.
300 synthetic cases through `run_cases`.

| Batch | Seconds | Cases/s | vs serial |
| ---: | ---: | ---: | ---: |
| 1 | 1.86 | 161.4 | 1.00× |
| 2 | 2.78 | 107.8 | 0.67× |
| 4 | 2.62 | 114.7 | 0.71× |
| 8 | 2.50 | 119.8 | 0.74× |
| 16 | 2.67 | 112.2 | 0.70× |
| 32 | 2.54 | 118.3 | 0.73× |
| 64 | 2.65 | 113.1 | 0.70× |

## It is not padding

That was the first hypothesis and it is wrong. The synthetic corpus compiles
to 142–146 tokens, so the attention work a random batch of 16 wastes on
padding is **1.02×**. Padding explains none of a 30% loss.

## It is that there is nothing to fill

Breaking one request down:

| Stage | Time |
| --- | ---: |
| `_embed` | 0.73 ms |
| `materialize_mask` (cached) | 0.011 ms |
| forward, batch 1, 2-D mask | **3.43 ms** |
| forward, batch 16, per request | **4.14 ms** |
| forward, batch 16, shared 2-D mask, per request | 3.76 ms |
| whole `infer` | 5.16 ms |

Batching wins on a GPU because a single small request leaves most of the
device idle and a batch fills it. On one CPU thread there is no idle capacity:
a batched GEMM of B×T×D is B times the work of one T×D GEMM, plus the cost of
building and expanding a per-sample mask. The 3-D mask is 10% of that penalty
on its own — `nn.MultiheadAttention` repeats it once per head, so a batch of
16 at 8 heads carries a 128×146×146 boolean tensor the single-request path
never allocates.

## What was kept, and why

The implementation stays. It is correct — `tests/test_batching.py` asserts a
batched answer is identical to the one the same request would have got alone,
across mixed lengths and mixed schemas, which is the block mask's guarantee
extended across requests — and it is the right shape for the GPU this project
does not have. `batch_size > 1` opts in; everything defaults to serial.

**The throughput claim B.2 wanted is not measured and cannot be measured
here.** It needs the same hardware A.3 and B.1 need. What is measured is that
on CPU the batching layer costs 10–20%, which is worth knowing before
building a scheduler on top of it.

## A note on how this was nearly published wrong

The first two runs of this benchmark were taken while four HelpSteer2 training
jobs held all four cores, and reported 8.3 cases/s serial against 36.1 batched
— a 4.3× *speedup*, the opposite conclusion, from contention noise. That is
the same mistake already recorded in `docs/ledger.md` about the first KV-cache
benchmark. The numbers above were taken with those jobs stopped
(`kill -STOP`) and resumed afterwards.
