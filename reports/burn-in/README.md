# Burn-in: what serving costs, on Modal's L4 (preliminary)

Two runs of `scripts/burn_in.py` on Modal's serverless L4, through
`scripts/modal_burn_in.py`. Both time the certified Banking77 adapter itself
(Qwen2.5-1.5B + LoRA, seed 2, the deployed model) beside random-weight shapes,
at the Banking77 shape: 77 options, one Choice, the schema KV cache on, 2,000
requests per row. **$0.80/hour is an input**, and a serverless L4 is not the
dedicated one the cost model assumes. So B.1 stays open until a rented L4
runs the same command.

| Run | Commit | What changed | Directory |
| --- | --- | --- | --- |
| First | `cd91859` | the batched path did not read the schema cache | `modal-l4-batch-uncached/` |
| Re-time | `3b967ad` | the batched path reads it (`Engine.answer_many` → `infer_many`) | `modal-l4/` |

**The certified model:**

| Batch | p50 ms, first | p50 ms, re-time | req/s, first | req/s, re-time | $/MTok billed, re-time | p50 ≤ 150 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 102.91 | 72.31 | 9.6 | 13.5 | 0.0410 | pass |
| 8 | 487.81 | 171.45 | 16.3 | 46.4 | 0.0119 | fail (171 ms) |
| 32 | 2193.66 | 514.18 | 14.5 | 60.8 | 0.0091 | fail |

**Batching now pays.** Batch 8 serves 2.8× the requests batch 8 did before,
and batch 32 serves 4.2×. Computed tokens per second now differ from billed,
766 against 24,427 at batch 32, where before they were equal: the batch reads
the 97% of each sequence that is schema from the cache instead of recomputing
it.

**Batch 1 moved by 1.4× on unchanged code.** It reads 102.9 ms in the first
run and 72.3 ms in the re-time, and the single-request path did not change
between them. That is the host: two serverless L4 containers, on whatever
machines Modal placed them. It is the size of the error bar on any number
here, and it is why the $/MTok below is quoted as a range.

## What it costs, and what that buys against an LLM

`scripts/price.py --typed-usd-per-mtok <rate> --llm-usd-per-mtok 0.25`
bills every token sent at the burn-in's per-billed-token rate:

| Serving mode | $/MTok billed | Savings (moderation / sentiment / triage) | Latency |
| --- | ---: | --- | --- |
| Batch 1, interactive | 0.0410–0.0574 | 4.2–6.4× | p50 72–103 ms, inside the 150 ms target |
| Batch 8 | 0.0119 | 20.4–21.9× | p50 171 ms, just over the target |
| Batch 32, offline | 0.0091 | 26.7–28.7× | p50 514 ms, for queues and backfills only |

**Quote the interactive row for anything with a latency budget.** The
batched rows are real, but only where requests can wait for a batch to fill.
The inherited $0.007/MTok is still out of reach: the batch-32 rate is the
closest, and it is not an interactive price.

**The stand-in shapes remain a poor proxy.** The random-weight 1.5B shape
batches worse than the certified model (21.2 against 60.8 req/s at batch 32)
because the stand-in is an `nn.TransformerEncoder` without the Qwen path's
prefix broadcast. The certified row is the one that describes the product.
