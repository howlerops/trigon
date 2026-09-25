> **Preliminary: Modal's serverless L4, not a dedicated rental.** `docs/next.md` B.1 stays open until the same command runs on a rented L4.

# Burn-in

**NVIDIA L4** — `NVIDIA L4, 580.95.05, 23034 MiB, 72.00 W, 2040 MHz`

torch 2.14.0+cu130, CUDA 13.0, Python 3.11.12; commit `cd918597a7c2`

77 options, one Choice, schema KV cache on, 2000 requests per row after 10 warm rounds. Weights are random at each shape; latency and throughput depend on shape only. The `certified` row, when present, is the trained checkpoint itself.

Latency target: p50 ≤ 150 ms, p99 ≤ 500 ms.

| Shape | Params | Batch | p50 ms | p99 ms | req/s | billed tok/s | computed tok/s | $/MTok billed | p50 target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| certified | 1,567.0M | 1 | 102.91 | 127.17 | 9.6 | 3,869 | 121 | 0.0574 | pass |
| certified | 1,567.0M | 8 | 487.81 | 605.40 | 16.3 | 6,533 | 6,533 | 0.0340 | FAIL |
| certified | 1,567.0M | 32 | 2193.66 | 2377.43 | 14.5 | 5,822 | 5,822 | 0.0382 | FAIL |
| spike | 0.5M | 1 | 20.17 | 25.53 | 48.9 | 35,521 | 949 | 0.0063 | pass |
| spike | 0.5M | 8 | 191.20 | 237.50 | 41.4 | 30,103 | 30,103 | 0.0074 | FAIL |
| spike | 0.5M | 32 | 738.77 | 954.61 | 42.6 | 30,935 | 30,935 | 0.0072 | FAIL |
| 0.5b | 393.9M | 1 | 39.58 | 56.11 | 25.0 | 18,147 | 485 | 0.0122 | pass |
| 0.5b | 393.9M | 8 | 920.72 | 944.82 | 8.7 | 6,299 | 6,299 | 0.0353 | FAIL |
| 0.5b | 393.9M | 32 | 3659.31 | 3826.81 | 8.7 | 6,331 | 6,331 | 0.0351 | FAIL |
| 1.5b | 1,427.7M | 1 | 71.15 | 79.84 | 14.0 | 10,164 | 271 | 0.0219 | pass |
| 1.5b | 1,427.7M | 8 | 2388.55 | 2414.13 | 3.3 | 2,432 | 2,432 | 0.0914 | FAIL |
| 1.5b | 1,427.7M | 32 | 9005.16 | 9242.93 | 3.5 | 2,578 | 2,578 | 0.0862 | FAIL |

`$/MTok` uses **$0.8/hour**, an input rather than a measurement.

The spike row is the reference model's shape and is **not** the product's cost; the backbone rows are what a shipped model of that size would cost to serve.
