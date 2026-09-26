> **Preliminary: Modal's serverless L4, not a dedicated rental.** `docs/next.md` B.1 stays open until the same command runs on a rented L4.

# Burn-in

**NVIDIA L4** — `NVIDIA L4, 580.95.05, 23034 MiB, 72.00 W, 2040 MHz`

torch 2.14.0+cu130, CUDA 13.0, Python 3.11.12; commit `3b967add5277`

77 options, one Choice, schema KV cache on, 2000 requests per row after 10 warm rounds. Weights are random at each shape; latency and throughput depend on shape only. The `certified` row, when present, is the trained checkpoint itself.

Latency target: p50 ≤ 150 ms, p99 ≤ 500 ms.

| Shape | Params | Batch | p50 ms | p99 ms | req/s | billed tok/s | computed tok/s | $/MTok billed | p50 target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| certified | 1,567.2M | 1 | 72.31 | 128.14 | 13.5 | 5,424 | 170 | 0.0410 | pass |
| certified | 1,567.2M | 8 | 171.45 | 193.36 | 46.4 | 18,648 | 585 | 0.0119 | FAIL |
| certified | 1,567.2M | 32 | 514.18 | 641.02 | 60.8 | 24,427 | 766 | 0.0091 | FAIL |
| spike | 0.5M | 1 | 16.31 | 19.91 | 60.8 | 44,198 | 1,180 | 0.0050 | pass |
| spike | 0.5M | 8 | 117.91 | 125.59 | 67.2 | 48,836 | 1,304 | 0.0046 | pass |
| spike | 0.5M | 32 | 459.24 | 725.86 | 68.0 | 49,422 | 1,320 | 0.0045 | FAIL |
| 0.5b | 394.1M | 1 | 33.45 | 37.76 | 29.9 | 21,697 | 579 | 0.0102 | pass |
| 0.5b | 394.1M | 8 | 198.38 | 273.19 | 39.9 | 28,981 | 774 | 0.0077 | FAIL |
| 0.5b | 394.1M | 32 | 804.49 | 925.04 | 39.4 | 28,636 | 765 | 0.0078 | FAIL |
| 1.5b | 1,427.9M | 1 | 67.26 | 72.89 | 14.8 | 10,777 | 288 | 0.0206 | pass |
| 1.5b | 1,427.9M | 8 | 398.39 | 458.87 | 19.9 | 14,476 | 387 | 0.0154 | FAIL |
| 1.5b | 1,427.9M | 32 | 1496.27 | 1633.45 | 21.2 | 15,429 | 412 | 0.0144 | FAIL |

`$/MTok` uses **$0.8/hour**, an input rather than a measurement.

The spike row is the reference model's shape and is **not** the product's cost; the backbone rows are what a shipped model of that size would cost to serve.
