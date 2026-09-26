# Drop-in compatibility

A caller with a working integration should change a base URL and a key and
keep working. This page is the honest version of that claim: what is
translated, what is translated *lossily*, and what is not there at all.

The shapes are the incumbent's published contract (its API reference and the
three primitive pages, read 2026-09-21). `src/trigon/server/compat.py` is the
translation and `tests/test_compat.py` exercises it with their own example
bodies, so the mapping is checked against the documentation rather than
against my memory of it.

## Running it

```bash
trigon serve --compat --backend torch --weights reports/run.pt   # their path at the root
trigon serve          --backend torch --weights reports/run.pt   # ours, with theirs under /compat
```

`--compat` exists because a migration changes a base URL and nothing else, and
that only works if the path is theirs exactly. The native gateway also mounts
the same router under `/compat`, so one process serves both — which is what
lets `tests/test_compat.py` assert that the two front doors return the same
numbers.

## What is translated

| Theirs | Ours | Note |
| --- | --- | --- |
| `questions[q].criteria` as a map | `options: [{name, criteria}]` | Map order is option order |
| `questions[q].criteria` as an array | `levels: [{name, criteria, value}]` | Index is the level name **and** its value |
| `answers[q].choice` | `answers[q].selected` | |
| `answers[q].noul` | `answers[q].probability` | Neither carries a confidence field |
| `answers[q].legend` | — | Rebuilt from the request's levels |
| `usage.input_tokens` | `usage.prefill_tokens` | |
| `413 schema_too_large` | | Sent as `422`: their contract has no 413 |

The level naming is not cosmetic. Their `score` is a probability-weighted mean
of the level *numbers*, so the levels must be named `"0"`, `"1"`, `"2"` and
anchored at those values, or the returned `score` is on a different scale than
the caller's and the `probabilities` keys are not the ones their SDK reads.

## What is translated lossily, and why

**A Noul's `criteria` is folded into its instructions.** Their Noul may carry
`{"true": ..., "false": ...}` boundary descriptions. Our `NoulQuestion` has
instructions and nothing else, by design — for a binary question the schema is
the question. Dropping the criteria would be silent: the answer would still be
well-formed and the caller's own words would simply never have reached the
model. So they are appended as `Counts as true: …` / `Counts as false: …`. The
text is no longer separable from the instructions; every word of it is still
in the prefix.

**Structured instructions and criteria are serialised as compact JSON.** Their
contract allows rubrics (`what` / `not_for` / `examples`), nested taxonomies
and structured instruction objects. Ours are strings. They are serialised
verbatim and in the order written — never summarised, never truncated. A rubric
that quietly lost its `not_for` clause would still produce a confident,
well-calibrated, differently-scoped answer, which is the exact failure this
project exists to avoid.

## What is not there

- **Authentication, rate limiting and overload shedding are off unless you
  configure them** — and that is the only thing in this list that is a
  *choice* rather than a gap. A self-hosted gateway should not invent a policy
  its operator did not choose, so an unset key means no `401`, an unset rate
  means no `429`, and an unset concurrency cap means no `529`. Set any of them
  and the codes and headers match the contract your client already has:

  | Variable | Emits | Headers |
  | --- | --- | --- |
  | `TRIGON_API_KEYS=k1,k2` | `401 authentication_error` | `WWW-Authenticate: Bearer` |
  | `TRIGON_RATE_PER_MINUTE=600` | `429 rate_limit_error` | `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining` |
  | `TRIGON_MAX_CONCURRENT=32` | `529 overloaded_error` | `Retry-After` |

  `/healthz` stays open whatever you set, because a liveness probe that needs
  a credential reports the credential's health.

  The limiter is a fixed-window counter, in memory, per process. That is
  stated because it decides what it is good for: it protects one process from
  one caller's burst and it does not coordinate across replicas. A deployment
  that needs a global budget puts a real limiter in front.
- **`output_tokens` is always 0.** Not a stub: there is no decode half, and the
  readout slots are prefill. Cost accounting that multiplies output tokens by a
  rate gets zero because zero is what it costs.
- **The response names the build that answered**, not the model that was asked
  for. The model name the caller asked for goes in; what comes back is this checkpoint's
  weight-fingerprinted name. A response naming a model it did not run would
  make the one identifier that reaches the caller useless for the case it
  exists for.

## What compatibility does not buy

Wire compatibility is the cheap obligation and it is now met. The response
shapes match, the envelope is a superset (`COMPAT_BUDGET` reproduces their
limits and `tests/test_compat.py` sends a request at every one of them), and
the status codes a retry loop branches on line up.

**Semantic compatibility is a separate claim and it is not met.** The reference
model answers one of its three evaluation questions at Bayes-optimal, one on
three seeds of four, and one not at all (`docs/ledger.md`). An adapter cannot
fix that, and a drop-in that returns well-formed, well-calibrated, wrong
answers is worse than no drop-in — the calibration makes the wrongness
credible. `scripts/migrate.py` exists so a caller measures that on their own
traffic instead of taking this paragraph's word for it.
