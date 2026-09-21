# Ledger

What has been built, what has been measured, and what was believed and turned
out to be wrong. Updated at every milestone — see `CLAUDE.md`, *Keeping the
ledger*, for when and how.

The numbers in **State of the repository** are pinned by
`tests/test_ledger_drift.py`, so this page cannot quietly go stale. The
narrative sections are a discipline, not a test.

---

## State of the repository

| | |
| --- | ---: |
| Commits | 101 |
| Tests | 430 |
| Python files (`src`, `tests`, `scripts`) | 89 |
| Lines in `src/` | 9,730 |
| Release gates | 8 |
| Green-tier corpora in the licence audit | 9 |
| Committed use cases | 3 |

**Certified configuration.** 8,000 cases, 8 epochs, d_model 128, 2 layers,
noise 0.2, `--option-scoring auto`. Clears every blocking gate on all four
seeds tried: ECE 0.0084–0.0247, adaptive 0.0153–0.0289, lift over the marginal
predictor +0.1614 to +0.2277. Evidence in `reports/iso/`.

**What the model can and cannot do.** `plan` (copy a value from the state) is
learned to Bayes-optimal, +0.57 lift. `at_risk` (a conjunction plus a threshold
over 13 values) is learned on three seeds of four, +0.09. `size` (a threshold
over 500 values) is **not learned on any reproducible configuration** — see
*Open* below.

---

## Built

### The contract and the serving path
- Typed primitives — Choice, Score, Noul — with schema violations made
  unrepresentable rather than validated. A Noul carries no confidence field by
  design.
- Schema compiler: layout, block mask, group-local positions, cache keys,
  budgets. Per-question independence and schema-prefix cacheability asserted to
  floating-point equality in `tests/test_independence.py`.
- `/v1/systemone` gateway, `spec/openapi.json` generated from it, drift-tested.
- Generated Python and TypeScript SDKs, both dependency-free, both exercised
  against a live gateway in CI.
- CLI: `ask`, `serve`, `spec`, `eval`, `fit`, `train`.
- **Compatibility adapter** — the incumbent's published request and response
  shapes, served at their path (`trigon serve --compat`) and mounted under
  `/compat` on the native gateway. Both front doors are asserted to return the
  same numbers. Where it is lossy it says so: `docs/compat.md`.
- **Schema KV prefix cache** — per-layer pre-attention normed states plus the
  schema block's outputs, keyed on `schema_hash`. Off by default; see
  *Corrected in our own favour*.

### Calibration
- Temperature scaling and isotonic calibration, **selected per primitive** on a
  held-out slice, applying whichever demonstrably helps or neither.
- Split conformal (LAC and APS) with a coverage gate whose floor is derived
  from the sample size.
- ECE, adaptive ECE, MCE, Brier, NLL, reliability diagrams, simulated noise
  floors; slices per domain **and** per primitive.

### Evaluation
- Four suites, one runner, release gates that exit non-zero.
- `scripts/seed_sweep.py` — certify on the spread, not the best draw.
- `scripts/regate.py` — refit calibration on a saved checkpoint and re-gate in
  a minute instead of retraining for twenty.
- `scripts/decline_rule.py`, `scripts/calibrator_choice.py` — score calibration
  rules against heads whose true calibration is known, which a seed sweep
  cannot do.
- `scripts/load_test.py`, `scripts/gateway_cost.py`, `scripts/price.py`.
- `scripts/migrate.py` — point it at both endpoints with the same traffic and
  read per-question agreement, calibration on each, and where they diverge. A
  caller switches on a diff over their own traffic, not on a promise.

### Reference model
- Prefill-only transformer, byte-level BPE trained on the project's own data,
  batched training with best-epoch selection, int8 quantized twin for the
  quantization gate, weight-fingerprinted build names.

---

## Measured

| Claim | Result |
| --- | --- |
| Gateway cost per request | 2.28 ms, 1.5% of the p50 budget, 438 req/s per process |
| Gateway tail under load | p99/p50 flat at 1.5–1.8× from 1 to 32 concurrent; no GC or GIL pathology |
| Token cost vs a prompted LLM | Typed path sends *more* per request; break-even 1.6–2.9× |
| Retrieval recall at the shortlist | 1.0000 at 2,048 across every difficulty probed |
| Quantization ECE delta (int8) | 0.0002 |
| Conformal coverage | 0.9027 against a 0.90 target, clears its floor |
| Per-question independence | Exact, to floating-point equality, at 20 extra questions |
| Per-question independence *with the KV cache on* | 4.6e-08 — a tolerance, not a guarantee |
| Schema share of a typical request | 77%; the cache skips 53 of 69 positions |
| Compat path vs native path | Identical answers through one process, asserted per primitive |

---

## Believed, then disproved

The most useful section. Each of these was argued for before it was measured.

| Belief | What measurement said |
| --- | --- |
| The Rust gateway rewrite is worth phase-3 time | Gateway is 1.5% of the budget. Dropped. |
| p99 under load would show GIL pauses | p99/p50 *narrows* under pressure. Falsifier did not fire. |
| CLIP-style cosine would fix the dot-product head | Did nothing alone; cancels the residual's gain. |
| The reference configuration works | It decides its own outcome by seed. Led to the sweep rule. |
| Fitting temperature on the training split is the discipline | It is the bug. Raised ECE on half the seeds. |
| A Score temperature of 0.20 is a degenerate fit | Constructed test: sharpening is correct for an underconfident head. |
| Burden of proof belongs on *declining* a calibrator | Seven constructed heads say the opposite, on six of them. |
| `size` fails because arithmetic is a non-goal | Never measured. A threshold over 13 values *is* learned. |
| The tokenizer hid the numbers, so splitting digits fixes it | `size` did not move; two certified seeds regressed. |
| Not enough capacity | 256×4 leaves it exactly where it was. |
| The Score head was the constraint, +0.2425 proves it | Did not reproduce. The vocabulary had changed underneath. |

---

## Corrected in our own favour

Errors that flattered the project, found by re-measuring rather than by review:

- **`gateway_cost.py` charged the test client to the gateway.** httpx costs
  1.35 ms/call; a third of the published figure was the instrument.
- **The price comparison cached only our side.** The baseline's instruction
  block is equally cacheable; fixing it cut the headline by a third.
- **The price comparison used two different tokenizers** — our character
  heuristic against the baseline's real BPE. Inflated the baseline ~40%.
- **Pooled ECE was quoted as if it described the model.** It cancels: two heads
  at 0.0885 and 0.0928 in opposite directions pool to 0.0516.
- **The seed sweep reported 0 of 4 certified when 3 of 4 had passed**, by
  filtering on a flag the report format never emitted.
- **The tokenizer was trained on `docs/*.md`.** The vocabulary moved 5,635 →
  4,712 → 6,392 → 4,776 across commits, driven by *documentation edits*. It
  destroyed the one positive `size` result by changing the thing that result
  was measured against.
- **The first KV cache recomputed the block it was caching.** It stored each
  layer's keys and replayed the schema positions to recover their outputs,
  which saved nothing. Caching the outputs too is what made it a cache.
- **The KV cache's training guard was tested through `infer`**, which forces
  eval mode — so the test asserted a condition it could never reach. It passed
  for the whole time the guard was untested.
- **A Noul's isotonic map was fitted on `max(p)` and applied to `P(yes)`.** ECE
  0.3313 against 0.0251 uncalibrated. The held-out check *approved* it, because
  the check applied it the same wrong way: a selection rule that reproduces the
  bug it is meant to catch is not a check.
- **The isotonic fit did not pool tied x-values before merging**, so
  `confidence(0.1)` returned 0.0 on a head whose observed rate at 0.1 was 0.26.
- **Calibrator selection scored on plain ECE while the gates read both
  estimators.** On one Noul head the two read 0.0251 and 0.1625 over the same
  answers, so selection was optimising the blinder one and declining the
  calibrator the gate was about to fail.

---

## Open

`docs/plan.md` is the execution plan for closing these: what has to be true, in
what order, and how each step is known to be done.

- **`size` is unlearned** on every reproducible configuration. The diagnosis
  that survives: it is a Score, a Score ran the dot-product head without its
  documented repair, and it emits a near-constant answer (sd 0.019) rather than
  a noisy one. Four seeds are running on a prose-independent vocabulary.
- **$/MTok is unmeasured.** The whole cost argument beyond ~2× rests on it.
  Needs the L4 burn-in.
- **The KV cache is off by default.** Not caution: turning it on changes the
  GEMM shape, and per-question independence goes from exact to 4.6e-08. A claim
  asserted to exact equality should not quietly become a tolerance to save
  compute, so an operator opts in. Wall-clock saving is still unmeasured — the
  benchmark ran under four concurrent training jobs and is not publishable.
- **Semantic compatibility is unmet.** The wire, envelope and status codes now
  line up (`docs/compat.md`); the model answers one question of three well. An
  adapter cannot fix that, and calibration makes a wrong answer credible.
- **Four of five data streams unbuilt.** Outcome grounding rests on synthetic
  data alone.
- **CC BY-SA on a derived model** — counsel opinion requested, unresolved.
