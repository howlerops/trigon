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
| Commits | 150 |
| Tests | 495 |
| Python files (`src`, `tests`, `scripts`) | 101 |
| Lines in `src/` | 11,057 |
| Release gates | 8 |
| Green-tier corpora in the licence audit | 9 |
| Committed use cases | 3 |
| Real corpora loadable | 2 |

**Certified configuration.** 8,000 cases, 8 epochs, d_model 128, 2 layers,
noise 0.2, `--option-scoring auto`. Clears every blocking gate on all four
seeds tried: ECE 0.0084–0.0247, adaptive 0.0153–0.0289, lift over the marginal
predictor +0.1614 to +0.2277. Evidence in `reports/iso/`.

**What the model can and cannot do.** `plan` (copy a value from the state) is
learned to Bayes-optimal, +0.57 lift. `at_risk` (a conjunction plus a threshold
over 13 values) is learned on three seeds of four, +0.09. `size` (a threshold
over 500 values) is **not learned by any of seven interventions over
twenty-two runs** — the investigation is closed and the evidence is in
`reports/perlevel/README.md`.

---

## Built

### The contract and the serving path
- Typed primitives — Choice, Score, Noul — with schema violations made
  unrepresentable rather than validated. A Noul carries no confidence field by
  design.
- Schema compiler: layout, block mask, group-local positions, cache keys,
  budgets. Per-question independence and schema-prefix cacheability asserted to
  floating-point equality in `tests/test_independence.py` where the shapes
  match, and to a float32 bound where the comparison spans sequence lengths.
- `/v1/systemone` gateway, `spec/openapi.json` generated from it, drift-tested.
- Generated Python and TypeScript SDKs, both dependency-free, both exercised
  against a live gateway in CI.
- CLI: `ask`, `serve`, `spec`, `eval`, `fit`, `train`.
- **Auth, rate limiting and overload shedding** — 401, 429 and 529 with the
  headers a client acts on, on both the native and compat paths, all off
  unless an operator configures them.
- **Batching across requests** — `Engine.answer_many` coalesces forward
  passes, asserted to answer identically to the one-at-a-time path. Off by
  default: it is 30% slower on CPU.
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
- `scripts/load_test.py`, `scripts/gateway_cost.py`, `scripts/price.py`,
  `scripts/cache_bench.py`.
- `scripts/migrate.py` — point it at both endpoints with the same traffic and
  read per-question agreement, calibration on each, and where they diverge. A
  caller switches on a diff over their own traffic, not on a promise. It
  speaks the incumbent's wire, so it can actually be pointed at one.
- `scripts/train_corpus.py` — train, calibrate and gate on a real corpus, with
  the marginal predictor as the floor since a human-labelled corpus has no
  Bayes-optimal loss to quote.
- **Training on a GPU, from this sandbox.** `scripts/modal_train.py` runs one
  Modal container per seed and writes each seed's reports to a Volume before
  returning, so a `--detach`ed run outlives the VM that launched it and
  `--collect` fetches it later. It refuses a dirty tree, records the commit,
  the GPU it was given and the wall clock, and asks `train_corpus.py` for
  `--device cuda` by name so a missing GPU fails rather than falls back.
  Verified end to end on 2026-09-23.
- `scripts/burn_in.py` — B.1, written and handed over. Times the serving path
  at 0.5B and 1.5B backbone shapes rather than the spike's, because the spike
  would flatter `$/MTok` by the ratio of the models' compute.

### Data
- **Real corpora, licence-gated in code.** `trigon.evals.corpora` refuses the
  uses a corpus's tier forbids — amber evals and never trains, red ships in
  nothing — and `purpose` has no default, because a default is the argument a
  caller least often thinks about. Committed tiers are pinned against
  `docs/data.md`. Two are loadable: Banking77 (Choice, 77 intents, marginal
  ~1.3%) and HelpSteer2 (Score, five ordered ratings over one state) — the
  first real exercise of the Score primitive and of multi-question
  independence on data the generator did not write.

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
| Per-question independence, comparisons at a fixed shape | Exact, to floating-point equality |
| Per-question independence, across sequence lengths | 0.0 here, 1.4e-08 on GitHub's CPUs — not portable |
| The KV cache's cost in exactness | None that is portable: on GitHub's CPUs the *cached* path is the exact one |
| Schema share of a typical request | 77%; the cache skips 53 of 69 positions |
| Compat path vs native path | Identical answers through one process, asserted per primitive |
| Schema KV cache, 77 options | 91.8 ms → 15.3 ms p50, a **6× speedup**; 23× at 256 options |
| Batching 16 requests into one pass, on CPU | 4.14 ms/request against 3.43 ms serial — a 20% loss |
| Vectorized attention mask | 208 ms → 8.2 ms per request; the whole pass 399 ms → 82.8 ms |
| Padding waste in a training chunk, HelpSteer2 | 2.82× at chunk 8 in random order; 1.04× length-sorted |
| Length bucketing, end to end | **1.63×** — 2.7× on the attention term, diluted by everything linear |
| GPU on this machine | **Checked, absent.** `nvidia-smi` missing, `torch.cuda.is_available()` False |
| GPU through Modal | **Works.** Asked for an A10G, got a device reporting `NVIDIA A10`; 30.9 cases/s against ~1.1 on this VM's CPU |
| The training path's attention mask, per HelpSteer2 request | 352 ms in Python against 10.2 ms vectorized, bit-identical |
| Banking77, four seeds | 0.7126–0.7404 against a 1.6% marginal; ECE 0.0177–0.0361, floor 0.0153 |
| HelpSteer2, three seeds, 1,400 cases | **Collapsed to the marginal.** Lift +0.0023 median; ECE 0.0108–0.0207, all passing |
| `size`, across 7 interventions and 22 runs | Below its own marginal on every seed; median −0.0095 |
| Banking77 accuracy (pilot, 2 seeds) | 0.4640 / 0.4193 against a 1.8% marginal — **it transfers** |

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
| One readout slot carrying two bits is the bottleneck | A slot per level: median −0.0095. The last structural hypothesis, dead. |
| Stage 4.1 is blocked on the incumbent's wire format | It is published. `decisions.md` had already cited that source. |
| Stage 2.1 is blocked on data | Banking77 is reachable, green, and was cleared by our own audit. |
| The served path is exact to floating-point equality | On this machine. On GitHub's it drifts 2.4e-08 with no cache at all. |
| A configuration certified on four seeds is certified | Four seeds on *one machine*. Hardware is a second axis of the same perturbation. |
| The KV cache's saving is worth less than its exactness cost | 6× at the served shape, and the exactness cost was never the cache's. |
| Batching across requests is where the latency story is won | 30% *slower* on CPU: 3.43 ms alone against 4.14 ms in a batch of 16. |
| A 256-entry cache limit bounds a mask cache | It bounds a count. Masks are quadratic, and it cost three OOM-killed seeds. |
| The mask build is cheap next to the forward pass | It was 208 of 399 ms. Banking77's uniform lengths hid it behind a cache hit. |
| Removing 2.82× of padding waste makes training 2.82× faster | 1.63×. Attention is not the whole step; everything linear is unaffected. |
| A.3 and B.1 are blocked on hardware *(assumed)* | Checked. No GPU is present or reachable. Still blocked, now on evidence. |
| The distribution corpora need a Parquet reader | HelpSteer2 is gzipped JSONL. Three of four do; it does not. |
| "ECE ≤ 0.05 per corpus" is a reachable done-condition | Not on a corpus whose test split is below the 5,000-sample floor. |
| Modal is reachable from here: `api.modal.com` answers 200 | A `GET` is not the client. It speaks gRPC and behind this proxy needs `python-socks`; without it, 50 ms to "could not connect", the cause two exceptions down. |
| `scripts/modal_train.py` is ready and waiting on a token | It would have trained on the CPU. Nothing in the backend or trainer moved a tensor to a device; `--gpu` was ignored; results lived only on the VM that gets reclaimed. |
| A chunk of eight fits on a 24 GB GPU | HelpSteer2's longest case is 7,171 tokens; the chunk asked a 22 GiB A10 for 6.13 GiB at once and died four minutes in. |

---

## Confirmed the hard way

Rules this project wrote down before it had evidence for them, and which
measurement has since borne out. Shorter than the disproved list, and it
should stay that way — a rule that keeps being confirmed was probably cheap
to hold.

- **"Calibration never certifies alone."** Written into `CLAUDE.md` after the
  first trained model reported each question's marginal at 46% accuracy and
  passed every ECE gate. HelpSteer2 reproduced it on real data: a model whose
  accuracy is its own marginal to four decimal places, ECE 0.0108–0.0207
  against a 0.05 limit, a noise floor of 0.0055 so the number is real, and the
  calibrator correctly declining to touch a model that is calibrated by
  construction. Only `accuracy_over_baseline` rejects it.

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
- **The batching benchmark was run twice under four training jobs** and
  reported a 4.3× speedup where an idle machine shows a 30% loss. The same
  mistake as the first KV-cache benchmark, made twice in one session; the
  published numbers were taken with those jobs `kill -STOP`ped.
- **A test asserted the speedup that benchmark reported.** It required the
  summed latencies of a batched run to beat a serial one, and passed while the
  machine was busy enough for noise to cover it. It failed the first time it
  ran idle. It now asserts the accounting it always claimed to be about — the
  reported latencies sum to the run's wall clock rather than a multiple of
  it — which is true whether batching is faster or slower.
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
- **The migration harness reported 100% agreement on Score questions it never
  compared.** `_selected` read a `level` key no answer in this contract has, so
  both sides were `None` and `None == None`. The test asserted the same wrong
  key, so code and test agreed with each other and neither agreed with
  `trigon.types`. Found by running it end to end, not by reading it.
- **The migration harness could not reach the incumbent at all**, because it
  sent our body shape to a service that speaks theirs. The artifact the plan
  calls its most persuasive one answered 422 on every request.
- **"Exact, to floating-point equality" was published on one machine's
  evidence, and I said it twice.** `test_independence.py` and
  `test_prefix_cache.py` both asserted exact equality across different
  sequence lengths; both read exactly 0.0 here and 1.4e-08 / 2.4e-08 on
  GitHub's runners, because a different length selects a different GEMM
  kernel. The structural claim is untouched — the mask has no path from one
  question to another — but the numerical restatement was stronger than the
  evidence. Exactness is now asserted only where the shapes match.
- **The reason the KV cache is off by default was wrong.** It was "the cache
  turns a guarantee into a tolerance". On GitHub's runners the *cached* path
  is the exact one and the uncached path drifts, so the ordering is backwards
  there and the difference was never between the two paths at all. The cache
  stays off for a reason that survives measurement: its wall-clock saving has
  never been timed.
- **The mask vectorization was credited with rescuing HelpSteer2 training and
  never reached it.** `_mask_tensor`'s docstring tells the story — 208 ms a
  request, four seeds on course for 45 hours, fixed — and the fix went into
  `logits`, the serving path. Training calls `logits_batch`, which went on
  building every mask in Python: 352 ms a request against 10.2 ms. Found by
  timing a training step before paying for a GPU to run it, not by review.
- **The first real-corpus evaluation set was a single intent.** Banking77's
  test split is ordered by label and the loader sliced `[:eval_n]`; the
  marginal predictor scored 1.0000 and `accuracy_over_baseline` read −1.0000.
  No gate can catch that — the model really did lose to that baseline.
- **The fix for the evaluation floor reached it by leaving one training
  case.** Topping the evaluation set up from train was capped at
  `len(remaining) - 1`, so a small corpus met `sample_size` by handing almost
  everything to evaluation. Every count in the report was correct and the
  report was worthless. Capped at half the pool now, and when half is not
  enough the gate fails instead.

---

## Open

`docs/plan.md` is the execution plan for closing these: what has to be true, in
what order, and how each step is known to be done.

- **`size` is unlearned, and the investigation is closed.** Seven
  interventions, twenty-two runs, below its own marginal on every seed
  (`reports/perlevel/README.md`). What is established is narrow: *this* model
  does not learn *this* question and six attempts to fix it inside the model
  failed. What is **not** established is that the architecture cannot — every
  run shares the 128-wide two-layer backbone that has been the confound under
  every finding here. Stage 1.3 is the experiment that settles it.
- **$/MTok is unmeasured.** The whole cost argument beyond ~2× rests on it.
  Needs the L4 burn-in.
- ~~The KV cache is off by default because nobody has timed it.~~ **Closed.**
  Timed on an idle machine: 6× at the served shape, 23× at 256 options
  (`reports/cache/README.md`). It is on by default now and `/healthz` reports
  it.
- **Semantic compatibility is unmet.** The wire, envelope and status codes now
  line up (`docs/compat.md`); the model answers one question of three well. An
  adapter cannot fix that, and calibration makes a wrong answer credible.
- **Training pads accumulation chunks to their longest member**, which costs
  2.82× of the attention work on a corpus whose lengths run 253–3,647 tokens.
  Sortish batching would fix it and changes which cases share a gradient step,
  so it is `docs/next.md` A.5 rather than a quiet edit mid-certification.
- **HelpSteer2 has a result and it is a failure.** Three seeds at 1,400
  training cases: twelve of fifteen question-level accuracies land *exactly*
  on their own marginal, `accuracy_over_baseline` fails on every seed, and
  every calibration gate passes. The size was chosen to fit a cloud session's
  idle window rather than because it was enough — Banking77 needed 7,083
  cases — so this does not establish that Score cannot learn the corpus.
  `reports/helpsteer2/README.md`.
- **A run longer than a session's idle window cannot finish here** — but it
  no longer has to run here. `scripts/modal_train.py --detach` executes on
  Modal and writes to a Volume, so the VM can be reclaimed mid-run. This
  closes when the first long run is collected that way.
- **CI has stopped executing.** Runs 26 and 27 failed with every job ending in
  three to five seconds, no steps recorded and logs 404 — the runner never
  reached checkout. Run 12 was green on substantially this workflow, and run
  26 predates the only workflow change since. Metered Actions minutes on a
  private organization repository is the likeliest explanation and cannot be
  confirmed without billing access. A.4 is blocked on it.
- **Three of five data streams unbuilt.** Two corpora load. The
  *annotator-distribution* data — the stream that teaches a model what
  disagreement looks like, which is the product — is still not among them:
  HelpSteer2's main split carries aggregated integer ratings, and its
  `disagreements/` split is a separate thing to load.
- **GoEmotions, measuring_hate_speech and Circa are Parquet-only.** A reader
  for them would put a compiled dependency in the import path of the
  calibration math and the drift tests. They get converted in `scripts/`
  first, or not at all.
- **"Certified on four seeds" means four seeds on one machine.** The
  configuration failed its gates on GitHub's hardware: choice accuracy 0.530
  against 0.648–0.849 across the certified four, and choice ECE 0.1076 against
  0.0064–0.0240. The mechanism is the one `scripts/seed_sweep.py` already
  documents — "a perturbation far smaller than a seed change, the summation
  order of a batched matmul, is enough to move a given seed from one outcome
  to the other" — and the prefix-cache finding above shows that hardware *is*
  such a perturbation. Hardware is a second axis of the seed problem and it
  was never swept. The CI job publishes rather than blocks, because a single
  draw on unswept hardware is not evidence either way; sweeping four seeds
  there is `docs/next.md` A.4.
- **CC BY-SA on a derived model** — counsel opinion requested, unresolved.
