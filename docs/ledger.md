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
| Commits | 217 |
| Tests | 537 |
| Python files (`src`, `tests`, `scripts`) | 113 |
| Lines in `src/` | 12,633 |
| Release gates | 9 |
| Green-tier corpora in the licence audit | 9 |
| Committed use cases | 3 |
| Real corpora loadable | 7 |

**Certified on real data: Qwen2.5-1.5B on Banking77**, LoRA rank 16, lr
1e-4, 4 epochs — median accuracy 0.9009 against the spike's 0.7248, every seed
clearing every blocking gate (`reports/banking77/README.md`).

**Certified configuration (synthetic, the spike).** 8,000 cases, 8 epochs, d_model 128, 2 layers,
noise 0.2, `--option-scoring auto`. Clears every blocking gate on all four
seeds tried: ECE 0.0084–0.0247, adaptive 0.0153–0.0289, lift over the marginal
predictor +0.1614 to +0.2277. Evidence in `reports/iso/`.

**What the pretrained backbone does on the synthetic suite.** All three
questions far above their marginals on four seeds of four -- `size` included,
at +0.57–0.59, after seven failed interventions on the spike
(`reports/synthetic/README.md`).

**What the spike can and cannot do.** `plan` (copy a value from the state) is
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
- **Evidence** (2026-09-25). `options.include_evidence` returns, per answer,
  the spans of the state that drove it — character offsets, text, score — and
  `evidence_method` saying where they came from: a span head only when the
  checkpoint was trained on human rationales, gradient × input otherwise,
  `unavailable` from a backend that cannot attribute. Off by default and
  absent from the default response. Independence of evidence across questions
  is asserted for both methods, cache on and off, with a positive control
  that sees a real leak (`docs/architecture.md`, *Evidence*).
- **Integrated gradients** (2026-09-25), the next attribution after gradient ×
  input's falsifier fired on the backbone: served as `integrated_gradients`
  when a deployment asks (`TRIGON_UNSUPERVISED_EVIDENCE`), not by default.
  32 `u³`-spaced points from a zero baseline, 16 to a batched pass through the
  cached schema prefix; independence, completeness and batching-invariance
  asserted on the spike and the Qwen2 forward. `train_corpus.py --weights`
  scores it beside gradient × input on an existing checkpoint.

### Calibration
- Temperature scaling and isotonic calibration, **selected per primitive** on a
  held-out slice, applying whichever demonstrably helps or neither.
- Split conformal (LAC and APS) with a coverage gate whose floor is derived
  from the sample size.
- ECE, adaptive ECE, MCE, Brier, NLL, reliability diagrams, simulated noise
  floors; slices per domain **and** per primitive.

### Evaluation
- Four suites, one runner, release gates that exit non-zero.
- **`brier_over_marginal`, the gate for a drawn-annotator corpus** (Q20,
  2026-09-25). On such a corpus no predictor can pass the accuracy gates, so
  they are reported there as advisory. The Brier skill over the training
  marginal, at least +0.02, is the blocking term that fails a model ignoring
  its input. **The per-question and per-primitive gates now block on every
  backbone run** (Q16), which ends the "advisory until a real backbone"
  arrangement.
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
- **Plausibility of evidence** (`trigon.evals.rationale`): token F1 and IOU F1
  against human rationales, on words of the state, printed beside three
  floors — the lexical floor, a word list fitted to the training split's
  highlights, and every word. `train_corpus.py` appends it for any corpus
  with rationales and trains the evidence head on them.
- **Training on a GPU, from this sandbox.** `scripts/modal_train.py` runs one
  Modal container per seed and writes each seed's reports to a Volume before
  returning, so a `--detach`ed run outlives the VM that launched it and
  `--collect` fetches it later. It refuses a dirty tree, records the commit,
  the GPU it was given and the wall clock, and asks `train_corpus.py` for
  `--device cuda` by name so a missing GPU fails rather than falls back.
  Verified end to end on 2026-09-23.
- **A deployed model.** The certified Banking77 adapter behind the real
  gateway on Modal (`scripts/modal_serve.py`): API-key auth, calibrated,
  schema cache on, scale to zero; 81–108 ms model time warm.
- **C.1, the weights, packaged** (Q21). `releases/banking77-qwen15b-v1/` holds
  the model card and checksums, pinned to the committed calibrators and
  report by `tests/test_release.py`. The bundle, adapter included, is on the
  `trigon-runs` Volume. It is not yet a GitHub Release: this session's git
  proxy refused every push that was not to its working branch, a tag
  included.
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
- **Three more annotator-distribution corpora load, one per primitive.**
  GoEmotions as seven Nouls (Ekman groups plus neutral) over 57,877 comments,
  grouped from one row per rater; measuring_hate_speech as ten Score survey
  items over 29,488 comments, 7,912 annotators; Circa as one eight-way Choice
  over 34,268 question–answer pairs, five judgements each. Every one is held
  out by a hash of its grouping text, pinned by URL revision and SHA-256, and
  runs end to end through `scripts/train_corpus.py` on the CPU spike. Only
  measuring_hate_speech needed converting from Parquet
  (`scripts/convert_corpus.py`). The first Noul with a distribution exposed
  that the Noul loss ignored it and trained on the one drawn annotator; it
  now fits the share of annotators who said yes.
- **CC BY-SA evaluates and never trains** (owner's decision, 2026-09-25, Q17).
  Enforced on the licence string, not the tier: a share-alike corpus refuses
  `purpose="train"` whatever tier it carries, cannot be declared green, and a
  test holds every row of `docs/data.md`'s audit — BoolQ, FEVER, DBpedia-14,
  Circa — to it. `train_corpus.py` refuses such a corpus before building a
  model, since a calibrator fitted on it ships too.
- **HateXplain**, the rationale stream (2026-09-25): three-way labels,
  annotator distributions, and the tokens annotators marked as the reason.
  Licence read from both primary sources — MIT on the repository, CC BY 4.0 on
  the authors' dataset card — so green; pinned to a commit and a SHA-256.

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
| Qwen2.5 tokenizer, Python port against Rust | **Exact**: 0 of 34,520 texts differ over 10.3M tokens. Speed a wash against the forward pass: Rust 1.8× in bulk, Python 2× per warm call, 2.4× slower on unseen text |
| The training path's attention mask, per HelpSteer2 request | 352 ms in Python against 10.2 ms vectorized, bit-identical |
| Certified Banking77, re-gated at `ACCEPT_CONFIDENCE` 0.80 | Median ECE 0.0332 → 0.0209, worst 0.0489 → 0.0448; seed 0 now calibrated (0.0202), seed 1 still declined (0.0448); accuracy unchanged |
| **Banking77 on Qwen2.5-1.5B, lr 1e-4, four seeds — certified** | Every seed clears every blocking gate: accuracy 0.8502–0.9118, median 0.9009; ECE 0.0105–0.0489, median 0.0332 |
| Banking77 on Qwen2.5-1.5B, four seeds, the spike's config | **0.9004–0.9228 on three seeds**, median 0.9065; seed 2 collapsed to chance (0.0232). ECE 0.0077–0.0481 on the three |
| Soft (distribution) targets against majority-vote, same splits | Raw ECE 0.0096 vs 0.0560 median; Brier better on all four seeds (0.5985 vs 0.6039); quality questions move on 4/4 seeds vs 3/4 |
| **Qwen trained on annotator distributions, four seeds** | ECE 0.0082–0.0271 against a random annotator (seed 0 within its noise floor); lift +0.018–0.024 against the annotators' own +0.021 ceiling; `helpfulness`/`correctness` +0.011–0.019 on every seed; Brier 5.4–6.6% under the marginal |
| HelpSteer2 on Qwen2.5-1.5B at lr 1e-4, four seeds | Lift +0.0241 to +0.0288; Brier +7.9% to +8.8% over the marginal on every seed; ECE 0.0104–0.0179. No collapsed seed |
| HelpSteer2 on Qwen2.5-1.5B, four seeds | Lift +0.0059 to +0.0327, median +0.0259 against the spike's +0.0188; `complexity` to +0.096, `verbosity` to +0.049; `helpfulness`/`correctness` up to +0.015 on two seeds, flat on two; `coherence` never moves |
| The same seed on the same GPU type, twice | Not a replay: accuracy 0.9228 and 0.9252. GPU attention's backward is not deterministic, so a rerun is another draw |
| A trained Qwen adapter, served on this VM's CPU | 12 of 12 held-out Banking77 intents; ~0.5 s a request, 284 of 323 tokens from the schema cache; 89 MiB on disk |
| Qwen2 forward in-repo vs `transformers`, real 1.5B weights | Bit-identical: max \|diff\| 0.0, next-token agreement 1.0 |
| Banking77, four seeds | 0.7126–0.7404 against a 1.6% marginal; ECE 0.0177–0.0361, floor 0.0153 |
| HelpSteer2, three seeds, 1,400 cases | **Collapsed to the marginal.** Lift +0.0023 median; ECE 0.0108–0.0207, all passing |
| HelpSteer2, four seeds, 12,000 cases, 12 epochs | No change from 6: lift +0.0170 to +0.0202; the same three quality questions on their marginals; ECE 0.0187–0.0232 |
| HelpSteer2, four seeds, 12,000 cases, on a GPU | **Fails, and is no longer a collapse.** Lift +0.0189 median, +0.0155 to +0.0203; `complexity` +0.06–0.08 and `verbosity` +0.02 on every seed, `coherence`, `correctness`, `helpfulness` on their marginals; ECE 0.0082–0.0139, calibrator declined on all four |
| `size`, across 7 interventions and 22 runs | Below its own marginal on every seed; median −0.0095 |
| **`size` on Qwen2.5-1.5B, four seeds** | **+0.571 to +0.594 over its marginal on every seed**; `plan` +0.57–0.60, `at_risk` +0.13–0.14; every blocking gate passes; best validation loss 0.5556–0.6037 against a Bayes floor of 0.5585 |
| **measuring_hate_speech on Qwen2.5-1.5B, four seeds** | **Certified, four of four**: Brier skill +0.1709 to +0.1778 against +0.02; every gate passes. The severe items are the weak ones: `genocide` about +0.01 and `violence` about +0.03 lift (`reports/measuring_hate_speech/README.md`) |
| **Evidence on Qwen2.5-1.5B, HateXplain, four seeds per arm** | The supervised span head beats the rationale lexicon on both metrics on every seed: token F1 0.715–0.720 against 0.571–0.574, IOU F1 0.614–0.621 against 0.449–0.454. Gradient × input is below *every word* on both arms (0.179–0.308 against 0.434–0.437). Rationale supervision also narrows the accuracy spread from 0.115 to 0.012, median 0.6934 against 0.6625 (`reports/hatexplain/README.md`) |
| **GoEmotions on Qwen2.5-1.5B, four seeds** | **Certified, four of four**: Brier skill +0.2857 to +0.3010 against the +0.02 limit; ECE 0.0034–0.0076 at a floor p95 of 0.0029–0.0033. `joy` carries it (+0.21 lift); `fear` and `disgust` sit barely off their marginals on every seed (`reports/goemotions/README.md`) |
| **HelpSteer2 annotator distributions under `brier_over_marginal`** | **Certified, four seeds of four**: skill +0.0543 to +0.0658, median +0.0559, against +0.02. The hard-label ablation clears it too (median +0.0475), but only after its calibrator ran |
| The synthetic suite on Qwen, with the per-primitive gate blocking | **Three seeds of four.** Seed 0's Score head is at ECE 0.0551, which its pooled 0.0408 hid. The median worst primitive is 0.0303, so the configuration still certifies. Banking77 passes on all four seeds (worst 0.0448) |
| Banking77 accuracy (pilot, 2 seeds) | 0.4640 / 0.4193 against a 1.8% marginal — **it transfers** |
| **Evidence on HateXplain, the spike, two seeds** | The trained span head: token F1 0.488–0.495, IOU F1 0.330–0.331. **A word list beats it**: 0.573 / 0.452 — every word highlighted in half its training occurrences, very nearly a slur list. Gradient × input 0.30–0.32 token F1, below highlighting every word (0.434) |
| HateXplain accuracy, the spike, with and without rationale supervision | 0.5798 on both supervised seeds, 0.5664 / 0.5702 without, against a 0.408 marginal; every blocking gate passes on all four. Two seeds a side: not an effect |
| Evidence cost, the spike, one question | p50 2.68 ms plain, 3.36 ms span head, 5.16 ms gradient × input; the span head's answers bit-identical to the plain ones |
| Evidence across shapes | float32 gradient × input moves 1.7e-06 when a question is added, 14× the logits' 1.2e-07; float64 reads exactly 0.0. A real leak moves it 1e-03 |
| Qwen2.5 offsets against `tokenizers` | Offset for offset on NFC text; on text NFC changes, ours cover the whole composed character and the reference drops the combining mark |

---

## Believed, then disproved

The most useful section. Each of these was argued for before it was measured.

| Belief | What measurement said |
| --- | --- |
| The Rust gateway rewrite is worth phase-3 time | Gateway is 1.5% of the budget. Dropped. |
| p99 under load would show GIL pauses | p99/p50 *narrows* under pressure. Falsifier did not fire. |
| CLIP-style cosine would fix the dot-product head | Did nothing alone; cancels the residual's gain. |
| The reference configuration works | It decides its own outcome by seed. Led to the sweep rule. |
| Latency and throughput depend on a model's shape, not its weights, so a random encoder at 1.5B's shape stands in for the real one (`burn_in.py`) | The certified adapter is **1.45× slower** than its stand-in at batch 1 (102.9 against 71.2 ms): GQA, SwiGLU, a bf16 backbone and LoRA are different kernels from `nn.TransformerEncoderLayer`. The shape rows now sit beside a row of the real model |
| Integrated gradients would rescue unsupervised attribution where gradient × input could not | On Qwen2.5-1.5B it improves on gradient × input on seven of eight checkpoints and still misses *every word* on token F1 on all eight (0.193–0.385 against 0.434–0.437). It is also far from complete under bf16 (median error 78–895%). The unsupervised default is now `none` (`reports/hatexplain/README.md`) |
| Sortish batching would cut training time by ~2.82× on HelpSteer2 | **1.13×** (1.09–1.18×, four seeds each arm). 2.82× was the padded *attention work*, and on the spike attention is a small share of a step. Outcomes unchanged (`reports/helpsteer2/README.md`, A.5) |
| Fitting temperature on the training split is the discipline | It is the bug. Raised ECE on half the seeds. |
| A Score temperature of 0.20 is a degenerate fit | Constructed test: sharpening is correct for an underconfident head. |
| Burden of proof belongs on *declining* a calibrator | Seven constructed heads say the opposite, on six of them. |
| `size` fails because arithmetic is a non-goal | Never measured. A threshold over 13 values *is* learned. |
| The tokenizer hid the numbers, so splitting digits fixes it | `size` did not move; two certified seeds regressed. |
| Not enough capacity | 256×4 leaves it exactly where it was. |
| `size` cannot be learned by this architecture | It can. On a pretrained 1.5B backbone under the same layout, mask and heads, +0.57–0.59 on four seeds of four. The spike was the bottleneck, as the last standing explanation said. |
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
| The distribution corpora need a Parquet reader | HelpSteer2 is gzipped JSONL. Then GoEmotions' authors publish raw per-rater CSV and Circa's repository a TSV: only the Hugging Face mirrors are Parquet-only. One of four needs converting. |
| Circa is CC BY 4.0, so green | Its README says CC BY 4.0 and links the BY-SA 4.0 text as the full licence. Read as the stricter: amber, evaluation only. |
| "ECE ≤ 0.05 per corpus" is a reachable done-condition | Not on a corpus whose test split is below the 5,000-sample floor. |
| Modal is reachable from here: `api.modal.com` answers 200 | A `GET` is not the client. It speaks gRPC and behind this proxy needs `python-socks`; without it, 50 ms to "could not connect", the cause two exceptions down. |
| `scripts/modal_train.py` is ready and waiting on a token | It would have trained on the CPU. Nothing in the backend or trainer moved a tensor to a device; `--gpu` was ignored; results lived only on the VM that gets reclaimed. |
| The certified spike's Banking77 report says how it was trained | Its command line reads `--epochs 4`; its training record has 6 epochs on every seed. The header was written by a later invocation with default flags. |
| A pretrained backbone would do for HelpSteer2 what it did for Banking77 | Qwen2.5-1.5B, four seeds: median lift +0.0259 against the spike's +0.0188. It learns the surface questions better and the quality questions barely at all. |
| HelpSteer2's failures are the models' | The annotators' own ceiling: an oracle using the other annotators' ratings of the same response reaches +0.021 against a +0.05 gate; half-panels predict each other at −0.024. On Brier every run beats the marginal, the spike by ~5.5%, Qwen by up to 9.4%. |
| A naive oracle puts HelpSteer2's ceiling at +0.18 | It let the labelling annotator vote for itself. Leave one out and it is +0.021. Caught before anything was built on it. |
| HelpSteer2's spike at 12,000 cases was under-trained | 12 epochs: median lift +0.0188 against +0.0189 at 6; validation bottomed at epoch 6–11 on every seed. It is at its ceiling. |
| HelpSteer2 collapsed for want of data | Partly. 8.6× the data took lift from +0.0023 to +0.0189 and taught two questions of five; the three that judge quality did not move, and every seed kept its last epoch. |
| The Python tokenizer port is faster than Rust | Only one call at a time, where the binding's per-call overhead dominates. Batched across four cores, Rust is 1.8× ahead. |
| A launch runs the code it deployed, so its recorded commit is the code that ran | Not with one app name. Queued inputs are taken by any warm container of the app, including one from the previous deploy: four annotator seeds launched at `298d283` ran on the re-gate's `ca90134` containers and died on a corpus that commit did not have. It failed loudly only because the old code lacked something. |
| `modal run --detach` plus a Volume survives this VM | It keeps the app, not the calls. The container restarted twelve minutes into a 12-epoch sweep; Modal cancelled all four `starmap` inputs and nothing was written. Now deploy + `spawn`, and the launcher exits at once. |
| Accepting a calibrator at 95% of resamples is the right burden of proof | At four classes, yes. At 77 classes on a 500-answer check it is more power than the check has: worst-case gate error 0.0844 over five known heads, three of them failing. 0.80 keeps all five under 0.05 (0.0438) and is identical at four classes. |
| A chunk of eight fits on a 24 GB GPU | HelpSteer2's longest case is 7,171 tokens; the chunk asked a 22 GiB A10 for 6.13 GiB at once and died four minutes in. |
| Evidence can be held to the answers' float32 bound across shapes | Gradient × input moved 1.7e-06 when one question was added, past the 1e-06 bound: a backward pass amplifies the forward's rounding ~14×. In float64 it reads 0.0, so the tests compare there. |
| Asking for evidence leaves the answer bit-identical | Not under gradient × input: autograd takes `nn.TransformerEncoder` off its no-grad fast path, and the answer moves 2e-08. The span head, which needs no gradients, now stays on that path and is exact. |
| Gradient × input would be a usable unsupervised attribution, and the spike's was below *every word* only because the spike is small | On Qwen2.5-1.5B, four seeds, it is still below highlighting every word: token F1 0.287–0.308 against 0.434–0.437, IOU F1 0.211–0.234 — no better than the spike's 0.30–0.32. The span head on the same weights scores 0.715–0.720. The falsifier in `docs/decisions.md` fired; integrated gradients is built and is not the default until it is measured to clear the same bar |
| Integrated gradients' completeness failure on the backbone (median 78–895%) is bf16 rounding, and a float32 path fixes it | Half right. On tiny Qwen2s bf16 is the whole failure: median 8.8% in bf16 against 0.04% in float32, and more points do not help. On Qwen2.5-1.5B's own weights on CPU, float32 still misses by 130% and 853% (63× and 200× with a non-zero segment embedding). The path jumps by up to 2.6 nats between points 0.025 apart. Where it is smooth, autograd matches finite differences (−0.5065 against −0.5085); where it is rough, they disagree. The float32 path stays as the default, since it removes the rounding. It does not make the method complete here |
| Integrated gradients needs only enough evenly spaced steps | On a pre-norm forward the path's change is packed against the zero baseline: 32 evenly spaced points on a tiny Qwen2 summed 12–91% away from the difference they must add up to, and 64 were no better. Spaced as `u³`, 32 are within 0.04% |

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

- **The cost figure was the spike's, and the savings it printed were 15–20×
  too high.** The inherited $0.007/MTok assumed ~30k prefill tokens a second
  on an L4. Measured on Modal's L4, the 0.5M-parameter spike does 35k and
  costs $0.0063; the certified 1.5B model does 3.9k and costs **$0.0574**.
  `price.py` turned the old figure into savings of 63–96× against a
  $0.25/MTok LLM, and also multiplied that rate by post-cache tokens, taking
  the cache's saving twice. Measured and counted once, the savings are
  **4.2–4.5×** (`docs/pricing.md`).

- **Circa was cleared green on its Hugging Face card alone.** The repository
  it links to names CC BY 4.0 and gives the BY-SA 4.0 text as the licence.
  Green would have let it into a training mix and put a ShareAlike question
  on the weights; it is amber now, eval only. Caught on the second source,
  before anything trained on it.

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

- ~~`size` is unlearned.~~ **Closed: the backbone learns it** (+0.57–0.59, four
  seeds of four; `reports/synthetic/README.md`). The record below stands as
  what was true of the spike.
- **`size` was unlearned on the spike, and the investigation is closed.** Seven
  interventions, twenty-two runs, below its own marginal on every seed
  (`reports/perlevel/README.md`). What is established is narrow: *this* model
  does not learn *this* question and six attempts to fix it inside the model
  failed. What is **not** established is that the architecture cannot — every
  run shares the 128-wide two-layer backbone that has been the confound under
  every finding here. Stage 1.3 is the experiment that settles it.
- ~~A pretrained backbone learns Banking77 on three seeds of four.~~
  **Closed.** At lr 3e-4 seed 2 learned for 200 steps and collapsed to ln 77
  at the peak rate; at lr 1e-4 all four seeds certify (median 0.9009).
- ~~The calibrator declined a head at ECE 0.0481.~~ **Closed: the threshold
  was the cause, and it is 0.80 now** (`reports/calibration/decline-power.md`).
  The history, for the record: On Qwen seed 1 the
  500-case held-out check could not show the isotonic map helped a 77-way head
  beyond its noise, and the run passed the 0.05 gate by 0.002. **Then the
  same seed, rerun, flipped:** seed 0 applied the map (ECE 0.0077) in one run
  and declined it (ECE 0.0473) in another. On a 77-way head that check sits
  at the edge of its noise, so whether a model ships calibrated is close to a
  coin toss. The rule did what it says; the rule is what is open.
- ~~Preemption costs a whole seed.~~ **Closed in code.** Three Modal
  containers restarted their seeds from step 0 on the first day, up to two
  hours each. The trainer now writes a resume file every epoch and continues
  from it (`TrainingConfig.resume_path`); every Modal seed gets its own on the
  Volume. A run killed after epoch 1 and restarted in a fresh process ends
  bit-identical to an uninterrupted one on CPU (`tests/test_training.py`).
  **Exercised on Modal, 2026-09-25**: a container killed mid-epoch 3 was
  restarted by Modal in about 8 s and resumed after epoch 2/8
  (`reports/resilience/README.md`). A Modal-initiated preemption has still not
  been observed.
- ~~Whether soft targets cause the calibrated disagreement.~~ **Closed: they
  do.** Same splits, majority-vote targets: raw ECE against a random
  annotator 0.0560 median against 0.0096, two seeds of four failing the gate
  uncalibrated, and Brier worse on every seed even after the calibrator
  (`reports/helpsteer2-annotators/README.md`).
- ~~`accuracy_over_baseline` cannot certify an annotator-distribution
  corpus.~~ **Closed by decision Q20**: `brier_over_marginal` is the blocking
  term there, and HelpSteer2's annotator distributions certify on four seeds
  of four (`reports/helpsteer2-annotators/README.md`). The record:
  **`accuracy_over_baseline` cannot certify an annotator-distribution
  corpus.** On HelpSteer2 no predictor clears +0.05 -- the annotators do not
  (`reports/helpsteer2/ceiling.md`). `CLAUDE.md` requires every gate set to
  keep a term that fails a model ignoring its input; for such corpora that
  term would have to be a proper score against the marginal distribution
  (Brier or NLL), which every report now prints beside the gates. Whether to
  gate on it is a decision, not made here.
- ~~Evidence on the backbone is unmeasured.~~ **Closed, 2026-09-25**: on
  Qwen2.5-1.5B the supervised span head beats the word list on both metrics on
  four seeds of four (token F1 0.715–0.720 against 0.571–0.574). What opened in
  its place is the item below.
- ~~Integrated gradients on the backbone is unmeasured.~~ **Closed,
  2026-09-26: it does not beat every word either** (token F1 0.193–0.385
  against 0.434–0.437, eight checkpoints). Unsupervised checkpoints now serve
  no spans by default (`none`, reported as `unavailable`).
- **Integrated gradients is not complete on the backbone.** The summed
  attributions miss the log-probability difference by a median of 78–895%
  across the eight checkpoints, against under 1% in float32 on CPU. The
  backbone runs under bf16 autocast, so the measurement above is of this
  implementation on this hardware, not of the method. **The float32 path is
  built (2026-09-26) and is IG's default** (`IG_PRECISION`), and bf16 is
  measured to be a cause on CPU: eight tiny Qwen2s under bf16 autocast miss
  by a median 8.8% on their worst question (max 86%) and 256 points do not
  help, where float32 misses by 0.04%. **On the real weights it is not the
  only cause**: the path is rough, and float32 still misses by 130–853%
  (*Believed, then disproved*). Still open: the eight checkpoints rescored
  on the GPU in float32, at 32 points and at 256.
- **Faithfulness of evidence is unmeasured.** Plausibility says a person would
  agree with a highlight, not that the model used it. Comprehensiveness and
  sufficiency — delete the spans, measure the answer move — are not built.
- **$/MTok has a preliminary measurement: $0.0574, not $0.007.** Modal's L4,
  the certified model, batch 1, $0.80/h as an input
  (`reports/burn-in/modal-l4/`). It closes on a rented, dedicated L4.
- **The batched serving path does not use the schema cache.** At batch 8 and
  32 the burn-in's computed tokens equal its billed ones. Every batched row
  is slower per request than batch 1 and fails the latency target. So
  `Engine.answer_many` pays full price for the 97% of the sequence that
  batch 1 reads from the cache.
- ~~The KV cache is off by default because nobody has timed it.~~ **Closed.**
  Timed on an idle machine: 6× at the served shape, 23× at 256 options
  (`reports/cache/README.md`). It is on by default now and `/healthz` reports
  it.
- **Semantic compatibility is unmeasured.** The wire, envelope and status
  codes line up (`docs/compat.md`). The backbone now answers all three
  synthetic questions and certifies Banking77. But agreement with an
  incumbent on real traffic has never been run: `scripts/migrate.py` needs
  that traffic and an incumbent endpoint, and neither is here. An
  adapter cannot fix that, and calibration makes a wrong answer credible.
- ~~Training pads accumulation chunks to their longest member.~~ **Closed
  (A.5).** Bucketing is on by default and measured: 1.13× faster on
  HelpSteer2, with outcomes indistinguishable from the unbucketed arm. The
  2.82× it was sized by was attention work, not wall clock (see *disproved*).
- **HelpSteer2 fails at 12,000 cases, and has stopped collapsing.** Four
  seeds on a GPU, median lift +0.0189 against the +0.05 gate, spread
  +0.0155 to +0.0203. `complexity` and `verbosity` are learned on every seed;
  `coherence`, `correctness` and `helpfulness` — the three that judge quality
  rather than surface — sit on their marginals. Twelve epochs changed
  nothing (+0.0188), so this is the spike's ceiling, not under-training.
  **Qwen2.5-1.5B answers it partly**: lift +0.024 to +0.029, and Brier 7.9–8.8%
  better than the marginal. On the per-annotator split, `helpfulness` and
  `correctness` move on every seed and `coherence` moves on none. The
  aggregated labels still fail +0.05. How far anything can reach there is
  bounded only loosely: one half-panel predicts the other at −0.024
  (`reports/helpsteer2/ceiling.md`).
  `reports/helpsteer2/README.md`.
- ~~A run longer than a session's idle window cannot finish here.~~
  **Closed, 2026-09-25.** A run launched by another cloud session, which was
  then archived and its container released, was collected by id from this one:
  return code 0, reports and checkpoint intact (`reports/resilience/README.md`).
  The record of how it got here: **A run longer than a session's idle window cannot finish here.** It was
  moved to Modal, and the 12,000-case HelpSteer2 run finished there — but only
  because this container outlived it. **Reopened by the first real reclamation**: the next sweep lost all
  four seeds to it (see *disproved*). Closes again when a spawned run is
  collected after the launching container has gone.
- **CI has stopped executing.** Runs 26 and 27 failed with every job ending in
  three to five seconds, no steps recorded and logs 404 — the runner never
  reached checkout. Run 12 was green on substantially this workflow, and run
  26 predates the only workflow change since. Metered Actions minutes on a
  private organization repository is the likeliest explanation and cannot be
  confirmed without billing access. A.4 is blocked on it.
- **Two of five data streams unbuilt.** Six corpora load. The
  annotator-distribution stream has four now — HelpSteer2's `disagreements/`
  split (trained and certified), GoEmotions and measuring_hate_speech
  (loadable, green, smoke-run on the CPU spike, **never trained at size**) and
  Circa (evaluation only). Nothing on the new three is a result yet: it needs
  a backbone sweep on Modal. Synthetic workflows with teacher labels, and the
  adversarial and paired stream, remain unbuilt.
- ~~GoEmotions, measuring_hate_speech and Circa are Parquet-only.~~ **Closed.**
  Only measuring_hate_speech is; it is converted once by
  `scripts/convert_corpus.py`, and the loader stays stdlib.
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
- ~~CC BY-SA on a derived model.~~ **Closed by the owner, 2026-09-25 (Q17):**
  evaluation only, never training, enforced in `trigon.evals.corpora`.
