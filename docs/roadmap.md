# Roadmap

Five phases to a credible v1. The plan costed this at sixteen weeks with two
engineers; **we are not hiring, and the work is executed by one operator with
parallel agents**, so the week numbers below are an ordering with rough
proportions rather than a schedule — see *Staffing* for what actually binds.
The honest version of the destination is unchanged and worth repeating: this
yields a useful, fast, type-safe, publicly-calibrated v1 — not parity on
calibration breadth with a team that spent two years on it.

| Phase | Weeks | Work | Exit criteria |
| --- | --- | --- | --- |
| **0 — Contract + scaffolding** | 1–2 | API spec, adapter, GPU pools, eval methodology frozen, backbone selection, licence audit | spec reviewed; jaggedness suite runs against an LLM baseline |
| 1 — Architecture spike | 2–5 | schema compiler, prefix-LM conversion, block masks, categorical readouts, distillation-only fine-tune | single-pass readout ≥ prompted structured-output baseline on 3 public suites; multi-question batching shows no per-question degradation |
| 2 — Data + calibration | 4–11 | all five streams; distill → mixed → anneal; auxiliary losses; conformal fitting | ECE ≤ 0.05 held-out; injection suite beats the LLM baseline; negation coherence beats documented behaviour |
| 3 — Inference stack (parallel) | 6–13 | vLLM fork prefill path, KV-cache quantization, ANN stage, Rust gateway, load tests | ≤150 ms p50 at target QPS on L4; quantized-vs-BF16 ECE delta ≤ 0.01 |
| 4 — Evals + release | 12–16 | full three-suite run, docs, cookbooks, SDKs, weights + harness published | reproducible Pareto plots; drop-in adapter demo |

## Phase 0 status

Phase 0's exit criteria are met by this repo, and three of its assumptions did
not survive being checked.

- **API spec** — `spec/openapi.json`, generated from the reference gateway,
  with `tests/test_openapi_drift.py` failing if the two ever disagree.
- **Eval methodology frozen** — four suites, one runner, release gates that
  exit non-zero. `trigon eval all`.
- **Jaggedness suite runs against an LLM baseline** —
  `LLMBaselineBackend` wears the same interface as every other backend, so the
  suite runs against a prompted model with no special-casing.
- **Licence audit** — `docs/data.md`, with a tier per corpus and a policy that
  blocks red and amber entries from training mixes.
- **Budgets and gates** — `trigon.limits`, single-sourced.

- **The loop is closed** — `trigon train` fits the reference model on
  outcome-grounded data, calibrates it, runs the gates and writes a servable
  checkpoint, so the gates are exercised by a model rather than only defined.
- **The calibration-transfer mitigation ships** — `trigon fit --conformal-out`
  fits the wrapper the plan's biggest risk depends on, and a fitted profile is
  committed.

Not phase 0, and not here: real weights on a real backbone, the vLLM fork, the
Rust gateway, the ANN stage, the SDKs.

### What phase 0 changed

Eight findings that alter phase 1's work rather than confirming it. All are
written up in `docs/decisions.md`.

**The context envelope was wrong.** The plan assumed a flat ~32k budget. The
published contract is 64k per request *and* 32k for state plus the longest
single question. Building to one flat number halved the total and missed the
constraint that actually governs a high-cardinality Choice. `limits.py` is
rewritten and `max_question_tokens` is now derived from the envelope rather
than configured.

**Per-question independence needs more than a block mask.** Sequence-global
positions shift when a question is added. Positions are group-local, and state
does not attend to the schema. Both would have been expensive to discover after
a training run.

**The release gate was not a test.** At n=1,000 a perfectly calibrated model's
95th-percentile ECE is about 0.049 — the whole 0.05 gate. `check_gates` now
gates the measurement before the model.

**A calibration-only gate set certifies the one model guaranteed to be
useless.** Reporting each question's marginal distribution is calibrated by
construction, so no ECE gate can reject it. The dot-product arm demonstrated
this on a real run — 0.3887 accuracy against a 0.3922 marginal predictor, and
every calibration gate green, adaptive ECE included. `accuracy_over_baseline`
was added because of it, and no future gate set may drop a term that fails a
model which ignores its input. This is the finding that most changes what
"calibration is the product" is allowed to mean.

**Dot-product option scoring did not learn.** The plan scheduled the crossover
between a readout slot per option and one slot dotted with pooled option states
as a phase-1 ablation. Run early, the dot-product head finished below the
marginal predictor while the per-option head reached 0.4561 on identical data
and seeds. It is the head that makes large option sets affordable at all, so
this is phase 1's largest technical risk rather than a preference between two
working designs.

**The cardinality stress test has no licensed corpus.** UFET is unusable (no
licence; LDC-derived), and nothing permissive exists above 151 classes. The
recall gate runs on generated confusable sets instead.

**Every resolved-outcome corpus is blocked.** Home Credit, IEEE-CIS and
Autocast are all red. Outcome calibration in v1 rests on verifiable synthetic
data plus green classification labels — a narrower claim than the plan assumes,
and one the calibration report has to state.

**The trainable pool has no e-commerce domain.** Amazon ESCI's repository
licenses "the project" Apache-2.0 and says nothing about the data, which the
§3 policy reads as amber: eval only. The green Choice corpora left are all
conversational or editorial. A coverage gap for the model card, and measurable
because ESCI stays in the eval tier.

### What is still unverified

Fourteen of the sixteen figures phase 0 inherited were checked against primary
sources; the vendor's latency envelope, pricing, 255-option cap, nine
documented failure modes and absence of published calibration evidence all
hold, and the ecosystem turned out larger than assumed (28 open reproductions,
17 independent evaluations, against the plan's 17 and 16). Two did not resolve, and both were closed at sign-off rather than carried:

- **"96% on a 50-case validation task vs 84–86% for small LLMs."** No primary
  source surfaced, and the claim is **dropped** — not softened, not
  attributed, not repeated in a deck. An unsourced accuracy figure is the
  exact species of claim this project exists to replace with a measured one.
- **The L4 unit economics** — $0.80/hr, ~30k prefill tok/s, therefore
  ~$0.007/MTok. The arithmetic checks out; the inputs are unsourced. This is
  the entire cost argument, so the **burn-in is pulled forward out of phase 3**:
  at week 13 the number arrives after the point where it would shape
  positioning. Until it is measured, the cost claim is an assumption and is
  labelled one wherever it appears.

## Staffing

**One operator, plus parallel agents. No hires.** Decided 2026-09-20, and it
replaces the plan's two-engineer and solo-engineer variants rather than
selecting between them — neither describes this.

The schedule in the phase table is in weeks of *calendar*, and it was derived
from weeks of engineer-time. That derivation no longer holds, so read the
phase table as an ordering with rough proportions, not as dates. What binds
instead:

| Old constraint | Now |
| --- | --- |
| Engineer-hours writing code | Not binding. Agents parallelise cleanly across independent work — three ablation arms ran concurrently at full speed on a 4-core box, because each training run is single-threaded |
| — | **Compute.** The one resource agents cannot supply themselves, and the phase-2/3 line items ($20–50k teacher labels, $30–70k total) are unchanged |
| — | **Verification.** Agents produce plausible code quickly; the eval harness and the gates are the only thing that separates plausible from correct |
| — | **Judgement and sign-off.** Licence calls, positioning, what a number is allowed to claim. Unparallelisable by construction |

Two consequences worth stating rather than discovering:

**The eval harness moves from differentiator to load-bearing infrastructure.**
It was already the product's public claim. Under agent execution it is also
the only mechanism that catches work that looks finished and is not — this
repo's own history has two cases where the whole suite was green over a
component that could not run at all. Every ground rule in `CLAUDE.md` about
testing claims rather than asserting them gets stricter, not more relaxed,
and the per-question gate above exists because a pooled one was too easy to
satisfy.

**Parallelism buys breadth, not depth.** Three arms of an ablation at once,
yes. A single training run does not go faster, a GPU-bound phase-2 sweep does
not go faster, and the two measurements phase 3 owes — KV memory at 65,536
state tokens, end-to-end p50/p99 — need hardware, not agents. Schedule the
serial, hardware-bound work first and fan the rest out around it.

## Cut order under schedule pressure

1. KV-cache quantization
2. Large-N retrieval stage (the ANN index)
3. Premium tier

This briefly changed and then changed back, which is worth recording rather
than tidying away. Running D1's falsifier showed a lexical prefilter failing
the recall gate at 10,000 options on underdetermined queries, and the ANN stage
was promoted out of the cut order on that basis. Raising the per-question token
budget then let the shortlist grow from 256 to 2,048 **with the option criteria
intact**, and recall went to 1.0000 at every difficulty the probe generates —
with plain BM25 and no ANN index.

The recall failure was a budget problem wearing a retrieval problem's clothes.
The ANN stage returns to the cut order as what the plan called it: an
optimisation. A BM25 scan of 10,000 options is not free and an index is the
right answer at 100,000, but it is no longer load-bearing for correctness.
Numbers in `docs/decisions.md` §1.

All three are optimisations, not dependencies. **Never cut the calibration
training or the eval suites.** They are the product — the differentiator is
published calibration evidence, and there is no version of this project that
ships without it.

Plan a v2 cycle aimed purely at data scale and domain coverage.

## Risks

**Calibration transfer.** Outcome-grounded calibration fitted on public and
synthetic data may not transfer to users' domains. Mitigated by per-domain
conformal wrappers users fit on their own labels — implemented
(`trigon.calibration.conformal`), because the mitigation is worth more shipped
early than promised.

**Base-checkpoint availability.** Some families release fewer non-instruct
checkpoints over time. Verify at kickoff.

**Prefix-LM conversion cost.** Converting a causal base may cost quality before
fine-tuning recovers it. De-risked by the causal fallback layout, which the
compiler supports today (`bidirectional=False`) and which schema-first ordering
keeps workable.

**KV quantization under-delivers.** It is an optimisation, not a dependency —
the stack works with FP8 KV or no cache at all, and it is first in the cut
order.

**The category crowds fast.** The shape is commoditised: encoder-with-heads,
small-decoder fine-tunes, parallel constrained decoding. Phase 0 benchmarks the
strongest of them as baselines. Durable differentiation lives in calibration
training at scale, in the eval standard, and in the serving stack — not in the
architecture, which is close to multi-task classification over a shared encoder
with a schema compiler bolted on.

**Head-to-head numbers depend on API access we may not get.** The suites must
stand alone without competitor numbers in them. They do: every suite runs
against any `Engine`, and the floor backend means a reader can reproduce the
harness before reproducing any result.
