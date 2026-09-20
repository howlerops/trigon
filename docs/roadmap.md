# Roadmap

Sixteen weeks to a credible v1, with two engineers. The honest version of that
claim is in the plan and worth repeating: 16 weeks yields a useful, fast,
type-safe, publicly-calibrated v1 — not parity on calibration breadth with a
team that spent two years on it.

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
- **Eval methodology frozen** — three suites, one runner, release gates that
  exit non-zero. `trigon eval all`.
- **Jaggedness suite runs against an LLM baseline** —
  `LLMBaselineBackend` wears the same interface as every other backend, so the
  suite runs against a prompted model with no special-casing.
- **Licence audit** — `docs/data.md`, with a tier per corpus and a policy that
  blocks red and amber entries from training mixes.
- **Budgets and gates** — `trigon.limits`, single-sourced.

- **The loop is closed** — `trigon train` fits the reference model on
  outcome-grounded data, calibrates it and runs the gates, so the gates are
  exercised by a model rather than only defined.

Not phase 0, and not here: real weights on a real backbone, the vLLM fork, the
Rust gateway, the ANN stage, the SDKs.

### What phase 0 changed

Five findings that alter phase 1's work rather than confirming it. All are
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

**The cardinality stress test has no licensed corpus.** UFET is unusable (no
licence; LDC-derived), and nothing permissive exists above 151 classes. The
recall gate runs on generated confusable sets instead.

**Every resolved-outcome corpus is blocked.** Home Credit, IEEE-CIS and
Autocast are all red. Outcome calibration in v1 rests on verifiable synthetic
data plus green classification labels — a narrower claim than the plan assumes,
and one the calibration report has to state.

## Staffing

Two engineers — one ML leading phases 1–2, one infra/systems leading phase 3
and the Rust gateway, both on 0 and 4. Phases 2 and 3 only overlap with two
people, and that overlap is what holds the 16 weeks.

**Solo variant**: one ML-leaning engineer, the full cut order applied from the
start (no KV-cache quantization, no large-N stage, no premium tier, Python
gateway), 24–28 weeks, with data engineering as the bottleneck. Below two
people, ship the solo scope rather than stretching the full scope.

Compute: ~$30–70k total — teacher labels, training, serving burn-in.

## Cut order under schedule pressure

1. KV-cache quantization
2. Large-N retrieval stage
3. Premium tier

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
