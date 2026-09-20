# Evals

Three suites, one runner. The harness — not any single result — is the durable
asset, so everything here runs on a fresh clone with no weights.

```bash
trigon eval all -n 200 --out reports/run.md   # exits non-zero on a failed gate
```

## 1. Calibration suite

ECE, adaptive ECE, MCE, Brier, NLL and reliability diagrams, per primitive and
per domain, plus conformal coverage checks.

**Release gates**, from `trigon.limits`:

| Gate | Limit | What it checks |
| --- | ---: | --- |
| `sample_size` | ≥ 5,000 | the run is large enough for ECE to mean anything |
| `gate_is_testable` | floor p95 ≤ ½ × limit | a calibrated model would clear the gate with room |
| Workhorse ECE (and adaptive ECE) | ≤ 0.05 | the model |
| Premium ECE | ≤ 0.03 | the model |
| Quantized-vs-BF16 ECE delta | ≤ 0.01 | the serving path |

The delta gate exists because probabilities degrade well before argmax does.
An accuracy-only check would wave through a KV bit-width that quietly
destroyed calibration — which is why quantization is gated on ECE and why
temperature layers are re-fitted after any change to the serving path.

### The gate on the measurement comes first

Both ECE estimators are biased upward at small n: bin accuracy carries sampling
noise of order `sqrt(p(1-p)/m)`, and ECE averages the *absolute* gap, so noise
accumulates rather than cancelling. A perfectly calibrated model scored on a few
hundred examples reports a large ECE.

`metrics.noise_floor` measures exactly how large, by resampling labels from the
model's own predicted distributions — which produces a model that is calibrated
by construction — and reporting the ECE it still scores. On 4-way predictions:

| n | Mean ECE of a **perfectly calibrated** model | 95th percentile |
| ---: | ---: | ---: |
| 60 | ~0.124 | ~0.189 |
| 1,000 | ~0.030 | ~0.049 |
| 4,000 | ~0.015 | ~0.021 |

Read the middle row against the 0.05 workhorse gate: at n=1,000 a calibrated
model's 95th percentile *is* the gate. The gate is not a test there. That is why
`MIN_CALIBRATION_SAMPLES` is 5,000 and why `check_gates` refuses to certify a
run whose own simulated floor sits above half its limit.

Every published report prints the measured ECE beside the floor and states
whether the two are separable. **"Indistinguishable from perfectly calibrated at
this sample size"** is the strongest claim the data supports, and it is a
different claim from "ECE is 0.03".

This is the failure the whole section exists to prevent, and it has already
happened in public: an independent re-analysis observed that published Jev ECE
figures of 0.0505–0.0712 at n=60 are equally what serious miscalibration looks
like at that sample size. Reproducible evals are the differentiator, so our own
numbers have to survive the same scrutiny.

And equal-mass bins must not split ties — see `docs/decisions.md`.

Equal-width ECE is published because it is what everyone else publishes and it
makes our number comparable. Adaptive (equal-mass) ECE is published because it
is the honest one. Both, always.

## 2. Jaggedness suite

One benchmark per documented failure mode, so every claim of "we handle X" —
and every claim of "we also don't" — is a number.

| Benchmark | Measures | Headline metric |
| --- | --- | --- |
| `literal_reading` | explicit statements beating topical association | accuracy |
| `counting` | **non-goal**, measured anyway | accuracy |
| `date_comparison` | **non-goal**, measured anyway | accuracy |
| `indirection_depth` | accuracy vs. hops to the answer | `accuracy@depth=k` |
| `context_rot` | accuracy vs. distractor volume | `rot` (clean − padded) |
| `injection_steering` | can embedded text steer another answer | `flip_rate`, `mean_drift` |
| `contradictory_criteria` | confidence when criteria cannot separate options | `mean_confidence` (lower is better) |
| `negation_coherence` | P(claim) + P(complement) − 1 | `mean_incoherence` |
| `noul_choice_agreement` | same question, two primitives | `mean_disagreement` |

Cases are generated, not scraped: ground truth is constructed, difficulty is a
dial, and the suite ships in the repo at zero licence risk.

**Injection: steering, not detection.** An independent bench already shows the
incumbent *detecting* injections well, while their own jaggedness page admits
embedded instructions can steer *other* answers. Those are different
properties and only the second is a weakness. So every case is a pair — the
same routing question over a clean state and an injected one — and the metric
is how far the answer moved. A third variant checks the other direction: a
guardrail question *about* the injected text must still read it as data and
flag it. Robustness means judging adversarial text against the schema, not
refusing to look at it.

**Two benchmarks are contracts, not regressions.** Counting and date
comparison are explicit non-goals — the contract is "keep math in code" — and
they carry `non_goal = True`. Publishing a bad number on them is the point.

**The floor earns its keep here.** Running the lexical baseline against the
first draft of this suite caught three benchmarks it could ace by surface
statistics: label-correlated length in literal reading, and keyword leakage in
indirection and context rot. A benchmark the floor aces measures nothing.

## 3. Cardinality recall gate

Decision D1 puts the retrieval trigger at 1,024 options or 16,384 question
tokens with a 256-option shortlist, and names its own falsifier: recall@256 must
hold at **0.99**, because everything below the prefilter's recall is accuracy no
model quality recovers.

```bash
python -c "from trigon.evals import run_cardinality_gate; [print(r) for r in run_cardinality_gate()]"
```

The option sets are generated, and that is a consequence of the licence audit
rather than a convenience: UFET — the ~10k-type corpus the build plan named for
this — has no stated licence and its distant-supervision half derives from
LDC-licensed Gigaword, and the largest green corpus in `docs/data.md` is
CLINC150 at 151 classes, below the trigger. No permissively-licensed corpus
exists in the regime the feature is built for.

The probe therefore builds confusable sets at 256 to 10,000 options, with every
true option surrounded by near-neighbours sharing most of its words, and queries
that paraphrase rather than quote. For a recall measurement that is arguably
better than a found corpus: distractor similarity becomes a dial instead of
whatever the data happened to contain. CLINC150 stays as the real-data check at
151 classes.

## 4. Workflow suite

Fixed compute graphs over one state, with later steps depending on earlier
answers — the shape real pipelines have.

Two departures from the methodology it borrows from:

**Scored against resolved outcomes**, not against a frontier ensemble's
probabilities. Agreement-with-frontier scoring is vendor-graded: when the
reference and the candidate share an error, the error is invisible.
Independent audits have flagged exactly this. Reference agreement is still
reported as `reference_kl`, because it keeps our numbers comparable with
published ones — it just is not the headline.

**Cost and latency on the same run**, so a result is a point on a Pareto plot
rather than a percentage with no denominator. `mean_model_calls` is the axis
that matters against a prompted baseline: the baseline pays per question, the
model answers them all in one pass.

## Baselines


Every suite runs against any `Engine`, so the comparison is apples to apples:

- **`LexicalBackend`** — BM25-flavoured overlap. The floor. No weights needed.
- **`LLMBaselineBackend`** — a prompted chat model, with three ways to get a
  distribution: `LOGPROBS` (closest to a real distribution), `VOTING`
  (works anywhere, quantises at 1/k), `VERBALIZED` (what most production code
  actually does, and the one the calibration suite should measure rather than
  assume).
- **`TorchReadoutBackend`** — the reference architecture.

The baseline's system prompt states that state is data and never instructions.
Without that, the injection comparison measures our prompt rather than their
model.

## Operational gates

- Every quantization or serving change re-runs the calibration suite before
  rollout.
- The API pins model versions. Responses name a concrete build, never an
  alias — answers change under users when an alias moves, and
  `tests/test_server.py` asserts the response carries a version, not a name.
