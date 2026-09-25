# Evals

Four suites, the release gates and one closed loop, all on one runner. The
harness — not any single result — is the durable asset, so everything here runs
on a fresh clone with no weights.

| | What it answers |
| --- | --- |
| §1 Calibration suite | are the probabilities worth believing |
| §2 Jaggedness suite | which documented failure modes we have, and which we keep |
| §3 Cardinality gate | does the prefilter pass the true option through |
| §4 Reference run | does the whole pipeline close, and do the gates bite |
| §5 Workflow suite | does it make the right decisions, at what cost and latency |
| §6 Plausibility | does its evidence match what people highlighted, above a vocabulary floor |

`trigon eval all` runs §1–§5 and exits non-zero on any failure. §6 needs a
corpus with human rationales, so it runs in `scripts/train_corpus.py`.

```bash
trigon eval all -n 200 --out reports/run.md   # exits non-zero on a failed gate
```

## 1. Calibration suite

ECE, adaptive ECE, MCE, Brier, NLL and reliability diagrams, per primitive and
per domain, plus conformal coverage checks.

The per-primitive cut was claimed in this sentence from the first draft and
not implemented until the run that needed it. A temperature is fitted *per
primitive*, so a primitive is exactly the unit at which a fit can go wrong,
and the pooled number cannot name the part that failed — the same argument
that made `worst_question_over_baseline` advisory rather than absent.

**Release gates**, from `trigon.limits`:

| Gate | Limit | What it checks |
| --- | ---: | --- |
| `sample_size` | ≥ 5,000 | the run is large enough for ECE to mean anything |
| `gate_is_testable` | floor p95 ≤ ½ × limit | a calibrated model would clear the gate with room |
| `accuracy_over_baseline` | ≥ +0.05 | the model uses its input at all; advisory on a drawn-annotator corpus, where no predictor can pass it |
| `brier_over_marginal` | ≥ +0.02 | the same, on a drawn-annotator corpus: 1 − Brier / the training marginal's Brier |
| Workhorse ECE (and adaptive ECE) | ≤ 0.05 | the model |
| Premium ECE | ≤ 0.03 | the model |
| Quantized-vs-BF16 ECE delta | ≤ 0.01 | the serving path |
| `conformal_coverage` | ≥ target − 3σ | the wrapper's only promise |
| `worst_question_over_baseline` | ≥ +0.05 | blocking on a backbone run, advisory on the spike; the pooled lift hides a question answered by rote |
| `worst_primitive_*_ece` | ≤ tier limit | blocking on a backbone run, advisory on the spike; the pooled ECE cancels |

The first three gate the *measurement and the premise*, not the model, and they
run first. Two of them exist because a run failed to catch something: see §4
for the model that passed every ECE gate at 45.6% accuracy.

The delta gate exists because probabilities degrade well before argmax does.
An accuracy-only check would wave through a KV bit-width that quietly
destroyed calibration — which is why quantization is gated on ECE and why
temperature layers are re-fitted after any change to the serving path.

**What feeds it now.** For most of this phase, nothing did: `check_gates` took
a `quantized=` run and the only caller that ever passed one was a unit test
that perturbed probabilities by hand. A gate whose only input is synthetic
tests the gate, not the system — the same mistake as a benchmark you only run
when you expect to win. `trigon train` now builds a real quantized twin of the
model it just trained (`TorchReadoutBackend.quantized()`: weight-only,
symmetric, per-output-channel int8 over every linear and the attention input
projection), runs the full calibration suite through it under the same
temperature, and publishes it as its own row beside the fp32 one. On the
reference configuration it scores an identical accuracy and an ECE 0.0002 away,
which is the shape the gate predicts — the decision survives, the distribution
under it moves.

Two honest limits on that. It is *weight* precision, not the KV-cache
bit-width phase 3 actually ships; they share the property the gate is about (a
serving-numerics change that spares argmax) and nothing else. And the weights
are rounded onto the int8 grid and held in fp32, so the matmul is fp32 over
int8-representable values — the standard way quantization error is measured,
and identical to an int8 kernel up to accumulation order. It is deliberately
not built on `torch.ao.quantization`, which is scheduled for removal in torch
2.10; a gate on a deprecation clock is a gate that stops running.

### Pooled ECE cancels, so the worst primitive is gated too

ECE averages the *signed* confidence gap within each bin before taking the
absolute value. Two heads sharing a bin and erring in opposite directions
therefore offset each other, and the pooled figure comes out below either of
them. Constructed, at 8,000 predictions per head over shared bins:

| | Underconfident head | Overconfident head | Pooled |
| --- | ---: | ---: | ---: |
| Opposite directions | 0.0574 | 0.0279 | **0.0148** |
| Same direction, same magnitudes | 0.0574 | 0.0372 | 0.0473 |

The second row is what a reader assumes pooling does — the pooled number lands
between its parts. The first row is what actually happens when the errors have
opposite signs, and it lands below both.

**This is not hypothetical.** Seed 1 of the 8,000-case sweep reports pooled ECE
0.0516 against a 0.05 gate, and its parts are:

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `choice` | 6,000 | 0.847 | 0.791 | −0.056 | 0.0885 |
| `noul` | 6,000 | 0.669 | 0.701 | +0.033 | 0.0928 |
| `score` | 6,000 | 0.245 | 0.264 | +0.019 | 0.0193 |

Two heads at nearly double the limit, in opposite directions, pooling to a
number that misses the gate by a whisker. A gate reading only the pooled
figure can certify a model with no well-calibrated head in it.

It also settles what had been a live hypothesis: `score`, the head whose
fitted temperature swings by a factor of sixteen across seeds, has the *best*
calibration of the three. The suspicious temperature was not the problem, and
the per-primitive table is what said so — the pooled number could only report
that something was wrong.

`worst_primitive_<tier>_ece` reads the worst single primitive. It is advisory
for exactly the reason `worst_question_over_baseline` is: at a 128-wide
two-layer spike no per-primitive gate passes, and a gate nothing can pass
measures capacity rather than honesty. Both flip with the same
`require_per_question` switch, so they cannot drift apart.

**They block on a backbone run, since 2026-09-25.** Q16 promised the flip "the
moment a real backbone lands", and `trigon train`, `scripts/train_corpus.py`
and `scripts/regate.py` now set `require_per_question` whenever the model is a
pretrained backbone. It costs one committed seed its certificate: the
synthetic suite's seed 0 has a Score head at ECE 0.0551, which the pooled
0.0408 hid (`reports/synthetic/README.md`). On a drawn-annotator corpus the
per-question *accuracy* gate stays advisory with the pooled one, for the
reason `brier_over_marginal` exists.

### Conformal coverage is gated, and its power is stated

A conformal wrapper promises one thing: that its prediction set contains the
truth at the target rate. `README.md` tells a user facing calibration transfer
to fit one on their own labels and *read the coverage number rather than the
ECE*. For most of phase 0 that number was computed, printed to stderr, and
checked by nothing.

`trigon fit --conformal-out` now measures coverage on a third split and exits
non-zero if it falls below a floor — and the floor is derived, not chosen.
Empirical coverage on *n* held-out points is Binomial(*n*, 1 − α)/*n* even for
a perfect predictor, so its standard deviation is √(α(1−α)/*n*): 0.0077 at
n = 1,500, but 0.030 at n = 100. A fixed tolerance would be vacuous at small
*n* and spuriously red at large *n*, which is exactly the failure
`gate_is_testable` exists to prevent for ECE.

Three sigma, one-sided. The false-positive rate is then 0.135% at every *n*.
The *power* is not constant, and stating it is the point:

| n | floor | true 0.88 | true 0.87 | true 0.85 | true 0.80 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 0.8480 | 4.4% | 12.9% | 46.2% | 98.1% |
| 1,500 | 0.8768 | 35.0% | 78.2% | 99.8% | 100.0% |
| 3,000 | 0.8836 | 72.6% | 98.6% | 100.0% | 100.0% |
| 6,000 | 0.8884 | 97.7% | 100.0% | 100.0% | 100.0% |

So it catches a broken wrapper almost anywhere and a two-point shortfall only
with thousands of held-out answers. Read it as a floor against a wrapper that
does not work, not as an assurance that one is exact — the same reading
`accuracy_over_baseline` asks for. The first draft of the test asserted that a
0.87-covering predictor is caught 95% of the time at n = 1,500; it is caught
78% of the time, and that correction is why this table exists rather than a
sentence.

**The gate is one-sided on purpose.** A predictor that returns every option
covers perfectly and says nothing, so `mean_set_size` is reported beside
coverage — but over-covering is a utility problem, not a broken promise, and
there is deliberately no ceiling to pair with the floor.

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

Decision D1 puts the retrieval trigger at 1,024 options or 65,536 question
tokens with a 2,048-option shortlist, and names its own falsifier: recall at the
shortlist size must hold at **0.99**, because everything below the prefilter's
recall is accuracy no model quality recovers.

```bash
python -c "from trigon.evals import run_cardinality_gate; run_cardinality_gate(on_result=print)"
```

**The gate sweeps query difficulty, and the difficulty is the measurement.**
When a query names all four fields of its true option, that option is a 4/4
lexical match against distractors that are at best 3/4, and recall@256 reads
1.0000 at every option count — which tells you nothing except that exact
matching is easy. `drop_slots` leaves fields unstated, as real tickets do.

At the original 256-option shortlist the gate held for queries stating three or
four fields and **failed at two** (0.9400) — and holding it needed a
~1,536-option shortlist, which fitted the then per-question budget only with
the option criteria stripped out.

Raising the per-question budget to 65,536 tokens changed that. A 2,048-option
shortlist now fits **with criteria intact** (58,449 tokens), and recall is
1.0000 at every difficulty the probe generates, including a query naming a
single field — with plain BM25 and no ANN index:

| Query states | recall@2048, 10,000 options |
| --- | ---: |
| 3 of 4 fields | 1.0000 |
| 2 of 4 | 1.0000 |
| 1 of 4 | 1.0000 |

The recall failure was a budget problem wearing a retrieval problem's clothes.
What survives is the difficulty sweep, which is what made it visible;
`docs/decisions.md` §1 has the full correction, including putting the ANN stage
back in the cut order.

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

## 4. The reference run

```bash
trigon train --out reports/run.md --save-model reports/run.pt
```

Train the reference model on outcome-grounded data, fit a temperature on the
training split, measure on a held-out split generated from a different seed,
and run the gates. Everything is seeded — data generation, weight
initialisation and shuffling — so the run reproduces from its settings rather
than from a checkpoint, and each report prints the exact command that produced
it at the top. `reports/reference-run.md` is not what the bare command above
gives you; run the one in its own header.

Its purpose is not model quality. The reference model is a spike: 128-wide, two
layers, a hashing tokenizer, 2,500 cases. Its purpose is that the pipeline
closes and the gates are exercised by something rather than asserted about
something.

**It produced the most useful result available: a pass that exposed a missing
gate.**

| | Value |
| --- | ---: |
| Accuracy | 0.4561 |
| Marginal predictor | 0.3922 |
| Lift | **+0.0639** |
| ECE | 0.0111 |
| Adaptive ECE | 0.0158 |
| Noise floor (p95) | 0.0084 |
| Brier | 0.5947 |

Every ECE gate passed. The reliability bins aligned to within 0.024, and the
measured error sat above the simulated floor, so it was a real measurement
rather than luck. And the model had learned the label frequencies and little
else — 45.6% accuracy against a marginal predictor's 39.2%.

A model that reports true marginals is calibrated **by construction**. So a
calibration-only gate set certifies the one model guaranteed to be useless, and
hands it a reliability diagram on the way out. `accuracy_over_baseline` now
computes the marginal predictor's accuracy from the eval labels and requires
the model to beat it; it is a floor against the degenerate case, not an
accuracy target. The build plan's gates were ECE-only and would have passed
this model.

Temperature scaling moved ECE from 0.0112 to 0.0111 — correct behaviour, not a
failure. The model was already near-calibrated, and no temperature can make a
model use its input. Post-hoc calibration fixes the shape of a distribution,
never what it is conditioned on.

### The option-scoring ablation

The plan names this a phase-1 question: a readout slot per option, or one slot
dotted against pooled option states? The second is what makes huge option sets
affordable — one slot regardless of cardinality — and it is the path the
compiler takes above 64 options.

Same data, same seed, same budget; only the head differs.

| | readout per option | dot product |
| --- | ---: | ---: |
| Final loss | **1.0366** | 1.1398 |
| Gap to Bayes closed | **20%** | 2.6% |
| Accuracy | **0.4561** | 0.3887 |
| Lift over marginal predictor | **+0.0639** | **−0.0036** |
| ECE | 0.0111 | 0.0177 |
| Adaptive ECE | 0.0158 | 0.0441 |
| Gates | **all pass** | `accuracy_over_baseline` **fails** |

Both columns are the *original* pair. The dot-product column is what a
collapsed head looks like; the repair below moves it past the left column.

Chance is 1.1552 and the Bayes-optimal loss for this generator is 0.5585. The
dot-product arm's loss rose between epochs 5 and 6 (1.1395 → 1.1398): it never
left chance.

Two things follow. **The ablation had an answer, and it was inconvenient — so
it got a second run.** At this scale the cheap head did not learn, and it is
the head the high-cardinality path depends on, so "budget more training effort
in phase 1" was not a good enough response. Instrumenting the trained
checkpoint showed the query had frozen to a constant across every input and the
option keys had collapsed onto each other during training. Three repair arms
later, adding each option's input embedding back into its key turns 0.254 into
0.847 on the one question either head learns — past the per-option head's
0.457, at one readout slot instead of *n*:

| Arm | `plan` accuracy | Pooled lift | Gates |
| --- | ---: | ---: | --- |
| Readout slot per option | 0.457 | +0.0673 | 5/5 |
| Dot product, as first shipped | 0.254 | −0.0002 | blocked |
| … + `match_normalize` | 0.254 | −0.0002 | blocked |
| … + `match_residual` | **0.847** | **+0.1962** | 5/5 |
| … + both | 0.457 | +0.0673 | 5/5 |

`docs/decisions.md` has the mechanism, the repair that was expected to work and
did not, and what one seed at spike scale does not establish.

**And it is the sharpest case for `accuracy_over_baseline` available.** The
dot-product arm is *worse than ignoring the state* — a negative lift — and it
still passes every ECE gate, adaptive ECE included. Under the build plan's
ECE-only gates this arm would have been certified shippable.

## 5. Workflow suite

```bash
trigon eval workflow -n 200
```

Fixed compute graphs over one state, with later steps depending on earlier
answers — the shape real pipelines have. Two ship
(`trigon.evals.workflows`), both with conditional steps:

| Workflow | Graph | Lexical floor | Calls/case |
| --- | --- | ---: | ---: |
| `support_triage` | route to a team → ask the refund question **only of billing tickets** | 0.2550 | 1.06 |
| `moderation_queue` | policy gate → classify and rate **only what the gate admitted** | 0.4150 | 1.16 |

Floor figures at `n=200, seed=0`, reproducible with
`trigon eval workflow -n 200`.

Two departures from the methodology this borrows from:

**Scored against resolved outcomes**, not against a frontier ensemble's
probabilities. Agreement-with-frontier scoring is vendor-graded: when the
reference and the candidate share an error, the error is invisible, and an
independent audit flagged exactly that on the competitor's own dashboard
(67.8% against reference answers that were themselves an average of two
frontier models). Every decision here has an outcome the generator computed, so
a wrong answer is wrong against reality. Reference agreement is still reported
as `reference_kl` when a case carries one, because it keeps our numbers
comparable — it just is not the headline.

**Cost and latency on the same run**, so a result is a point on a Pareto plot
rather than a percentage with no denominator. `mean_model_calls` is the axis
that matters here: a graph that asks the refund question only of billing
tickets lands at ~1.05 calls per case, not 2, and against a prompted baseline —
which pays per question — that difference is the whole argument.

**The routing step was built twice before it measured anything.** The first
draft put the option criteria's own words in the ticket text and the lexical
floor scored 0.832 — string overlap, not comprehension. The second made the
surface signal *anti-correlated* with the truth, and the floor scored 0.000,
which measures the distractor rather than the router. The shipped version gives
the complaint and the distractor comparable lexical pull, and the floor lands
at 0.255 against a 0.25 chance baseline. Same rule as the jaggedness suite: run
the floor against a benchmark before believing it.

## 6. Plausibility of evidence

```bash
python scripts/train_corpus.py hatexplain --out reports/hatexplain/run.md
```

When a corpus carries human rationales — HateXplain is the one that does —
`scripts/train_corpus.py` appends a plausibility table
(`trigon.evals.rationale`): does the model highlight what the annotators
highlighted? Token F1 and IOU F1, as ERASER (DeYoung et al., 2020) defines
them, macro-averaged over cases, scored on **words of the state** so that a
byte-level tokenizer and a word-level floor are measured on the same units.

Every model row sits above three floors computed on the same rationales, and
the rule is the jaggedness suite's: run the floor before believing the
number.

| Floor | What it highlights | Token F1 | IOU F1 |
| --- | --- | ---: | ---: |
| lexical floor | state words the question's own text contains | 0.025 | 0.002 |
| **rationale lexicon** | every word highlighted in at least half its training occurrences | **0.573** | **0.452** |
| every word | all of them | 0.434 | 0.217 |

Two seeds' evaluation splits, `reports/hatexplain/README.md`. The lexicon is
the floor that bites: on hate speech it is very nearly a slur list, and it
beats both the spike's trained span head (0.49 / 0.33) and highlighting
everything. *Every word* is the other one worth reading — its recall is 1 by
construction, so its token F1 is set entirely by how much of a post people
mark (34% of words), and a highlighter below it is worse than not choosing.

Plausibility is not gated. It says whether a person would agree with the
highlight, not whether the model used it; faithfulness is open
(`docs/ledger.md`).

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

## The conformal wrapper

The build plan's largest risk is that outcome-grounded calibration fitted on
public and synthetic data does not transfer to a user's domain. The mitigation
is that they fit a wrapper on a few hundred of their own labels, and a
mitigation that exists only in a design document is not a mitigation — so it
ships as a command:

```bash
trigon fit --backend torch --weights reports/reference-run.pt \
           --out temperatures.json --conformal-out profiles/accounts.json
```

Fitted on a split the temperatures never saw and reported on a third, because a
coverage number measured where the threshold was fitted is not a coverage
number. The shipped profile (`reports/conformal/accounts.json`, LAC, α=0.1):

```
target 0.90, achieved 0.9227 on 1500 held-out answers, mean set 2.49 of 4
```

Served on the certified model, with `options.conformal_profile` set:

```
truth=free   point answer=free  conf=0.600  set: ['free']
truth=pro    point answer=pro   conf=0.105  set: ['standard', 'pro', 'enterprise']
```

A singleton is a point answer with a guarantee. A wide set is the model saying
it cannot separate those options at that coverage — which, for the second case,
is the honest answer: its top two probabilities are 0.323 and 0.320.

## Operational gates

- Every quantization or serving change re-runs the calibration suite before
  rollout.
- The API pins model versions. Responses name a concrete build, never an
  alias — answers change under users when an alias moves, and
  `tests/test_server.py` asserts the response carries a version, not a name.
  The build is named from a hash of its weights, so two runs cannot answer
  under one version and an untrained deployment says so in the version string
  as well as on `/healthz`.
