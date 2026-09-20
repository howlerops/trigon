# Decisions

Answers to the four open questions in the build plan, plus the calls the
implementation forced. Each one names what would change our mind, because a
decision without a falsifier is a preference.

## Sign-off, 2026-09-20

Phase 0 closed. All five decisions ratified as written, all eight findings
acknowledged. The six open questions resolved as follows:

| | Question | Resolution |
| --- | --- | --- |
| Q14 | The "96% on 50 cases vs 84–86%" figure | **Dropped permanently.** No primary source surfaced. It is not to be quoted, in a deck or anywhere else |
| Q15 | L4 unit economics, unsourced | **Burn-in pulled forward.** It is the whole cost argument; waiting until phase 3 puts it at week 13, after the point where it would shape positioning |
| Q16 | `accuracy_over_baseline` pools questions | **Gate per question**, implemented now and advisory until a real backbone lands. See *A calibration-only gate set…* below |
| Q17 | BoolQ and FEVER, CC BY-SA on a derived model | **Counsel opinion requested.** Two corpora today, but the question recurs for every CC BY-SA set, so it is worth answering once |
| Q18 | LMSYS gated agreement; Kaggle terms | **Dropped.** None is load-bearing now that outcome grounding comes from verifiable synthetic data |
| Q19 | Staffing | **No hires.** One operator plus parallel agents; see `docs/roadmap.md`, *Staffing*, which this replaces rather than amends |

Q19 is the one that changes how the rest is built, and its consequences are
written up there: compute and verification bind where engineer-hours used to,
the eval harness stops being only a differentiator and becomes the mechanism
that catches unfinished work, and parallelism buys breadth rather than depth.

---

## 1. Per-request option/token budget before the retrieval stage

**Decision.** The large-cardinality stage engages when **either** trigger
fires: more than **1,024 options** in one Choice, or a single compiled question
over **65,536 tokens**. The prefilter returns a **2,048-option shortlist**. The
numbers live in `src/trigon/limits.py`; nothing else may hard-code them.

These are far above the contract we mirror, deliberately — see "Capacity"
below. Drop-in compatibility is unaffected: a superset never rejects a request
the narrower contract would have accepted, and `COMPAT_BUDGET` reproduces their
limits exactly for like-for-like benchmarking.

**The envelope was checked, and the plan had it wrong.** The plan states a
"~32k context budget, matching Jev's", and the first version of `limits.py`
carved 32,768 tokens into state, schema and readout slices. The published
contract is not one limit but two: **64k tokens per request** (state plus every
question) and **32k for state plus the longest single question**
(docs.typesafe.ai, verified 2026-09-20). Building to a single flat number got
this wrong in both directions — it halved the total budget, and it missed the
constraint that actually governs a high-cardinality Choice.

The corrected carve-up, then raised — see "Capacity" below for why the two
halves are sized so differently:

| Slice | Compat | Extended | Why |
| --- | ---: | ---: | --- |
| Total request | 65,536 | **524,288** | State plus every question |
| State + longest question | 32,768 | **131,072** | The binding constraint for large option sets |
| State | 16,384 | **65,536** | The only term quadratic in the whole request |
| Schema | 40,960 | **393,216** | Block-diagonal, so near-free to grow |
| Readout slots | 4,096 | **32,768** | One per question under dot-product scoring |
| Questions per request | 64 | **1,024** | Linear cost; their cookbook batches 13 |

`max_question_tokens` is therefore **derived, not configured**: whatever the
per-question envelope leaves once state has taken its budget. It cannot drift
away from the contract it mirrors.

The **count** trigger protects the readout head. With a readout slot per
option, 1,024 options is 1,024 slots — a quarter of the readout budget for one
question. Above the crossover the compiler switches to dot-product scoring
(one slot regardless of cardinality), so the count trigger is really about
where per-option slots stop being affordable at all.

The **token** trigger is per-question, and it is why a count-only trigger is
not enough. A bare option name runs ~4 tokens, so 1,024 names is ~4k tokens.
The same 1,024 options *with criteria* run ~20 tokens each — 20k tokens, which
overran the 16,384-token budget this section originally set and still sizes a
third of the 65,536 it has now. Option count alone cannot predict whether a
question is admissible; tokens can.

**Shortlist size: 2,048**, raised from 256. The original 256 was chosen to sit
just above the 255-option cap the competitor imposes natively, so that our
two-stage path never scored fewer candidates than their one-stage path. That
was a limit matched to someone else's product rather than derived from
anything, and it is precisely where the recall gate failed. At 2,048 the
shortlist fits the per-question budget with the option criteria intact, and
recall holds at 1.0000 across every difficulty the probe generates.

**What would change our mind, and what happened when we ran it.**
`recall_at_k` at the shortlist size, gated at **0.99**, because everything
below the prefilter's recall is accuracy no model quality recovers.

That falsifier is implemented (`trigon.evals.cardinality`), it has been run,
and **it failed** — which is the most useful thing a falsifier can do.

It could not be run on the corpus the plan named. UFET has no stated licence
and its distant-supervision half derives from LDC-licensed Gigaword
(`docs/data.md`), and no permissively-licensed corpus exists near the
1,024-option regime; the largest green set in the audit is CLINC150 at 151
classes. So the probe generates confusable option sets at 256 to 10,000
options, every true option surrounded by near-neighbours sharing most of its
words. For a recall measurement that is arguably better than a found corpus:
difficulty becomes a dial instead of whatever the data happened to contain.

That dial is the point. Results, at a 256-option shortlist:

| Query states | 1,024 options | 4,096 | 10,000 |
| --- | ---: | ---: | ---: |
| all 4 fields | 1.0000 | 1.0000 | 1.0000 |
| 3 of 4 | 1.0000 | 1.0000 | 1.0000 |
| **2 of 4** | 1.0000 | 1.0000 | **0.9400** ❌ |

The first two rows say nothing: when a query names every field, the true option
is a 4/4 lexical match and every distractor is at best 3/4, so any prefilter
wins. Real tickets do not name every field. At two fields stated the gate
fails at 10,000 options.

**How far a bigger shortlist gets us, measured rather than assumed:**

| Shortlist | recall (2 of 4 fields, 10,000 options) |
| ---: | ---: |
| 256 | 0.9533 |
| 1,024 | 0.9800 |
| **1,536** | **1.0000** ✅ |

So the shortlist that holds the gate is about 1,536 — six times what we
budgeted. And that is where it collides with the token budget from the top of
this section:

| Shortlist | With criteria | Names only | Per-question budget |
| ---: | ---: | ---: | ---: |
| 256 | 7,367 | 2,735 | 16,384 |
| 1,024 | 29,209 ❌ | 10,775 | 16,384 |
| 1,536 | 43,775 ❌ | 16,138 ⚠️ | 16,384 |

A 1,536-option shortlist fits **only if the criteria are stripped** — and the
criteria are what tell the model how to choose between options. Even then it
uses 16,138 of 16,384 tokens, leaving 1.5% headroom.

**What we changed, and then what changed it back.** The first response was
three things: restate the gate with a difficulty level; go two-stage inside
retrieval (lexical to ~1,536 names, rerank to 256 with criteria); and move the
ANN stage out of the cut order, on the grounds that lexical matching cannot
separate options a short query underdetermines.

Then the budgets were raised (see "Capacity" below) and the picture changed.
At a 65,536-token per-question budget a **2,048-option shortlist fits with the
criteria intact** — 58,449 tokens, 11% headroom — where at 16,384 it did not
fit at all. Re-running the falsifier there:

| Query states | recall@2048, 10,000 options |
| --- | ---: |
| 3 of 4 fields | 1.0000 |
| 2 of 4 | 1.0000 |
| **1 of 4** | **1.0000** |

So the gate holds at every difficulty the probe can generate, including a query
that names a single field — with a plain BM25 prefilter and no ANN stage.

**The correction:** the recall failure was a budget problem wearing a retrieval
problem's clothes. The prefilter was never the bottleneck; a 256-option
shortlist was, and 256 was chosen to match a competitor's cap rather than
derived from anything. What survives from the first response:

1. **The gate keeps its difficulty level.** "recall@k ≥ 0.99" is not a claim
   until it says which queries, and `run_cardinality_gate` sweeps difficulty by
   default. This is what made the original failure visible at all.
2. **The two-stage rerank is dropped.** It existed to work around the token
   budget; the budget is gone as a constraint, and an unnecessary stage is
   latency and code for nothing.
3. **The ANN stage goes back into the cut order**, where the plan had it. It is
   an optimisation — a BM25 scan of 10,000 options is not free, and an ANN
   index is the right answer at 100,000 — but it is no longer load-bearing for
   correctness. `docs/roadmap.md` is corrected.

Lowering the gate was the one option ruled out in advance, and it stayed ruled
out.

## 2. Ordinal-aware loss for Score, or plain cross-entropy

**Decision.** Ship plain cross-entropy over levels in phase 1. Ablate
squared-EMD in phase 2 and adopt it only if it improves **calibration**, not
accuracy.

**Why.** Cross-entropy treats the levels of a Score as unordered, which is
plainly wrong: predicting "slightly positive" when the truth is "neutral"
should cost less than predicting "terrible". An ordinal loss (squared earth
mover's distance over the CDF) encodes that. But the reported score is an
**expectation over the distribution**, not an argmax, so cross-entropy already
gets partial credit for putting mass near the truth — much of what an ordinal
loss buys is already there.

The real argument is calibration, not accuracy. Cross-entropy is a strictly
proper scoring rule: the loss-minimising prediction is the true conditional
distribution. Squared-EMD is *not* proper in that sense — it can prefer a
distribution that is smoother than the truth, because smoothing is cheap under
a transport cost. On a project whose entire differentiator is published
calibration evidence, replacing a proper scoring rule with an improper one is
not a free win. That is a reason to measure it carefully, not to dismiss it:
the smoothing may be a good inductive bias on genuinely ordinal data.

Which is why the order-awareness that mattered most went somewhere else.
`score_confidence` uses the **dispersion** of the score under the predicted
distribution, not entropy, so a Score split between two adjacent levels reads
as confident and the same entropy split between the extremes reads as
uncertain (`tests/test_confidence.py`). That is an ordinal-aware statistic that
costs nothing at training time and cannot break propriety.

**Ablation design (phase 2).** Same data, same schedule, three arms: CE,
squared-EMD, CE + EMD as an auxiliary term. Report per-arm ECE, Brier,
reliability curves, and mean absolute score error. **Adopt only if ECE
improves and Brier does not regress.** An accuracy-only win is not sufficient
grounds.

---

## 3. License target, and the dataset license audit

**Decision.** **Apache-2.0** for code and weights, with no field-of-use
restriction and no acceptable-use rider. The `LICENSE` file in this repo is
the unmodified Apache-2.0 text. **The dataset audit has been run** — results
and evidence in `docs/data.md`, summarised at the end of this section.

**Why.** The differentiator against an API-only competitor is "open weights,
self-hostable". A licence with restrictions attached does not deliver that
claim: legal teams treat open-weight-with-restrictions as a bespoke licence
requiring review, which is the exact friction the positioning is meant to
remove. Apache-2.0 also carries an explicit patent grant, which
non-standard "open" licences usually do not, and it is the licence most of the
base checkpoints we would adapt from already use — so an Apache-2.0 release
introduces no new compatibility question downstream.

The cost is real: anyone may host the weights commercially, including the
incumbent. We are choosing that over the adoption friction, because the durable
asset in this project is the eval harness and the calibration data pipeline,
neither of which is in the weights.

**Dataset licensing is a separate and stricter policy**, because a permissive
code licence does not launder a research-only corpus. Three tiers, enforced as
a phase-0 gate:

| Tier | Licences | May be used for |
| --- | --- | --- |
| **Green** | Apache-2.0, MIT, CC0, CC BY 4.0, CC BY 3.0, ODC-BY | Training, eval, redistribution |
| **Amber** | CC BY-SA, NC clauses, "research use" terms, platform terms of use | Eval only. Never in a training mix, never redistributed |
| **Red** | Competition-only terms, unclear provenance, no stated licence | Prototyping on a local copy only; blocked from any shipped artifact |

Concrete consequences for the seed list in the plan: Banking77, MASSIVE,
Circa, HelpSteer2/3 (CC BY 4.0), GoEmotions (Apache-2.0) and Civil Comments
(CC0) are green — each confirmed against its dataset card, not inherited from
the plan. Amazon ESCI is **not**: see below. BoolQ and FEVER (CC BY-SA) are amber — the
share-alike term is a redistribution question for a derived model, so they are
eval-only until counsel says otherwise. The Kaggle entries (ASAP, Home Credit,
IEEE-CIS) are red by default: competition terms are frequently
competition-only, and a permissive HF mirror must be found or the dataset
dropped. Everything the plan marks "verify" stays red until verified — "verify"
is a blocker for training, not a footnote.

`docs/data.md` holds the audit table with a status column per corpus. A
dataset enters a training mix only by moving to green in that table.

**What the audit actually found**, checked against dataset cards and source
repositories on 2026-09-20 rather than against the plan's assumptions:

- CLINC150 clears to **green** (CC BY 3.0) — which matters, because it is the
  out-of-scope-option corpus that trains abstention.
- measuring_hate_speech clears to **green** (CC BY 4.0).
- AG News and DBpedia-14 were one row in the plan and do not share a licence:
  DBpedia-14 is CC BY-SA 3.0 plus GFDL (**amber**), AG News carries **no stated
  licence at all** (**red**).
- Amazon Reviews 2023 is governed by Amazon's Customer Reviews Terms of Use
  despite permissive repository scripts. Platform terms are not an open
  licence: **amber**.
- **UFET is unusable** — no stated licence, and its distant-supervision half
  derives from LDC-licensed Gigaword.
- **Autocast is unusable** — the code is MIT, but the dataset is hosted "with
  permission from Metaculus for research purposes only".
- LMSYS is behind a custom, gated dataset licence agreement: **red** until
  counsel reads it.
- **Amazon ESCI drops from green to amber.** Its repository carries an
  Apache-2.0 LICENSE and a README saying "this project is licensed under the
  Apache-2.0 License" — a code licence, with no separate grant for the data.
  This policy exists precisely to reject that pattern, and applying it to a row
  we wanted to keep is the test of whether the policy is real.

Three of those change the plan rather than confirming it, and all three are
written up in `docs/data.md`: the cardinality stress test has no licensed
corpus, every resolved-outcome corpus is blocked, and the trainable pool has no
e-commerce domain left.

**What would change our mind on the weights licence.** Evidence that the
practical alternative is not "someone hosts our weights" but "nobody adopts
because there is no hosted option" — in which case the answer is to run a
hosted service ourselves under the same Apache-2.0 weights, not to restrict
the licence.

---

## 4. Seeding the domain mix from FlowForge-style pipeline decisions

**Decision.** Yes as an **eval workflow and a cookbook**, no as a
**training-mix seed** in v1.

**Why the split.** As a first-party use case it is genuinely valuable: a
pipeline's routing and gating decisions are exactly the shape this model
serves, and — critically — those decisions have **resolved outcomes**. Did the
retry succeed? Did the escalation turn out to be necessary? That is
outcome-grounded eval data of the kind that is otherwise expensive, and it is
the ground-truth variant of the workflow suite the plan asks for. The
`Workflow` type in `src/trigon/evals/workflow.py` exists for exactly this
shape, including the conditional steps a real pipeline has.

As a training seed in v1 it is the wrong trade. First-party pipeline data is
narrow — one company's conventions, one product's taxonomy — and the v1 risk
that matters most is calibration failing to transfer to users' domains.
Over-weighting a single narrow domain in the training mix makes that risk
worse while looking like progress on the eval that shares its distribution.
Second, it creates a dependency between two projects' schedules in a 16-week
plan whose critical path is already data engineering.

**Revisit in v2**, when domain coverage is the explicit goal and first-party
data is one of twenty domains rather than a disproportionate slice of three.

**What would change our mind.** If the synthetic workflow generator turns out
not to produce realistic multi-step dependency structure — pipelines where a
later question's option set depends on an earlier answer — then real pipeline
traces become the cheapest source of that structure and go into the mix, with
their share capped and reported.

---

## Capacity: input and output ceilings

**Decision.** Raise the budgets well past the contract we mirror, and size the
two halves differently, because attention cost under our isolation rules is
asymmetric.

`CompiledRequest.attention_pairs` counts the (query, key) pairs the mask admits:

```
  sum over questions of (schema_q)^2                          -- block-diagonal
+ state^2                                                     -- quadratic in the whole
+ sum over questions of readout_q x (schema_q + state + readout_q)   -- linear
```

Measured — the tables live in `docs/architecture.md`, generated by
`scripts/attention_table.py` and drift-tested, so they are not repeated here.
The shape of them:

| Layout | Saving vs dense |
| --- | ---: |
| 32k state, 4 questions × 20 options | 7.4% |
| 8k state, 20 questions × 50 options | 85.4% |
| 8k state, 64 questions × 50 options | **96.3%** |

A question's schema block attends only to itself, so schema cost is the *sum*
of per-question squares rather than the square of their sum. Readouts are
linear. **State is the only term quadratic in the whole request**, which is
what the first row shows: when state dominates, the mask buys almost nothing.

So questions and option sets are cheap to grow and state is not, and the
budgets say so: schema gets 393,216 tokens and 1,024 questions, state gets
65,536. A full-size request costs between 67k and 173k dense-equivalent tokens
depending on how that schema budget is split — the honest form of a
long-context claim is that range, not "free" and not a single number.

**On the output side**, this contract has no generated tokens, so capacity
means the answer surface: more questions per request, and more options or
levels per answer. Both are raised. The constraint that appears instead is
*response size* — 100,000 options is roughly 2 MB of JSON, and a caller acting
on the top few should not pay to serialize the tail. `top_probabilities`
returns the k most probable options, always including the selected one, with
the real probabilities rather than renormalised ones and a `probability_mass`
field saying how much of the distribution they cover. Confidence, the Score
expectation and the conformal set are all computed on the full distribution
before any trimming: a confidence derived from a truncated vector would read
high simply because the tail was dropped.

**What would change our mind.** Two measurements neither of which we have yet:
KV memory at 65,536 state tokens on an L4 at target QPS, and end-to-end p50/p99
at the top of the range. The budgets are sized from attention arithmetic, which
is necessary and not sufficient — if the memory or latency numbers do not hold,
state comes down first, because it is the term that costs.

---

## Decisions the implementation forced

These were not in the plan's open-questions list. They came up while building
and are recorded here because each one is load-bearing.

### State does not attend to the schema

The plan's no-context-rot claim needs more than a readout block mask. If state
may attend to the schema, adding a second question changes the state's hidden
states, which changes the *first* question's answer — adding a question
silently moves an answer nobody asked about. So the default is
`state_attends_to_schema=False`: state encodes itself, a readout sees its own
schema block plus state, and per-question independence is exact and testable
(`tests/test_independence.py`).

The cost is real: state is encoded question-agnostically, so all
state/schema cross-referencing happens in the readout slots rather than
throughout the stack. The flag exists so phase 1 can measure what that costs.
It is also a caching win — the state KV becomes reusable across different
question sets over the same document.

### Positions are group-local

A block mask alone does not buy independence either. With ordinary
sequence-global positions, inserting a question shifts every later token's
position and moves hidden states the mask never let it see. Each schema block,
the state, and each question's readout slots therefore start at position 0,
with a learned segment-type embedding keeping the three kinds distinguishable.
This is also what makes a cached schema prefix portable: a prefix computed at
one offset is wrong at another.

Found by writing the test and watching it fail. It would have been easy to
miss on paper and expensive to find after training.

### Confidence is derived after calibration, and Score's statistic is ordinal

Confidence is computed from the temperature-scaled distribution, never from
raw logits — otherwise it looks like a probability and is not one, and the
confidence-gated escalation policy spends premium-tier money on the wrong
requests. Choice uses `1 - H(p)/log(n)`, scaled so a 2-option and a 77-option
answer are comparable. Score uses dispersion, because entropy discards the
level ordering.

### Equal-mass ECE must not split ties

Sorting by confidence and cutting into equal-sized bins slices through runs of
identical confidence and manufactures gaps out of sampling noise: a constant
predictor at the base rate — calibrated by definition — scored 0.064 before
this was fixed, which would have failed its own release gate. Both ECE
estimators are also biased upward at small n, so gates are stated at a fixed
sample size. See `src/trigon/calibration/metrics.py`.

### A release gate has to be a test before it can be a gate

Both ECE estimators are biased upward at small n, and simulating that bias
turned up a problem with our own gate rather than only with someone else's
numbers. On 4-way predictions a **perfectly calibrated** model scores a mean
ECE of about 0.12 at n=60, about 0.030 at n=1,000, and about 0.015 at n=4,000.
At n=1,000 its 95th percentile is roughly 0.049 — the whole 0.05 workhorse
gate. A gate that a calibrated model fails half the time is not measuring the
model.

So `check_gates` now gates the measurement before the model. Two checks run
first: the run must carry at least `MIN_CALIBRATION_SAMPLES` scored questions,
and the run's own simulated floor must sit at or below half the limit. A run
that cannot separate a calibrated model from a miscalibrated one certifies
neither.

`metrics.noise_floor` is what makes that possible: it resamples labels from the
model's own predicted distributions, which produces a model that is calibrated
by construction, and reports the ECE it still scores. Every published report
prints the measured ECE beside that floor and says whether the two are
separable. "Indistinguishable from calibrated at this sample size" is the
strongest honest claim, and it is a different claim from "ECE is 0.03".

This was prompted by an external observation, not invented here: an independent
re-analysis pointed out that published Jev ECE figures of 0.0505–0.0712 at n=60
are equally what serious miscalibration looks like at that sample size. The
right response to that critique is to make our own numbers immune to it.

### Calibration alone certifies a model that ignores its input

The first trained reference model passed every calibration gate: ECE 0.0111
against a 0.05 limit, adaptive ECE 0.0158, reliability bins aligned to within
0.024, and the measured error above its own simulated noise floor, so the
number was a real measurement rather than luck. It also scored **45.6%
accuracy** against a marginal predictor's 39.2% — it had learned the label
frequencies and very little else.

That is not a bug in the metrics. A model reporting each question's true
marginal distribution is calibrated *by construction*; it simply does not use
the state. So a calibration-only release gate certifies the one model
guaranteed to be useless, and does it with a reliability diagram attached —
which is worse than having no gate, because it looks like evidence.

`accuracy_over_baseline` is the fix: the run computes the marginal predictor's
accuracy from the eval labels themselves and requires the model to beat it by
`MIN_ACCURACY_OVER_BASELINE`. It is a floor against the degenerate case, not
an accuracy target; the real accuracy bar is the workflow suite.

The build plan's release gates were ECE-only. They would have passed this
model. Any future gate set has to keep a term that a state-ignoring model
fails.

**What the gate as built does not catch.** It pools every question, so it
rejects a model that ignores the state *everywhere* and passes one that ignores
it almost everywhere. The certified run is the demonstration — per question,
against each question's own marginal predictor:

| Question | Accuracy | Marginal predictor |
| --- | ---: | ---: |
| `plan` (copy a value from the state) | 0.490 | 0.253 |
| `at_risk` (conjunction over two fields) | 0.650 | 0.667 |
| `size` (threshold on a number) | 0.237 | 0.257 |

One question learned, two answered with a near-constant that lands below
chance, and the pooled lift is still +0.0639 — comfortably over the 0.05 limit.
The gate is doing its job, which is to floor the fully degenerate case; it is
just a lower bar than the single pooled number makes it look. Reading it as an
accuracy bar is the mistake it was introduced to prevent, one level up.

The cheap fix when it matters is to gate per question or per slice rather than
on the pooled figure — `slice_reports` already carries the machinery, and the
run above shows the breakdown is worth printing. It is not gated on today
because at a 128-wide two-layer spike every per-question gate would fail and
the signal would be noise about model capacity rather than about the gate.
Phase 1, with a real backbone, is where that stops being true.

Worth noting what temperature scaling did here: almost nothing (0.0112 →
0.0111). That is correct behaviour, not a failure — the model was already
near-calibrated, and a temperature cannot make a model use its input. Post-hoc
calibration fixes the shape of a distribution, never what it is conditioned on.

### The dot-product option head did not learn, and now does

The plan scheduled the Choice crossover — a readout slot per option below 64,
one slot dotted with pooled option states above it — as a phase-1 ablation
between two designs assumed to work. Run in phase 0 instead, the dot-product
arm scored **0.3887 against a 0.3922 marginal predictor**: worse than ignoring
the state, while passing every calibration gate. That made it phase 1's
largest technical risk, because it is the head that makes large option sets
affordable at all — one readout slot at any cardinality — and
`READOUT_BUDGET_TOKENS` and `MAX_QUESTIONS_PER_REQUEST` are sized on the
assumption that it works.

**The mechanism.** Instrumenting the trained checkpoint, the head was not
merely weak, it was disconnected: the query vector was *identical* across 40
different states, to a per-dimension standard deviation of 0.0000. Comparing
the option keys before and after training:

| | mean pairwise cosine | ‖δ‖ / ‖k̄‖ |
| --- | ---: | ---: |
| At initialisation | +0.4715 | 0.7988 |
| After training | +0.9971 | 0.0434 |

Training *collapsed* the option keys onto each other — an 18× reduction in the
component that distinguishes one option from another. The keys start
separable and the model learns to make them identical.

Why: softmax is invariant to the shared component of the keys, so only the
differences δ carry signal. Shrinking δ is a direct, high-gradient move toward
the marginal; learning the state→query→key alignment is a product of two small
quantities, a saddle. Descent takes the cheap route, and once δ ≈ 0 the
gradient with respect to the query is δ/√d ≈ 0 and the query freezes. It is
the degenerate-model failure `accuracy_over_baseline` exists to catch,
occurring *inside* the architecture rather than at the gate.

**Two repairs, measured.** Identical data, seeds and hyperparameters; held-out
accuracy per question against that question's own marginal predictor:

| Arm | `plan` (0.253) | `at_risk` (0.667) | `size` (0.257) | Pooled lift | Gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Readout slot per option | 0.457 | 0.662 | 0.254 | +0.0673 | 5/5 |
| Dot product, as shipped | 0.254 | 0.662 | 0.254 | −0.0002 | blocked |
| … + `match_normalize` | 0.254 | 0.662 | 0.254 | −0.0002 | blocked |
| … + `match_residual` | **0.847** | 0.662 | 0.251 | **+0.1962** | 5/5 |
| … + both | 0.457 | 0.662 | 0.254 | +0.0673 | 5/5 |

`match_residual` adds each option's own input embedding to its key. With it the
dot-product head does not merely recover, it **beats the per-option head by a
wide margin on the question either can learn** — 0.847 against 0.457 — while
costing one readout slot instead of *n*. The default is now on.

**What was expected and was wrong.** The repair predicted to work was
`match_normalize`: CLIP-style cosine similarity with a learnable temperature,
on the reasoning that normalising removes the shrink-δ escape route. It does
nothing whatsoever — 0.254, identical to the unrepaired head to three decimal
places — and combined with the residual it *destroys* two-thirds of the gain.
The shrink-to-marginal story is therefore not the whole mechanism. The
hypothesis that survives is simpler: a one-token option block that attends only
to itself comes out of two layers and a LayerNorm dominated by what every
option shares, so the option's identity never reaches the key at all; the
residual reinjects it, and L2-normalising discards the magnitude that carries
it. That is a hypothesis fitted after the fact to one experiment, and it is
marked as one.

**What this does not establish.** Only `plan` — a copy task, where the answer
appears verbatim in the state — is learned by any arm. `at_risk` (a
conjunction) and `size` (a threshold on a number) sit at their marginals in
every row, which is why the per-question gate fails for all five and is
advisory. The result is one task family, one scale, one seed, at 128-wide and
two layers with a hashing tokenizer; the mechanism it identifies is explicitly
scale-dependent, since a deeper encoder with a real tokenizer may preserve
option identity without help. **Phase 1 re-runs this arm on the real backbone**
— the flag exists so it can be turned off, and the ranking here is not
evidence about the ranking there.

The risk is retired as a blocker and stays open as a measurement.

### A single-seed training run is not evidence

The eval harness refuses to quote an ECE without simulating what a perfectly
calibrated model would score on the same run, and refuses to call a gate a test
if a perfect model fails it half the time. That discipline was applied to the
*measurement*. It was never applied to the *training run*, and it should have
been.

**What happened.** After the tokenizer, batching and dot-product work landed,
the reference configuration stopped reproducing its committed report: 2,500
cases at six epochs closed 3% of the gap to Bayes where the committed run
closed 20%. That looked exactly like a regression one of those changes had
caused, and it was investigated as one — batching was ruled out (the batched
forward is bit-identical to the unbatched one, and the two training
trajectories agree to 1e-6), the tokenizer was ruled out, the option residual
was ruled out.

Then the same configuration was run under four seeds, on one commit, with every
flag identical. Chance is 1.1552 and the committed run's final loss was 1.0366:

| Seed | 1 | 2 | 3 | 4 | 5 | 6 | |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1.1523 | 1.1433 | 1.1411 | 1.1409 | 1.1397 | 1.1401 | never escapes chance |
| 1 | 1.0801 | 0.9916 | 1.0118 | 1.1017 | 1.1053 | 1.0853 | escapes, then **falls back** |
| 2 | 0.9876 | 0.8973 | 0.8871 | 0.8806 | 0.8747 | **0.8706** | escapes and holds |
| 3 | 1.0173 | 0.9608 | 0.9676 | 0.9638 | 0.9572 | 0.9553 | escapes and holds |

There is no regression. There is a configuration whose outcome is decided by
the draw, and a certification that reported one draw as a result — the
committed reference run happened to get a lucky seed under the old arithmetic
and an unlucky one under the new, and the arithmetic changed by about 1e-6.

**The three behaviours are one behaviour.** Seed 0 never leaves the marginal;
seed 1 leaves it and is pulled back; seeds 2 and 3 get away. The marginal
solution is an attractor in this loss landscape, and a run's outcome is which
side of it the draw lands on. That is the same failure as the dot-product
head's, recorded two sections above, where training collapsed the option keys
onto each other until the head could not express a preference — reached by a
different route, into the same basin. Reporting each question's marginal
distribution is not merely a degenerate solution the gates have to exclude; it
is where gradient descent goes when nothing stops it.

It also pays for the best-epoch selection added alongside this: seed 1 ships
epoch 6 at 1.0853 without it, and epoch 2 at 0.9916 with it.

**What this costs.** The reference run's headline — that a trained model
cleared every gate — was one sample from a distribution nobody had measured.
It was not wrong, and it is not evidence either. Every downstream number
inherits that: the per-question breakdown, the dot-product comparison, the
served-answers block. They are all real measurements of one draw.

**What changes.** `scripts/seed_sweep.py` trains a configuration under several
seeds and reports the median and the range rather than the best, and says so
plainly when some seeds certify and others do not. A configuration is a
candidate for certification when its spread is narrow, not when its best seed
is good. Larger runs are the fix as well as the diagnosis: at 8,000 cases the
model learns two questions of three and does it on every seed tried, where
2,500 is on the knife edge.

**The methodological error underneath it** is worth naming, because it is the
one this repository exists to avoid and it still happened here. "A number is
not evidence until the floor is under it" was implemented for ECE and stopped
there. The same sentence applies to accuracy, to the loss curve, and to the
gate verdict itself. Anything reported from a single stochastic run needs its
spread reported with it.

A second error is worth recording too: the first bisect of this was invalid.
Runs of three epochs were compared against a reference of six, and the learning
rate schedule is a function of total steps — so the two had different learning
rates at the epoch being compared. Two rounds of conclusions were drawn from it
before that was noticed.

### Python for the gateway, Rust for one function, Go for nothing

The build plan specifies a Rust production gateway and phase 3 budgets time to
write it. Measured, that is optimising the wrong thing.

**Where a request's time actually goes.** Through the real gateway over HTTP,
lexical backend, single process. Reproduce with `python scripts/gateway_cost.py`,
which prints these numbers and says plainly if the conclusion has stopped
holding:

| | |
| --- | ---: |
| Whole request, end to end | **3.61 ms** (277 req/s per core) |
| …of which the model | 0.16 ms |
| Compiling a 1,000-option schema | 1.9 ms |
| Serialising a 206 KiB response (10,000 options) | 1.4 ms |
| Tokenizing a state at the 65,536-token ceiling | **26 ms** |
| p50 latency target | 150 ms |

The gateway's own work is **2.4% of the p50 budget**. A gateway that cost
literally nothing would move p50 by 3.6 ms. Rewriting it in Rust buys, at
best, three milliseconds of a hundred and fifty — and costs a second
implementation of the contract, which is the failure mode this repository
spends the most effort preventing. `spec/openapi.json`, the drift test and the
generated SDKs all exist because one definition of the contract is worth more
than any constant factor.

It is also worth naming what "Python" means here: `pydantic-core` is Rust and
the JSON encoder is C, so validation and serialisation — normally the
expensive parts of a gateway — are already native. The Python is a thin
orchestration layer over them.

**The cost argument does not rescue Rust either.** At 277 req/s per core, a
core-hour serves about a million requests. Gateway CPU is a rounding error
beside the GPU the model runs on, whatever the model costs. The gateway is not
on the critical path for latency or for spend.

**Rust for exactly one function, when it is measured to matter.** Tokenizing a
maximum-size state is 26 ms, 17% of the p50 budget, and it is the one piece
that is pure CPU with a stable interface. That interface already exists —
`CallableEstimator` and a backend's `.estimator` — so a native tokenizer drops
in without a second contract implementation. It also arrives free: phase 1
swaps in the backbone's own tokenizer, and that will be HuggingFace
`tokenizers`, which is Rust. **So the Rust we need is a dependency, not a
rewrite.** Until a request actually carries a ceiling-size state, 26 ms is a
number about a case nobody has sent.

**Go: no, and it is the clearest of the three.** It does not win on latency,
because nothing does at 2.4%. It has no ML ecosystem, so the calibration math,
the metrics and the eval harness could not live there — and those must stay
importable by training code, which is the actual constraint on this choice.
Its one real advantage is concurrency ergonomics, and that solves a problem
this architecture does not have: the gateway is prefill-only and stateless, so
it scales by process count and the GIL never binds. Choosing Go would trade
the ecosystem that the product's differentiator is written in for a
concurrency model the design does not need.

**What would change our mind.** Three measurements, none of which we have:

1. p99 under real load showing GC or GIL pauses rather than model queueing.
   The gateway's contribution to p99 is unmeasured — 3.61 ms is a median on an
   idle box, and tail behaviour under contention is a different question.
2. A traffic mix where ceiling-size states are common rather than theoretical,
   which would make the 26 ms tokenizer the headline instead of a footnote.
3. Gateway CPU appearing in the bill at all, which at a million requests per
   core-hour would take a traffic scale this project does not have.

If (1) or (2) lands, the response is still not a rewrite: it is moving that
function behind the seam that already exists, which is how the tokenizer is
already structured.

### An index that is stable across requests should be built once

`LexicalShortlister` rebuilt its whole BM25 corpus — tokenizing every option —
on every call. At the 10,000-option scale the retrieval stage exists for, that
cost more than the model does, and it scaled with QPS rather than with the
number of distinct schemas. It is now cached per option set and scored over
postings rather than over every option.

The same principle as the schema KV prefix and the attention mask, which is why
it is worth naming: anything derived only from the schema is derived once. The
bug survived unit tests because a shortlist of 500 options is instant; it only
showed up when the cardinality gate ran the feature at the scale it was built
for.

The gate found a second bug in the same run, this time in itself: the probe
drew option names by rejection sampling from a vocabulary of 2,744
combinations, so asking it for 10,000 distinct names did not terminate. Names
are now enumerated and shuffled, with a numeric discriminator past the
vocabulary — which is also what real taxonomies do once they outgrow their
naming scheme.

Two bugs, both invisible below the scale the feature exists for, both found by
running it at that scale. Load tests are not an optimisation exercise; they are
where this class of defect becomes visible at all, which is an argument for
moving phase 3's load testing earlier rather than treating it as a late
validation step.

### The floor backend is a real baseline, not a mock

`LexicalBackend` is BM25-flavoured token overlap. It makes the whole stack —
gateway, evals, CLI, load tests — runnable on a fresh clone with no weights,
and it is the honest bottom of every Pareto plot. It also earns its keep: it
caught three benchmarks in the jaggedness suite that it could ace by surface
statistics (label-correlated length in literal reading, keyword leakage in
indirection and context rot). A benchmark the floor aces measures nothing.
