# Decisions

Answers to the four open questions in the build plan, plus the calls the
implementation forced. Each one names what would change our mind, because a
decision without a falsifier is a preference.

---

## 1. Per-request option/token budget before the retrieval stage

**Decision.** The large-cardinality stage engages when **either** trigger
fires: more than **1,024 options** in one Choice, or a single compiled question
over **16,384 tokens**. The prefilter returns a **256-option shortlist**. The
numbers live in `src/trigon/limits.py`; nothing else may hard-code them.

**The envelope was checked, and the plan had it wrong.** The plan states a
"~32k context budget, matching Jev's", and the first version of `limits.py`
carved 32,768 tokens into state, schema and readout slices. The published
contract is not one limit but two: **64k tokens per request** (state plus every
question) and **32k for state plus the longest single question**
(docs.typesafe.ai, verified 2026-09-20). Building to a single flat number got
this wrong in both directions — it halved the total budget, and it missed the
constraint that actually governs a high-cardinality Choice.

The corrected carve-up:

| Slice | Tokens | Why |
| --- | ---: | --- |
| Total request | 65,536 | State plus every question |
| State + longest question | 32,768 | The binding constraint for large option sets |
| State | 16,384 | Dominates tokens in document-heavy workloads |
| Schema | 40,960 | The cacheable half; large option sets grow into it |
| Readout slots | 4,096 | One per question, or one per option below the crossover |
| Headroom | 4,096 | Scaffolding, plus one-sided tokenizer estimation error |

`max_question_tokens` is therefore **derived, not configured**: whatever the
per-question envelope leaves once state has taken its budget. It cannot drift
away from the contract it mirrors.

The **count** trigger protects the readout head. With a readout slot per
option, 1,024 options is 1,024 slots — a quarter of the readout budget for one
question. Above the crossover the compiler switches to dot-product scoring
(one slot regardless of cardinality), so the count trigger is really about
where per-option slots stop being affordable at all.

The **token** trigger is per-question, and it is why a count-only trigger is
not enough. A bare option name runs ~4 tokens, so 1,024 names is ~4k tokens and
fits comfortably. The same 1,024 options *with criteria* run ~20 tokens each —
20k tokens against a 16,384-token per-question budget, which does not. Option
count alone cannot predict whether a question is admissible; tokens can.

**Shortlist size: 256.** Chosen so the post-retrieval path is never narrower
than the 255-option cap the competitor imposes natively. Large-N is a
convenience edge, not a moat, and it would be a strange edge to ship if our
two-stage path scored fewer candidates than their one-stage path.

**What would change our mind, and what happened when we ran it.**
`recall_at_k` at the shortlist size, gated at **0.99**, because everything
below the prefilter's recall is accuracy no model quality recovers.

That falsifier is implemented and runs: `trigon.evals.cardinality`. It could
not be run on the corpus the plan named. UFET has no stated licence and its
distant-supervision half derives from LDC-licensed Gigaword (`docs/data.md`),
and no permissively-licensed corpus exists anywhere near the 1,024-option
regime the trigger fires in — the largest green set in the audit is CLINC150 at
151 classes. So the probe generates confusable option sets at 256 to 10,000
options, every true option surrounded by near-neighbours sharing most of its
words. For a recall measurement that is arguably better than a found corpus:
distractor similarity becomes a dial instead of whatever the data happened to
contain.

Results are in `docs/evals.md`. If the gate cannot be held at 256, the
shortlist goes up and the latency cost gets published — it does not get quietly
absorbed by lowering the gate.

---

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
| **Green** | Apache-2.0, MIT, CC0, CC BY 4.0, ODC-BY | Training, eval, redistribution |
| **Amber** | CC BY-SA, NC clauses, "research use" terms | Eval only. Never in a training mix, never redistributed |
| **Red** | Competition-only terms, unclear provenance, no stated licence | Prototyping on a local copy only; blocked from any shipped artifact |

Concrete consequences for the seed list in the plan: Banking77, MASSIVE,
Circa, HelpSteer2/3 (CC BY 4.0), GoEmotions, Amazon ESCI (Apache-2.0) and
Civil Comments (CC0) are green. BoolQ and FEVER (CC BY-SA) are amber — the
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

Two of those change the plan rather than confirming it, and both are written up
in `docs/data.md`: the cardinality stress test has no licensed corpus, and
every resolved-outcome corpus is blocked.

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
for. Benchmarks at production scale find things correctness tests do not.

### The floor backend is a real baseline, not a mock

`LexicalBackend` is BM25-flavoured token overlap. It makes the whole stack —
gateway, evals, CLI, load tests — runnable on a fresh clone with no weights,
and it is the honest bottom of every Pareto plot. It also earns its keep: it
caught three benchmarks in the jaggedness suite that it could ace by surface
statistics (label-correlated length in literal reading, keyword leakage in
indirection and context rot). A benchmark the floor aces measures nothing.
