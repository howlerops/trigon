# Decisions

Answers to the four open questions in the build plan, plus the calls the
implementation forced. Each one names what would change our mind, because a
decision without a falsifier is a preference.

---

## 1. Per-request option/token budget before the retrieval stage

**Decision.** The large-cardinality stage engages when **either** trigger
fires: more than **1,024 options** in one Choice, or a compiled schema over
**12,288 tokens**. The prefilter returns a **256-option shortlist**. The
numbers live in `src/trigon/limits.py`; nothing else may hard-code them.

**Why those numbers.**

Start from the 32,768-token context the plan targets and carve it up:

| Slice | Tokens | Why |
| --- | ---: | --- |
| State | 12,288 | State dominates tokens in document-heavy workloads |
| Schema | 12,288 | The cacheable half; large option sets live here |
| Readout slots | 2,048 | One per question, or one per option below the crossover |
| Headroom | 6,144 | Scaffolding, plus one-sided tokenizer estimation error |

The **count** trigger protects the readout head. With a readout slot per
option, 1,024 options is 1,024 slots — half the readout budget for one
question. Above the crossover the compiler switches to dot-product scoring
(one slot regardless of cardinality), so the count trigger is really about
where per-option slots stop being affordable at all.

The **token** trigger is the one that actually fires in practice, and it is
why a count-only trigger is not enough. A bare option name runs ~4 tokens, so
1,024 names is ~4k tokens and fits. The same 1,024 options *with criteria* run
~20 tokens each — 20k tokens, which does not fit. Option count alone cannot
predict whether a schema fits; tokens can.

**Shortlist size: 256.** Chosen so the post-retrieval path is never narrower
than the 255-option cap the competitor imposes natively. Large-N is a
convenience edge, not a moat, and it would be a strange edge to ship if our
two-stage path scored fewer candidates than their one-stage path.

**What would change our mind.** `recall_at_k` on the cardinality eval sets
(UFET, ~10k types). The prefilter puts a hard ceiling on accuracy that no
amount of model quality recovers, so the gate is recall ≥ **0.99** at the
shortlist size. If a lexical or ANN prefilter cannot hold that at 256, the
shortlist goes up and the latency cost gets published — it does not get
quietly absorbed by lowering the gate.

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
the unmodified Apache-2.0 text.

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

### The floor backend is a real baseline, not a mock

`LexicalBackend` is BM25-flavoured token overlap. It makes the whole stack —
gateway, evals, CLI, load tests — runnable on a fresh clone with no weights,
and it is the honest bottom of every Pareto plot. It also earns its keep: it
caught three benchmarks in the jaggedness suite that it could ace by surface
statistics (label-correlated length in literal reading, keyword leakage in
indirection and context rot). A benchmark the floor aces measures nothing.
