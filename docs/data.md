# Data plan and licence audit

Five streams. Two buy intelligence, three buy calibration, and the build plan
is explicit that conflating them is the mistake to avoid: distillation from a
teacher ensemble buys accuracy and **not** calibration, because frontier models
are themselves overconfident and their probabilities are not calibration
targets.

| Stream | Content | Labels | Buys |
| --- | --- | --- | --- |
| Public labelled corpora | classification, NLI, sentiment, ordinal rating, span selection, reformatted into the three primitives | ground truth | outcome calibration |
| Annotator distributions | subjective judgements with many human labels per item | label distributions | calibration where no outcome exists |
| Synthetic workflows | LLM-generated (state, schema) pairs across ~20 domains | teacher soft labels | coverage |
| Verifiable synthetic | generated structured state with code-checkable predicates | computed ground truth | cheap outcome data at scale |
| Adversarial + paired | injections, distractor padding, negation pairs, paraphrases | held fixed / derived | robustness and consistency losses |

The verifiable synthetic stream is **implemented** and runnable today:
`trigon.evals.synthetic_outcome_cases`. Its limits are stated in the module —
clean structured state, decidable predicates, nothing subjective. It exercises
the calibration machinery end to end. It does not stand in for the other four.

Its `noise` parameter is load-bearing. A zero-noise set is perfectly
predictable, which makes it useless for calibration: a model can be right every
time and any confidence below 1.0 reads as miscalibrated. Non-zero noise
creates genuine aleatoric uncertainty, so the correct behaviour is a
probability near `1 - noise` and the suite can tell a calibrated model from a
merely confident one.

**Subjective questions need the annotator-distribution stream.** A Score like
"how frustrated is this customer" has no resolvable outcome; its calibration
target is the human annotator distribution, and training against majority votes
throws that away. Without this stream, calibration claims cover objective
questions only — and saying so is part of publishing them honestly.

**Schema diversity matters as much as state diversity.** Vary option counts
(2 → hundreds), criteria verbosity, JSON-structured vs. prose instructions, and
questions per request (1 → 20+). A model trained only on three-option questions
will have a calibration cliff at twenty.

## Licence audit

Policy and the licence-to-tier mapping live in `docs/decisions.md` §3 and are
not restated here — the two copies had already drifted apart twice. What the
tiers permit: **green** trains, evals and redistributes; **amber** evals only,
never a training mix and never redistributed; **red** is a local copy for
prototyping and is blocked from every shipped artifact.

A permissive code licence does not launder a research-only corpus. A dataset
enters a training mix only by moving to green. "Verify" is a blocker for
training, not a footnote.

**Four corpora are dropped rather than red.** At sign-off (2026-09-20) the
Kaggle entries (ASAP, Home Credit, IEEE-CIS) and LMSYS were dropped outright
rather than left as blockers awaiting a licence answer: none is load-bearing
now that outcome grounding comes from verifiable synthetic data, and carrying
a red row indefinitely is how a blocker turns into a footnote. BoolQ and FEVER
stay amber pending a counsel opinion on CC BY-SA for a derived model — that one
is worth answering because it recurs for every share-alike corpus.

**Checked against primary sources on 2026-09-20.** Each row below records what
the dataset's own card or repository states, not what the build plan assumed.
Rows marked *unchecked* keep the plan's assumption and are still blockers.

| Dataset | Primitive | Licence found | Tier | Evidence |
| --- | --- | --- | --- | --- |
| Banking77 | Choice (77) | CC BY 4.0 | **green** ✓ | HF `PolyAI/banking77` metadata |
| CLINC150 | Choice (151) | **CC BY 3.0** | **green** ⬆ | HF `clinc/clinc_oos` metadata |
| MASSIVE | Choice (60) | CC BY 4.0 | **green** ✓ | HF `AmazonScience/massive` metadata |
| AG News | Choice (4) | **licence "unknown" on the dataset card** | **red** | HF `fancyzhx/ag_news` |
| DBpedia-14 | Choice (14) | **CC BY-SA 3.0 + GFDL** | **amber** ⬆ | HF `fancyzhx/dbpedia_14` |
| BoolQ | Noul | CC BY-SA 3.0 | **amber** | plan; share-alike |
| Circa | Noul | CC BY 4.0 | **green** ✓ | HF `google-research-datasets/circa` metadata |
| FEVER | Choice (3) | CC BY-SA 3.0 | **amber** | plan; share-alike |
| SST-5 | Score | **no licence field on the HF card** | **red** ✓ | HF `SetFit/sst5` carries no licence |
| Amazon Reviews 2023 | Score | **Amazon Customer Reviews Terms of Use** (repo scripts MIT) | **amber** | McAuley Lab card; platform terms are not an open licence |
| ASAP essay scoring | Score + rubric | Kaggle competition terms | **dropped** | not pursued; see sign-off Q18 |
| HelpSteer2/3 | Score | CC BY 4.0 | **green** ✓ | HF `nvidia/HelpSteer2` metadata |
| GoEmotions | Choice (27) + distributions | Apache-2.0 | **green** ✓ | HF `google-research-datasets/go_emotions` metadata |
| ChaosNLI | annotator distributions | **no licence field found** | **red** | HF mirror carries no licence |
| Civil Comments | Noul/Score soft labels | CC0-1.0 | **green** ✓ | HF `google/civil_comments` metadata |
| measuring_hate_speech | Score distributions | **CC BY 4.0** | **green** ⬆ | HF `ucberkeley-dlab/measuring-hate-speech` |
| deepset prompt-injection; WildGuardMix; ToxicChat | Noul guardrails | not resolved | **amber** | cards not conclusive |
| UFET | Choice (~10k types) | **no stated licence; distant-supervision half derives from LDC-licensed Gigaword** | **red** | UT Austin dataset page; `uwnlp/open_type` |
| Amazon ESCI | Choice (4) | **repo licensed Apache-2.0 as a "project"; no data-specific grant** | **amber** ⬇ | `amazon-science/esci-data` LICENSE + README |
| WDC Products; Magellan | Noul | "public research", terms unstated | **amber** | unchecked |
| LMSYS Arena preferences | Choice (A/B/tie) | **custom LMSYS-Chat-1M Dataset License Agreement, gated access** | **dropped** | not pursued; see sign-off Q18 |
| Home Credit; IEEE-CIS fraud | Noul on structured state | Kaggle competition terms | **dropped** | not pursued; see sign-off Q18 |
| Autocast | Noul/Choice | **code MIT; dataset hosted "with permission from Metaculus for research purposes only"** | **red** | `andyzoujm/autocast` |

### Green is a licence, not a format

Cleared and *loadable* are different facts, and the second one cost a check
worth recording. `trigon.evals.corpora` reads plain files with the standard
library only, because it is imported by the gateway's budgeting path and by
the drift tests, neither of which has a GPU stack — a Parquet reader here puts
`pyarrow` in all of them.

| Corpus | Format | Loadable today |
| --- | --- | --- |
| Banking77 | CSV over HTTP | ✅ built |
| HelpSteer2 | gzipped JSONL | ✅ built |
| GoEmotions | Parquet only | needs a conversion step in `scripts/` |
| measuring_hate_speech | Parquet only | needs a conversion step in `scripts/` |
| Circa | Parquet only | needs a conversion step in `scripts/` |

**HelpSteer2's main split is not the distribution stream.** It carries
aggregated integer ratings, 0–4, across five attributes. The per-annotator
disagreement data is in its `disagreements/` split and is a separate load.
The row above clears the licence; it does not deliver the thing that makes
annotator-distribution training different from hard-label training.

Net movement: CLINC150 and measuring_hate_speech clear to **green**, DBpedia-14
resolves to **amber**, AG News separates out as **red** (the two were one row in
the plan and do not share a licence), and **Amazon ESCI drops from green to
amber** — its repository says "this project is licensed under the Apache-2.0
License" and nothing else, which is a code licence being asked to carry data.
That is the §3 policy applied to us rather than to someone else: the rows we
*wanted* to be green get the same reading as the rows we expected to fail.

Six rows the plan assumed were confirmed against the dataset cards rather than
inherited — Banking77, MASSIVE, Circa, HelpSteer2, GoEmotions and Civil
Comments are green as stated, and SST-5's card does indeed carry no licence at
all. Rows still marked *unchecked* are all already amber or red, so verifying
them can only confirm a blocker, never create one.

### The three consequences that change the plan

**The cardinality stress test has no licensed corpus.** UFET was the plan's
~10k-type set and it is unusable: no stated licence, and its distant-supervision
half is derived from Gigaword, which is LDC-licensed. The largest green corpus
in this table is CLINC150 at 151 classes — well below the 1,024-option trigger
the retrieval stage exists for. There is therefore no public, redistributable
corpus in the regime the feature is built for.

The answer is not to ship without the gate. `trigon.evals.cardinality`
generates confusable option sets at 256 to 10,000 options — every true option
surrounded by near-neighbours sharing most of its words — and runs D1's
falsifier against them. For a *recall* measurement a generated set is arguably
better than a found one, because distractor similarity becomes a dial rather
than whatever the corpus happened to contain. CLINC150 remains the real-data
check at 151 classes.

**The green pool lost its e-commerce domain.** Amber means eval only, so ESCI
leaves the training mix and the green Choice sources are all conversational or
editorial — Banking77, CLINC150, MASSIVE, GoEmotions. Nothing in the trainable
set looks like product search. That is a coverage gap to state in the model
card rather than a blocker: the eval tier still holds ESCI, so the gap is
measurable even though it cannot be trained on.

**Every resolved-outcome corpus is blocked.** Home Credit, IEEE-CIS and
Autocast were the three sources of real, resolved outcomes on this list, and all
three are red — Autocast decisively so, since Metaculus granted research-only
hosting rather than a licence. Outcome-grounded calibration in v1 therefore
rests on verifiable synthetic data and on the ground-truth labels of the green
classification corpora, not on prediction-with-resolution data. That is a
narrower claim than the build plan assumes, and the calibration report has to
say so rather than implying coverage it does not have.

## Teacher labels

Keep every teacher call's full distribution — logprobs where the API exposes
them, k-sample voting otherwise. A teacher call that returns only an argmax has
thrown away most of what it cost, and re-running it later costs the same again.

Budget from the plan: $20–50k of teacher-label compute, a few thousand for
generation.
