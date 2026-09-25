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
a red row indefinitely is how a blocker turns into a footnote.

**CC BY-SA evaluates and never trains** — the owner's decision of 2026-09-25
(`docs/decisions.md` Q17), answering the counsel question BoolQ and FEVER had
been waiting on. A share-alike corpus may be used to evaluate our model; it is
never in a training mix, and nothing fitted on it — weights or calibrators —
ships, so the released weights carry no ShareAlike obligation. It is enforced
on the licence string rather than on the tier: `load(..., purpose="train")`
refuses any corpus whose licence is CC BY-SA whatever tier it was typed with,
a share-alike spec cannot be declared green, and a test holds every row of the
table below to the same rule. BoolQ, FEVER, DBpedia-14 and Circa are the
share-alike rows today.

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
| BoolQ | Noul | CC BY-SA 3.0 | **amber** | plan; share-alike — eval only (Q17) |
| Circa | Choice (8), annotator distributions | **CC BY-SA 4.0** — the repo's licence section names CC BY 4.0 and links the BY-SA 4.0 text; the HF card says cc-by-4.0; the stricter binds | **amber** ⬇ | `google-research-datasets/circa` README at `02ad965`, checked 2026-09-25 — eval only (Q17) |
| FEVER | Choice (3) | CC BY-SA 3.0 | **amber** | plan; share-alike — eval only (Q17) |
| SST-5 | Score | **no licence field on the HF card** | **red** ✓ | HF `SetFit/sst5` carries no licence |
| Amazon Reviews 2023 | Score | **Amazon Customer Reviews Terms of Use** (repo scripts MIT) | **amber** | McAuley Lab card; platform terms are not an open licence |
| ASAP essay scoring | Score + rubric | Kaggle competition terms | **dropped** | not pursued; see sign-off Q18 |
| HelpSteer2/3 | Score | CC BY 4.0 | **green** ✓ | HF `nvidia/HelpSteer2` metadata |
| HelpSteer2-annotators | Score, annotator distributions | CC BY 4.0 | **green** ✓ | HF `nvidia/HelpSteer2` `disagreements/`, same dataset and licence |
| GoEmotions | Noul ×7 (Ekman groups + neutral), annotator distributions | Apache-2.0 | **green** ✓ | HF `google-research-datasets/go_emotions` metadata (the publisher's own org); its licensing section cites the `google-research` repo LICENSE, Apache 2.0. The raw per-rater CSVs sit on a Google bucket with no separate notice. Rechecked 2026-09-25 |
| ChaosNLI | annotator distributions | **no licence field found** | **red** | HF mirror carries no licence |
| Civil Comments | Noul/Score soft labels | CC0-1.0 | **green** ✓ | HF `google/civil_comments` metadata |
| measuring_hate_speech | Score ×10 survey items, annotator distributions | **CC BY 4.0** | **green** ⬆ | HF `ucberkeley-dlab/measuring-hate-speech` metadata at `5468f6e`, rechecked 2026-09-25 |
| HateXplain | Choice (3) + distributions + **human rationales** | **MIT** (repository LICENSE); **CC BY 4.0** (the authors' dataset card) | **green** ✓ | `punyajoy/HateXplain` LICENSE at `01d74227`; HF `Hate-speech-CNERG/hatexplain` metadata; checked 2026-09-25 |
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
| HelpSteer2-annotators | gzipped JSONL, per-annotator lists | ✅ built — the distribution stream |
| GoEmotions | raw per-rater CSV on the authors' bucket | ✅ built — seven Nouls, one row per rater grouped per comment |
| measuring_hate_speech | Parquet only | ✅ built — `scripts/convert_corpus.py` writes gzipped JSONL once |
| Circa | TSV in its repository | ✅ built — **evaluation only**, CC BY-SA |

**The three were not all Parquet-only.** The Hugging Face mirrors are, which
is what the rows above used to say. GoEmotions' authors publish the raw
per-rater CSVs on their own bucket, and Circa's repository holds a TSV with
every annotator's judgement; only measuring_hate_speech needs converting.
Each is pinned — a Hugging Face revision or a commit in the URL, and a
SHA-256 of every file, since a bucket URL carries no revision at all.

| Corpus | Items (train / test) | Raters | Questions | Held out by |
| --- | ---: | --- | --- | --- |
| GoEmotions | 57,877 comments (46,211 / 11,666) | 82 raters; 3–5 per comment, "very unclear" abstentions dropped, at least two kept | 7 Nouls: anger, disgust, fear, joy, sadness, surprise (the authors' Ekman mapping), neutral | a fifth, by hash of the text — 173 texts recur under two ids |
| measuring_hate_speech | 29,488 comments (22,192 / 7,296) | 7,912 annotators; 2 to ~800 per comment, the 10,077 comments rated once dropped | 10 Scores: nine 0–4 survey items and `hatespeech` 0–2, all coded so higher is more hateful | a quarter, by hash of the text |
| Circa | 34,268 question–answer pairs (25,738 / 8,530) | 5 judgements per pair (20 have 4) | 1 Choice over 8 interpretations, `Other` included | a quarter, by hash of the question — each was answered about ten times |
| HateXplain | one JSON file at a pinned commit, SHA-256 checked | ✅ built — the rationale stream |

GoEmotions is seven Nouls rather than a Choice because a rater marks every
emotion that applies, so one rater's answer is not one option; and seven
rather than 28 because most single emotions are marked on under 3% of ratings,
where a gate reads the marginal, and 28 questions quadruple the schema.

**HelpSteer2's main split is not the distribution stream.** It carries
aggregated integer ratings, 0–4, across five attributes. The per-annotator
disagreement data is in its `disagreements/` split and is a separate load.
The row above clears the licence; it does not deliver the thing that makes
annotator-distribution training different from hard-label training.

**`helpsteer2-annotators` does.** 23,652 pairs with every annotator's rating,
two to six each, loaded as a distribution the trainer fits with soft
cross-entropy, plus one annotator drawn per case as the outcome the gates
score. Split by a hash of the prompt, a quarter held out, so no prompt is on
both sides. The modal rating carries only 67–75% of annotators on any
question — the disagreement the averaged split erases.

**`hatexplain` is the one corpus whose annotators said why.** Every annotator
who labelled a post hateful or offensive also marked the tokens that label
rested on; the loader keeps a token when at least half of them marked it, and
those spans are the `Expectation.rationale` the evidence head trains on and
plausibility is scored against (`docs/architecture.md`, *Evidence*). A post
whose majority is `normal` carries no rationale — nobody was asked — and a
three-way split has no majority and is dropped, as the authors drop it.
19,229 posts survive, split by a hash of the post id with a quarter held out;
the authors' own 8:1:1 split is not used, so published HateXplain numbers are
not directly comparable with ours.

Its licence was read from both primary sources on 2026-09-25: the repository
the file is fetched from is MIT, the authors' Hugging Face card says CC BY 4.0,
and both are green. The caveat worth recording is the one ESCI taught: the
posts are Twitter and Gab text, and neither source attaches platform terms to
them. That is the same position as `measuring_hate_speech`, which is also
social-media text and also green; if counsel reads platform terms onto either,
both move together.

Net movement: CLINC150 and measuring_hate_speech clear to **green**, DBpedia-14
resolves to **amber**, AG News separates out as **red** (the two were one row in
the plan and do not share a licence), and **Amazon ESCI drops from green to
amber** — its repository says "this project is licensed under the Apache-2.0
License" and nothing else, which is a code licence being asked to carry data.
That is the §3 policy applied to us rather than to someone else: the rows we
*wanted* to be green get the same reading as the rows we expected to fail.

Six rows the plan assumed were confirmed against the dataset cards rather than
inherited — Banking77, MASSIVE, Circa, HelpSteer2, GoEmotions and Civil
Comments were green as stated, and SST-5's card does indeed carry no licence at
all. Rows still marked *unchecked* are all already amber or red, so verifying
them can only confirm a blocker, never create one.

**Circa did not survive the second reading (2026-09-25).** The first check
read the Hugging Face card, which says cc-by-4.0. The repository's own README
says "Creative Commons Attribution 4.0 License" and, in the next sentence,
gives the BY-SA 4.0 deed as "a full copy of the license". A licence whose two
statements disagree is read as the stricter until its authors say which they
meant, so Circa is recorded as CC BY-SA 4.0 and drops to **amber**: it
evaluates the model and does not train it. The first reading was one source
deep, which is the mistake this table exists to stop.

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
