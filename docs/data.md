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

Policy from `docs/decisions.md` §3. A permissive code licence does not launder
a research-only corpus.

| Tier | Licences | May be used for |
| --- | --- | --- |
| **Green** | Apache-2.0, MIT, CC0, CC BY 4.0, ODC-BY | training, eval, redistribution |
| **Amber** | CC BY-SA, NC clauses, "research use" | eval only — never a training mix, never redistributed |
| **Red** | competition-only, unclear provenance, no stated licence | local prototyping only; blocked from every shipped artifact |

A dataset enters a training mix only by moving to green in this table.
"Verify" is a blocker for training, not a footnote.

| Dataset | Primitive | Why | Licence as stated in the plan | Tier |
| --- | --- | --- | --- | --- |
| Banking77 | Choice (77) | high-cardinality intent routing | CC BY 4.0 | green |
| CLINC150 | Choice (151) | out-of-scope option trains abstention | verify | red |
| MASSIVE | Choice (60) | multilingual intents | CC BY 4.0 | green |
| AG News, DBpedia-14 | Choice | easy topical baselines | verify | red |
| BoolQ | Noul | naturalistic yes/no over passages | CC BY-SA 3.0 | amber |
| Circa | Noul | indirect yes/no answers | CC BY 4.0 | green |
| FEVER | Choice (3) | claim verification for guardrails | CC BY-SA, verify | amber |
| SST-5; Amazon Reviews 2023 | Score | ordinal ratings at scale | research-use, verify | amber |
| ASAP essay scoring | Score + rubric | human-rated rubric scoring | competition terms | red |
| HelpSteer2/3 | Score | 0–4 Likert, 5 attributes, multi-annotator | CC BY 4.0 | green |
| GoEmotions | Choice (27) + distributions | rater-level labels; the ambiguity cliff | Apache-2.0 | green |
| ChaosNLI | annotator distributions | 100 annotations per item | verify | red |
| Civil Comments | Noul/Score soft labels | toxicity as annotator fraction | CC0 | green |
| measuring_hate_speech | Score distributions | continuous score, rater-level | verify | red |
| deepset prompt-injection; WildGuardMix; ToxicChat | Noul guardrails | injection and moderation | Apache / ODC-BY, verify | amber |
| UFET | Choice (~10k types) | cardinality stress test | verify | red |
| Amazon ESCI | Choice (4) | search-relevance grading | Apache-2.0 | green |
| WDC Products; Magellan | Noul | entity matching | public research | amber |
| LMSYS Arena preferences | Choice (A/B/tie) | LLM-routing use case | verify | red |
| Home Credit; IEEE-CIS fraud | Noul on structured state | real resolved outcomes on JSON state | competition-only | red |
| Autocast | Noul/Choice | resolved forecasts, calibration-native | verify | red |

Licences in this table are **as stated in the build plan** and are not
independently verified. Verifying them against each dataset's current terms is
a phase-0 task, and no red or amber entry reaches a training mix before it is
done. Prefer permissive Hugging Face mirrors over competition pages, and record
the licence text and retrieval date alongside each corpus.

Two consequences worth stating up front. UFET is the cardinality stress test
and it is red — the large-N story needs a green substitute or the recall gate
gets measured on a corpus we cannot ship. Home Credit, IEEE-CIS and Autocast
are the only real *resolved outcome* corpora on the list, and all three are
red; if none clears, the outcome-calibration stream leans harder on verifiable
synthetic data than the plan assumes, and the calibration claims have to say so.

## Teacher labels

Keep every teacher call's full distribution — logprobs where the API exposes
them, k-sample voting otherwise. A teacher call that returns only an argmax has
thrown away most of what it cost, and re-running it later costs the same again.

Budget from the plan: $20–50k of teacher-label compute, a few thousand for
generation.
