# A public decision benchmark, against the incumbent's own answers

`scripts/decision_bench.py` sends each case of a public 8,016-case
typed-decision benchmark (Leanmcp, 2026, on the Hugging Face Hub; cases under
their sources' licences, predictions and code MIT) through trigon's
compatibility route, and scores it beside the probability vectors the
benchmark publishes for the incumbent (version 1.13.0) and for a compatible
hosted service. Every system is scored on the cases all of them answered, with
the benchmark's conventions: argmax for a Choice, 0.5 for a yes/no, ten-bin
ECE over the top-label probability. SST-5 ships without text and is skipped.

| System | Cases | Macro accuracy | Table |
| --- | ---: | ---: | --- |
| incumbent | 8,016 | **0.838** | `broad.md` |
| **trigon `qwen3-14b-lms-agents` v1 (published, the `trigon-large` tier)** | 8,016 | **0.774** | `lms-agents-14b.md` |
| **trigon `qwen3-4b-lms-agents` v1 (published, the default tier)** | 8,016 | **0.735** | `lms-agents-4b.md` |
| **trigon LM score, Qwen3-14B, safety recipe (published: `qwen3-14b-lms-safe` v1, the `trigon-large` tier)** | 8,016 | **0.756** | `lms-safe-14b.md` |
| **trigon LM score, Qwen3-4B, yes/no fix + safety corpora (published: `qwen3-4b-lms-safe` v1)** -- four slices in-distribution, see below | 8,016 | **0.708** | `lms-safe-s0-4b.md` |
| an open distilled model, published beside it | 8,016 | 0.703 | `lms-yn-4b.md` |
| **trigon LM score, Qwen3-4B, yes/no fix (published: `qwen3-4b-lms-yn` v1)** | 8,016 | **0.637** | `lms-yn-4b.md` |
| trigon LM score, Qwen3-4B | 8,016 | 0.586 | `lms-4b.md` |
| trigon LM score, Qwen3-0.6B, yes/no fix | 8,016 | 0.496 | `lms-yn.md` |
| compatible hosted service | 8,016 | 0.486 | `broad.md` |
| **trigon LM score (Qwen3-0.6B, LoRA + residual, 1 epoch)** | 8,016 | **0.465** | `lms.md` |
| trigon broad (Qwen3-0.6B, readout heads, 4 epochs) | 8,016 | 0.424 | `broad.md` |
| Qwen3-0.6B zero-shot LM score, raw | 660 (60 a slice) | 0.418 | `zero-shot-cf0.md` |
| Qwen3-0.6B zero-shot LM score, contextual calibration | 660 | 0.415 | `zero-shot-cf1.md` |

On the 60-a-slice samples the incumbent scores 0.823 and the hosted service
0.495. They are seeded random samples; the first version of this table used
each file's first 60 cases, and at least one file is ordered by label, which
put the zero-shot rows at 0.439 and 0.429.

## Held out, and not

The `safety` model trains on the *train* splits of the four sources the
benchmark's safety slices test on (jailbreak, prompt injection, Aegis 2.0
prompt and response) -- every text the benchmark shows a model excluded, 43
cases. Those four slices are in-distribution for it the way Banking77 is for
every trigon model; the other seven are held out. Macro accuracy split that way:

| System | 7 held-out slices | 4 safety slices |
| --- | ---: | ---: |
| incumbent | 0.838 | 0.837 |
| an open distilled model | 0.659 | 0.778 |
| **trigon `qwen3-14b-lms-safe` (seed 0)** | **0.668** | **0.910** (in-distribution) |
| **trigon `qwen3-4b-lms-safe` (seed 0)** | **0.602** | **0.892** (in-distribution) |
| trigon `qwen3-4b-lms-yn`, seeds 0 / 1 / 2 | 0.582 / 0.582 / 0.603 | 0.732 / 0.767 / 0.652 |
| hosted service | 0.362 | 0.703 |

Adding the safety corpora cost nothing on the held-out slices (0.602 against
the yes/no model's 0.582–0.603).

## Certification of the yes/no recipe, three seeds

| Seed | Benchmark macro | BoolQ | CLINC150 | Mind2Web step success |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.637 | 0.842 | 0.919 | 0.600 |
| 1 | 0.649 | 0.836 | 0.934 | 0.617 |
| 2 | 0.621 | 0.855 | 0.919 | 0.597 |
| **median** | **0.637** | **0.842** | **0.919** | **0.600** |

## Certification of the `trigon-large` recipe (Qwen3-14B), three seeds

| Seed | Benchmark macro | Held-out slices | BoolQ | CLINC150 | Mind2Web step success |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (published) | 0.756 | 0.668 | 0.892 | 0.942 | 0.673 |
| 1 | 0.746 | 0.657 | 0.892 | 0.948 | 0.685 |
| 2 | 0.750 | 0.668 | 0.899 | 0.943 | 0.703 |
| **median** | **0.750** | **0.668** | **0.892** | **0.943** | **0.685** |

## What it says

**At 4B, with yes/no trained on balanced questions, trigon is second of the
open systems here and ahead of the hosted service by 15 points** (0.637
against 0.486). The yes/no fix -- every Choice corpus also asked as "is the
answer X?", true half the time, and the yes/no loss balanced -- is worth 3
points at 0.6B and 5 at 4B, most of it on the safety slices (jailbreak 0.665
→ 0.892, aegis2 0.586 → 0.754). Jailbreak (0.892), agent-trajectory safety (0.402) and
prompt injection (0.552) are still below the hosted service; knowledge slices are
where the incumbent's lead remains (MMLU-Pro 0.429 against 0.806). The 4B runs
read the benchmark at a later pinned revision that adds two reference systems
and two image slices; trigon reads text only and the image slices are skipped.

**The LM-score readout is the better model on every axis we measure but
yes/no.** One epoch, trained in 2.8 h on a laptop, takes the macro from 0.424
to 0.465 and beats the hosted service on 8 of 11 slices -- every Choice slice,
MMLU-Pro 0.241 against 0.125 -- with ECE at or under the incumbent's on
Banking77, MMLU-Pro and PubMedQA. It loses the macro to the hosted service
(0.486) on three yes/no slices alone: jailbreak 0.660 against 0.930, prompt
injection 0.483 against 0.810, and agent-trajectory safety 0.198 against
0.654 -- below chance, rating 236 of 250 safe trajectories unsafe. The same
model is worst on BoolQ in `reports/generality/lms-q3-06b`. Yes/no is the
open problem; `w` settled at 0.62, so the residual is doing more of the work
than public reports of this design describe (0.95–1.01).

**The old readout heads added nothing over the backbone's own language
model.** Four epochs of the broad mix give 0.424; the same backbone with no
training at all, scoring answers as its own tokens, gives 0.415–0.418. The
heads win only where the mix trained them (Banking77 0.604) and lose to the
zero-shot scorer on everything held out. This is the evidence for moving to
the LM-score readout (`trigon.backends.lm_score`).

**The gap is knowledge, and knowledge is parameters.** The incumbent answers
MMLU-Pro at 0.806 and MedQA at 0.874; every 0.4–0.6B system here, ours and the
hosted encoder, sits at 0.09–0.33 on those slices. No readout recovers facts
a 0.6B backbone does not hold. Closing that half of the gap needs a larger
backbone, which is the case for the 4B run.

**Calibration is not where we lose.** The broad model's ECE is at or under the
incumbent's on six of eleven slices (aegis2 0.023 against 0.029,
aegis2_response 0.007 against 0.025, atbench 0.031 against 0.139); its answers
are less often right, not overconfident.

One seed, one machine (Apple M1 Max, MPS). The 60-a-slice rows are a
screening measurement, not a result.
