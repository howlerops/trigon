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
| compatible hosted service | 8,016 | 0.486 | `broad.md` |
| trigon broad (Qwen3-0.6B, readout heads, 4 epochs) | 8,016 | 0.424 | `broad.md` |
| Qwen3-0.6B zero-shot LM score, raw | 660 (60 a slice) | 0.439 | `zero-shot-cf0.md` |
| Qwen3-0.6B zero-shot LM score, contextual calibration | 660 | 0.429 | `zero-shot-cf1.md` |

On the 60-a-slice subset the incumbent scores 0.808 and the hosted service 0.477.

## What it says

**The trained readout heads add nothing over the backbone's own language
model here.** Four epochs of the broad mix give 0.424; the same backbone with
no training at all, scoring answers as its own tokens, gives 0.43–0.44. The
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
