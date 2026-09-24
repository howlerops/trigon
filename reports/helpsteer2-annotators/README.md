# HelpSteer2 annotator distributions — the model reproduces how much people disagree

**HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0.
https://huggingface.co/datasets/nvidia/HelpSteer2** — the `disagreements/`
split: every annotator's rating, two to six per pair, 23,652 pairs.

Qwen2.5-1.5B, LoRA rank 16, lr 1e-4, 3 epochs (kept epoch 2 on every seed),
trained with soft cross-entropy on each pair's empirical annotator
distribution. Split by prompt hash, a quarter held out: 16,834 training pairs,
1,000 for the calibrator, 5,818 evaluated. The outcome each prediction is
scored against is **one annotator drawn per pair** -- a model whose
probabilities match annotator disagreement is calibrated against a random
annotator by definition, so the standard gates test that claim directly.
Commit `ead8b5b`, clean tree, its own Modal app; three seeds on `NVIDIA A10`
and one on `A10G`, 3.2–3.5 hours each (`qwen15b-annotators-e3-lr1e-4-modal-run.json`).

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Brier | vs marginal Brier 0.6340 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5496 | +0.0241 | **0.0082** | 0.0119 | 0.5923 | −6.6% |
| 1 | 0.5464 | +0.0209 | 0.0271 | 0.0269 | 0.5982 | −5.6% |
| 2 | 0.5435 | +0.0180 | 0.0083 | 0.0101 | 0.5996 | −5.4% |
| 3 | 0.5459 | +0.0204 | 0.0110 | 0.0134 | 0.5989 | −5.5% |

Per question, lift over each question's own marginal:

| Question | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Annotators' own ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| `complexity` | +0.0648 | +0.0615 | +0.0507 | +0.0565 | +0.0205 |
| `verbosity` | +0.0234 | +0.0179 | +0.0148 | +0.0193 | +0.0079 |
| `helpfulness` | +0.0194 | +0.0136 | +0.0141 | +0.0136 | +0.0596 |
| `correctness` | +0.0129 | +0.0113 | +0.0108 | +0.0131 | +0.0402 |
| `coherence` | −0.0002 | +0.0000 | −0.0005 | −0.0005 | −0.0230 |

The ceiling column is the leave-one-out oracle from `reports/helpsteer2/ceiling.md`:
predicting one annotator from the other annotators of the same response.

## How to read it

**Calibrated against the disagreement itself.** ECE 0.0082–0.0271 against a
random annotator, every value measured against its own noise floor (p95
0.0094–0.0100); seed 0 sits *within* that floor, statistically
indistinguishable from a perfectly calibrated model of how people rate these
responses. The calibrator declined itself on every seed: there was nothing to
fix. This is the claim the product is built on -- probabilities that mean
"how many of your raters would say this" -- measured on real raters.

**At the ceiling on accuracy, which is not the point.** Pooled lift +0.018 to
+0.024 against the annotators' own +0.021. `accuracy_over_baseline` fails on
every seed, because the gate asks for more agreement than the annotators have
with each other.

**The quality questions move on every seed.** `helpfulness` +0.014–0.019 and
`correctness` +0.011–0.013 on all four -- on the averaged labels they were flat
or negative on half the seeds. `complexity` and `verbosity` beat what the other
annotators can predict: the text carries more about a response's length and
difficulty than a second rater's opinion does. `coherence` does not move, and
cannot -- even the oracle is below its marginal.

**Not yet established:** that the soft targets are *why*. No hard-label model
has been trained on these splits, and the averaged-label models evaluated
elsewhere had a different target and overlapping prompts. The ablation --
same data, same splits, majority-vote labels -- is the next run.
