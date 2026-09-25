# GoEmotions: certified against how raters disagree, and carried by one emotion

**GoEmotions (Demszky et al., 2020), Google Research. Apache-2.0.
https://github.com/google-research/google-research/tree/master/goemotions.**
The raw per-rater CSVs: 3–5 raters per Reddit comment. Each comment is asked
seven Noul questions, one per Ekman group (anger, disgust, fear, joy, sadness,
surprise) plus neutral. A case's outcome is **one rater drawn per comment**,
so these are scored under `brier_over_marginal` (Q20).

Qwen2.5-1.5B with LoRA rank 16, lr 1e-4, 3 epochs, soft targets. 16,000
training comments, 1,000 for the calibrator and 11,666 evaluated, held out by
a hash of the text. Commit `cd91859`, clean tree. Three seeds ran on
`NVIDIA A10` and one on `A10G`, 62–75 minutes each
(`qwen15b-n16000-e3-lr1e-4-modal-run.json`).

| Seed | Accuracy | Lift | **Brier skill** | Brier | ECE | Adaptive ECE | Floor p95 | Raw ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8872 | +0.0428 | **+0.2857** | 0.1636 | 0.0035 | 0.0046 | 0.0032 | 0.0035 |
| 1 | 0.8880 | +0.0435 | **+0.2959** | 0.1612 | 0.0048 | 0.0051 | 0.0033 | 0.0095 |
| 2 | 0.8880 | +0.0436 | **+0.2945** | 0.1615 | 0.0076 | 0.0075 | 0.0029 | 0.0098 |
| 3 | 0.8888 | +0.0443 | **+0.3010** | 0.1601 | 0.0034 | 0.0034 | 0.0029 | 0.0034 |
| median | 0.8880 | +0.0436 | **+0.2952** | | 0.0042 | | | |

The marginal distribution of the training split scores Brier 0.2290 and NLL
0.3677 on the same outcomes.

**It certifies on four seeds of four.** Every blocking gate passes on every
seed. `brier_over_marginal` clears its +0.02 limit by more than a factor of
ten, and the four seeds sit within 0.016 of each other. ECE is at its noise
floor: on seeds 0 and 3 it is within 0.0005 of what a perfectly calibrated
model scores on this evaluation set. The calibrator had almost nothing to fix.

**Per question, lift over each question's own marginal:**

| Question | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: | ---: |
| `joy` | +0.2133 | +0.2155 | +0.2160 | +0.2188 |
| `neutral` | +0.0401 | +0.0401 | +0.0423 | +0.0401 |
| `surprise` | +0.0155 | +0.0189 | +0.0159 | +0.0189 |
| `sadness` | +0.0124 | +0.0148 | +0.0175 | +0.0162 |
| `anger` | +0.0137 | +0.0101 | +0.0071 | +0.0108 |
| `fear` | +0.0024 | +0.0030 | +0.0039 | +0.0031 |
| `disgust` | +0.0020 | +0.0021 | +0.0021 | +0.0022 |

**The pooled gate hides the same thing it hid on HelpSteer2.** `joy` carries
most of the lift. `fear` and `disgust` are marked on a few percent of ratings
and sit barely off their marginals on every seed. A caller asking this model
about disgust gets a calibrated rate close to the base rate, and a pooled
Brier skill of +0.295 does not say so. Read the per-question table, not the
headline, before deploying one of the rare emotions.
