# measuring_hate_speech: ten survey scales, certified against their annotators

**Measuring Hate Speech (Kennedy et al., 2020; Sachdeva et al., 2022), UC
Berkeley D-Lab. CC BY 4.0.
https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech.**
Converted once from Parquet by `scripts/convert_corpus.py`, inside the Modal
container. Ten Score questions per comment: nine 0–4 survey items and
`hatespeech` on 0–2, all coded so that higher is more hateful. Each case's
outcome is **one annotator drawn per comment**, from 2 to about 800 per
comment, so the corpus is gated on `brier_over_marginal` (Q20).

Qwen2.5-1.5B with LoRA rank 16, lr 1e-4, 3 epochs, soft targets. 16,000
training comments, 1,000 for the calibrator and 7,296 evaluated, held out by a
hash of the text. Commit `cd91859`, clean tree. Three seeds ran on
`NVIDIA A10` and one on `A10G`, 1.8–2.2 hours each
(`qwen15b-n16000-e3-lr1e-4-modal-run.json`).

| Seed | Accuracy | Lift | **Brier skill** | Brier | ECE | Adaptive ECE | Floor p95 | Raw ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5614 | +0.1189 | **+0.1759** | 0.5523 | 0.0160 | 0.0160 | 0.0062 | 0.0160 |
| 1 | 0.5580 | +0.1155 | **+0.1735** | 0.5540 | 0.0101 | 0.0109 | 0.0060 | 0.0360 |
| 2 | 0.5577 | +0.1152 | **+0.1709** | 0.5557 | 0.0332 | 0.0332 | 0.0064 | 0.0332 |
| 3 | 0.5609 | +0.1184 | **+0.1778** | 0.5511 | 0.0059 | 0.0081 | 0.0062 | 0.0150 |
| median | 0.5595 | +0.1170 | **+0.1747** | | 0.0131 | | | |

The marginal distribution of the training split scores Brier 0.6702 and NLL
1.2994 on the same outcomes.

**It certifies on four seeds of four,** with a narrow spread: Brier skill
varies by 0.007 between seeds. Unlike HelpSteer2, the lift is large on
nearly every question. The seed-to-seed variation is in calibration. Seed 2
shipped raw at ECE 0.0332, because the calibrator declined itself, and
passes. Seed 3 is at the noise floor.

**Per question, lift over each question's own marginal:**

| Question | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: | ---: |
| `respect` | +0.1939 | +0.1927 | +0.1927 | +0.1942 |
| `sentiment` | +0.1896 | +0.1811 | +0.1842 | +0.1919 |
| `attack_defend` | +0.1693 | +0.1569 | +0.1643 | +0.1702 |
| `insult` | +0.1465 | +0.1527 | +0.1391 | +0.1528 |
| `humiliate` | +0.1398 | +0.1334 | +0.1391 | +0.1319 |
| `dehumanize` | +0.1340 | +0.1271 | +0.1218 | +0.1286 |
| `hatespeech` | +0.1007 | +0.1022 | +0.1029 | +0.1054 |
| `status` | +0.0667 | +0.0681 | +0.0673 | +0.0622 |
| `violence` | +0.0382 | +0.0300 | +0.0329 | +0.0373 |
| `genocide` | +0.0106 | +0.0106 | +0.0077 | +0.0093 |

**The weak end is the severe end.** `genocide` and `violence` are rated at
their lowest point on nearly every comment, so their marginals are hard to
beat, and the model barely does. These are the two items a moderation
deployment would care about most. The pooled +0.175 does not speak for them.
