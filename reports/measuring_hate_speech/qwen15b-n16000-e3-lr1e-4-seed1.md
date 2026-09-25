# measuring_hate_speech

**Measuring Hate Speech (Kennedy et al., 2020; Sachdeva et al., 2022), UC Berkeley D-Lab. CC BY 4.0. https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 16,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 7,296 |
| — from the corpus's own test split | 7,296 |
| — held out of train to reach the floor | 0 |
| Questions per request | 10 |
| Labels per question | 5 |

| Question | Marginal predictor |
| --- | ---: |
| `attack_defend` | 0.3712 |
| `dehumanize` | 0.2470 |
| `genocide` | 0.7272 |
| `hatespeech` | 0.6623 |
| `humiliate` | 0.2961 |
| `insult` | 0.3320 |
| `respect` | 0.3527 |
| `sentiment` | 0.3751 |
| `status` | 0.5140 |
| `violence` | 0.5476 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6702 | 1.2994 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py measuring_hate_speech -n 16000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+6d8438e9

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| measuring_hate_speech/uncalibrated | 7296 | 0.5580 | 0.0360 | 0.0360 | 0.5560 | 58.9 | 64.9 | 355 |
| measuring_hate_speech/calibrated | 7296 | 0.5580 | 0.0101 | 0.0109 | 0.5540 | 58.6 | 65.4 | 355 |

## Release gates

- PASS sample_size: 72960.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0060 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0042 on this run, p95 0.0060
- PASS (advisory) accuracy_over_baseline: 0.1155 (limit 0.0500) -- model 0.5580 vs marginal predictor 0.4425; calibration cannot reject a model that ignores the state; advisory: a drawn annotator caps every predictor
- PASS brier_over_marginal: 0.1735 (limit 0.0200) -- Brier 0.5540 vs the training marginal's 0.6702; the term that fails a model ignoring its input, on a drawn-annotator corpus
- FAIL (advisory) worst_question_over_baseline: 0.0106 (limit 0.0500) -- worst is 'genocide' at 0.7378 vs its own marginal 0.7272; the pooled gate hides this
- PASS workhorse_ece: 0.0101 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0109 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0101 (limit 0.0500) -- worst is 'score' at ECE 0.0101 (overconfidence -0.001); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `measuring_hate_speech/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `genocide` | 7,296 | 0.7378 | 0.7272 | +0.0106 |
| `violence` | 7,296 | 0.5776 | 0.5476 | +0.0300 |
| `status` | 7,296 | 0.5821 | 0.5140 | +0.0681 |
| `hatespeech` | 7,296 | 0.7645 | 0.6623 | +0.1022 |
| `dehumanize` | 7,296 | 0.3740 | 0.2470 | +0.1271 |
| `humiliate` | 7,296 | 0.4294 | 0.2961 | +0.1334 |
| `insult` | 7,296 | 0.4846 | 0.3320 | +0.1527 |
| `attack_defend` | 7,296 | 0.5281 | 0.3712 | +0.1569 |
| `sentiment` | 7,296 | 0.5562 | 0.3751 | +0.1811 |
| `respect` | 7,296 | 0.5454 | 0.3527 | +0.1927 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| measuring_hate_speech | 72960 | 0.558 | 0.557 | -0.001 | 0.0101 | 0.0109 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 72960 | 0.558 | 0.557 | -0.001 | 0.0101 | 0.0109 |

## Is this number evidence?

On these 72960 predictions a perfectly calibrated model scores a mean ECE of 0.0042 (95th percentile 0.0060), simulated over 200 resamples. The measured ECE is 0.0360.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   266  0.251  0.226  +0.026  #########.|.............................
  [0.27,0.33)  3861  0.308  0.283  +0.025  ###########.|...........................
  [0.33,0.40)  7856  0.367  0.334  +0.033  #############..|........................
  [0.40,0.47)  9385  0.436  0.402  +0.034  ################.|......................
  [0.47,0.53) 11073  0.499  0.464  +0.035  ###################.|...................
  [0.53,0.60)  8819  0.565  0.519  +0.046  #####################..|................
  [0.60,0.67)  6792  0.632  0.591  +0.041  ########################.|..............
  [0.67,0.73)  5957  0.699  0.637  +0.062  #########################...|...........
  [0.73,0.80)  5495  0.767  0.739  +0.028  ##############################.|........
  [0.80,0.87)  5684  0.833  0.801  +0.033  ################################.|......
  [0.87,0.93)  4791  0.898  0.873  +0.026  ###################################.|...
  [0.93,1.00)  2981  0.966  0.955  +0.011  ######################################.|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
