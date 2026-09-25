# goemotions

**GoEmotions (Demszky et al., 2020), Google Research. Apache-2.0. https://github.com/google-research/google-research/tree/master/goemotions**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 16,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 11,666 |
| — from the corpus's own test split | 11,666 |
| — held out of train to reach the floor | 0 |
| Questions per request | 7 |
| Labels per question | 0 |

| Question | Marginal predictor |
| --- | ---: |
| `anger` | 0.8608 |
| `disgust` | 0.9756 |
| `fear` | 0.9799 |
| `joy` | 0.5967 |
| `neutral` | 0.7207 |
| `sadness` | 0.9177 |
| `surprise` | 0.8598 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.2290 | 0.3676 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py goemotions -n 16000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+863befba

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| goemotions/uncalibrated | 11666 | 0.8880 | 0.0095 | 0.0095 | 0.1614 | 49.6 | 55.8 | 155 |
| goemotions/calibrated | 11666 | 0.8880 | 0.0048 | 0.0051 | 0.1612 | 50.0 | 56.1 | 155 |

## Release gates

- PASS sample_size: 81662.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0033 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0023 on this run, p95 0.0033
- FAIL (advisory) accuracy_over_baseline: 0.0435 (limit 0.0500) -- model 0.8880 vs marginal predictor 0.8445; calibration cannot reject a model that ignores the state; advisory: a drawn annotator caps every predictor
- PASS brier_over_marginal: 0.2959 (limit 0.0200) -- Brier 0.1612 vs the training marginal's 0.2290; the term that fails a model ignoring its input, on a drawn-annotator corpus
- FAIL (advisory) worst_question_over_baseline: 0.0021 (limit 0.0500) -- worst is 'disgust' at 0.9776 vs its own marginal 0.9756; the pooled gate hides this
- PASS workhorse_ece: 0.0048 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0051 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0048 (limit 0.0500) -- worst is 'noul' at ECE 0.0048 (overconfidence +0.005); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

**Advisory, not blocking**: accuracy_over_baseline, worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `goemotions/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `disgust` | 11,666 | 0.9776 | 0.9756 | +0.0021 |
| `fear` | 11,666 | 0.9829 | 0.9799 | +0.0030 |
| `anger` | 11,666 | 0.8709 | 0.8608 | +0.0101 |
| `sadness` | 11,666 | 0.9325 | 0.9177 | +0.0148 |
| `surprise` | 11,666 | 0.8788 | 0.8598 | +0.0189 |
| `neutral` | 11,666 | 0.7608 | 0.7207 | +0.0401 |
| `joy` | 11,666 | 0.8122 | 0.5967 | +0.2155 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| goemotions | 81662 | 0.888 | 0.893 | +0.005 | 0.0048 | 0.0051 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noul | 81662 | 0.888 | 0.893 | +0.005 | 0.0048 | 0.0051 |

## Is this number evidence?

On these 81662 predictions a perfectly calibrated model scores a mean ECE of 0.0022 (95th percentile 0.0031), simulated over 200 resamples. The measured ECE is 0.0095.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.47,0.53)  1517  0.517  0.510  +0.007  ####################.|..................
  [0.53,0.60)  3243  0.567  0.562  +0.005  ######################.|................
  [0.60,0.67)  3429  0.634  0.619  +0.015  #########################|..............
  [0.67,0.73)  3765  0.701  0.675  +0.025  ###########################.|...........
  [0.73,0.80)  4550  0.768  0.739  +0.028  ##############################.|........
  [0.80,0.87)  5979  0.835  0.813  +0.022  #################################|......
  [0.87,0.93) 10134  0.904  0.891  +0.013  ####################################|...
  [0.93,1.00) 49045  0.983  0.979  +0.004  #######################################|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
