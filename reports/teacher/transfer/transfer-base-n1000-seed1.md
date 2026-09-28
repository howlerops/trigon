# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 1,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 3,080 |
| — held out of train to reach the floor | 1,920 |
| Questions per request | 1 |
| Labels per question | 77 |

| Question | Marginal predictor |
| --- | ---: |
| `intent` | 0.0150 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.9882 | 4.3958 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 4 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py banking77 -n 1000 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

`MIN_CALIBRATION_SAMPLES` is 5,000 and this
corpus's test split is smaller, so the evaluation set is topped up
from rows held out of train that neither training nor calibration
saw. They are held-out data by the only definition that matters and
they come from the train distribution, which is why the split is
reported above rather than summed into one number.

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+bf0b91da

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.6976 | 0.1298 | 0.1299 | 0.4415 | 63.7 | 69.0 | 298 |
| banking77/calibrated | 5000 | 0.6942 | 0.0498 | 0.0504 | 0.4318 | 63.2 | 67.2 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0197 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0143 on this run, p95 0.0197
- PASS accuracy_over_baseline: 0.6776 (limit 0.0500) -- model 0.6942 vs marginal predictor 0.0166; calibration cannot reject a model that ignores the state
- PASS worst_question_over_baseline: 0.6776 (limit 0.0500) -- worst is 'intent' at 0.6942 vs its own marginal 0.0166; the pooled gate hides this
- PASS workhorse_ece: 0.0498 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0504 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0498 (limit 0.0500) -- worst is 'choice' at ECE 0.0498 (overconfidence -0.002); pooled ECE cancels heads that err in opposite directions

**Blocked**: workhorse_adaptive_ece.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.6942 | 0.0166 | +0.6776 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.694 | 0.692 | -0.002 | 0.0498 | 0.0504 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.694 | 0.692 | -0.002 | 0.0498 | 0.0504 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0120 (95th percentile 0.0165), simulated over 200 resamples. The measured ECE is 0.1298.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.13,0.20)    13  0.181  0.077  +0.104  ###....|................................
  [0.20,0.27)    43  0.245  0.140  +0.105  ######....|.............................
  [0.27,0.33)    98  0.302  0.245  +0.057  ##########..|...........................
  [0.33,0.40)   158  0.368  0.222  +0.147  #########......|........................
  [0.40,0.47)   183  0.436  0.306  +0.130  ############.....|......................
  [0.47,0.53)   228  0.502  0.294  +0.208  ############........|...................
  [0.53,0.60)   253  0.568  0.415  +0.153  #################......|................
  [0.60,0.67)   224  0.635  0.438  +0.197  ##################.......|..............
  [0.67,0.73)   246  0.700  0.476  +0.224  ###################.........|...........
  [0.73,0.80)   244  0.767  0.537  +0.230  #####################..........|........
  [0.80,0.87)   248  0.836  0.609  +0.227  ########################.........|......
  [0.87,0.93)   419  0.903  0.654  +0.249  ##########################..........|...
  [0.93,1.00)  2643  0.988  0.917  +0.071  #####################################...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
