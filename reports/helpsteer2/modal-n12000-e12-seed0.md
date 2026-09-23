# helpsteer2

**HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0. https://huggingface.co/datasets/nvidia/HelpSteer2**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 12,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 1,038 |
| — held out of train to reach the floor | 3,962 |
| Questions per request | 5 |
| Labels per question | 5 |

| Question | Marginal predictor |
| --- | ---: |
| `coherence` | 0.7120 |
| `complexity` | 0.5330 |
| `correctness` | 0.4784 |
| `helpfulness` | 0.4094 |
| `verbosity` | 0.6292 |
| Epochs | 12 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 12 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+47fa9eb7

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5702 | 0.0232 | 0.0226 | 0.5623 | 13.6 | 58.6 | 1286 |
| helpsteer2/calibrated | 5000 | 0.5702 | 0.0232 | 0.0226 | 0.5623 | 13.3 | 59.3 | 1286 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0098 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0068 on this run, p95 0.0098
- FAIL accuracy_over_baseline: 0.0178 (limit 0.0500) -- model 0.5702 vs marginal predictor 0.5524; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0006 (limit 0.0500) -- worst is 'correctness' at 0.4778 vs its own marginal 0.4784; the pooled gate hides this
- PASS workhorse_ece: 0.0232 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0226 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0232 (limit 0.0500) -- worst is 'score' at ECE 0.0232 (overconfidence -0.019); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `correctness` | 5,000 | 0.4778 | 0.4784 | -0.0006 ⚠ |
| `helpfulness` | 5,000 | 0.4088 | 0.4094 | -0.0006 ⚠ |
| `coherence` | 5,000 | 0.7120 | 0.7120 | +0.0000 |
| `verbosity` | 5,000 | 0.6520 | 0.6292 | +0.0228 |
| `complexity` | 5,000 | 0.6006 | 0.5330 | +0.0676 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.570 | 0.551 | -0.019 | 0.0232 | 0.0226 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.570 | 0.551 | -0.019 | 0.0232 | 0.0226 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0068 (95th percentile 0.0098), simulated over 200 resamples. The measured ECE is 0.0232.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)    59  0.251  0.254  -0.003  ##########|.............................
  [0.27,0.33)   761  0.311  0.289  +0.022  ############|...........................
  [0.33,0.40)  2440  0.371  0.389  -0.018  ###############|........................
  [0.40,0.47)  4146  0.435  0.455  -0.020  #################|......................
  [0.47,0.53)  4133  0.498  0.489  +0.009  ####################|...................
  [0.53,0.60)  3164  0.566  0.569  -0.002  #######################|................
  [0.60,0.67)  4438  0.637  0.670  -0.033  #########################|#.............
  [0.67,0.73)  4698  0.697  0.742  -0.045  ############################|#..........
  [0.73,0.80)  1161  0.748  0.776  -0.028  ##############################|.........
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
