# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 7,083 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 3,080 |
| — held out of train to reach the floor | 1,920 |
| Questions per request | 1 |
| Labels per question | 77 |

| Question | Marginal predictor |
| --- | ---: |
| `intent` | 0.0130 |
| Epochs | 6 |
| Seed | 0 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 6 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0
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

Model(s): trigon-reference-0.1.0+c23d7a1d

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.7320 | 0.0269 | 0.0257 | 0.3711 | 35.1 | 65.9 | 609 |
| banking77/calibrated | 5000 | 0.7320 | 0.0269 | 0.0257 | 0.3711 | 35.3 | 86.1 | 609 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0208 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0153 on this run, p95 0.0208
- PASS accuracy_over_baseline: 0.7160 (limit 0.0500) -- model 0.7320 vs marginal predictor 0.0160; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.7160 (limit 0.0500) -- worst is 'intent' at 0.7320 vs its own marginal 0.0160; the pooled gate hides this
- PASS workhorse_ece: 0.0269 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0257 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0269 (limit 0.0500) -- worst is 'choice' at ECE 0.0269 (overconfidence +0.000); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.7320 | 0.0160 | +0.7160 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.732 | 0.732 | +0.000 | 0.0269 | 0.0257 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.732 | 0.732 | +0.000 | 0.0269 | 0.0257 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0153 (95th percentile 0.0208), simulated over 200 resamples. The measured ECE is 0.0269.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.07,0.13)     1  0.133  0.000  +0.133  .....|..................................
  [0.13,0.20)    20  0.183  0.150  +0.033  ######.|................................
  [0.20,0.27)    75  0.234  0.253  -0.020  #########|..............................
  [0.27,0.33)   173  0.304  0.283  +0.021  ###########.|...........................
  [0.33,0.40)   235  0.368  0.315  +0.053  #############..|........................
  [0.40,0.47)   321  0.436  0.393  +0.043  ################.|......................
  [0.47,0.53)   356  0.499  0.435  +0.064  #################...|...................
  [0.53,0.60)   308  0.567  0.568  -0.001  #######################|................
  [0.60,0.67)   328  0.633  0.591  +0.041  ########################.|..............
  [0.67,0.73)   364  0.701  0.698  +0.003  ############################|...........
  [0.73,0.80)   397  0.770  0.783  -0.014  ###############################|........
  [0.80,0.87)   470  0.835  0.864  -0.028  #################################|#.....
  [0.87,0.93)   669  0.903  0.942  -0.038  ####################################|#..
  [0.93,1.00)  1283  0.970  0.985  -0.016  #######################################|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
