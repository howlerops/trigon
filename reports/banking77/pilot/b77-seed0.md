# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 4,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases (the corpus's own test split) | 1,500 |
| Options | 77 |
| Marginal predictor accuracy | 0.0140 |
| Epochs | 4 |
| Seed | 0 |

```
python scripts/train_corpus.py banking77 -n 4000 --calibration-n 1000 --eval-n 1500 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-reference-0.1.0+eb6d2fee

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 1500 | 0.4640 | 0.1123 | 0.1150 | 0.6920 | 35.9 | 72.9 | 608 |
| banking77/calibrated | 1500 | 0.4633 | 0.0324 | 0.0411 | 0.6767 | 36.4 | 60.0 | 608 |

## Release gates

- FAIL sample_size: 1500.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- FAIL gate_is_testable: 0.0348 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0221 on this run, p95 0.0348
- PASS accuracy_over_baseline: 0.4453 (limit 0.0500) -- model 0.4633 vs marginal predictor 0.0180; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.4453 (limit 0.0500) -- worst is 'intent' at 0.4633 vs its own marginal 0.0180; the pooled gate hides this
- PASS workhorse_ece: 0.0324 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0411 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0324 (limit 0.0500) -- worst is 'choice' at ECE 0.0324 (overconfidence +0.013); pooled ECE cancels heads that err in opposite directions

**Blocked**: sample_size, gate_is_testable.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 1,500 | 0.4640 | 0.0180 | +0.4460 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 1500 | 0.463 | 0.476 | +0.013 | 0.0324 | 0.0411 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 1500 | 0.463 | 0.476 | +0.013 | 0.0324 | 0.0411 |

## Is this number evidence?

On these 1500 predictions a perfectly calibrated model scores a mean ECE of 0.0297 (95th percentile 0.0409), simulated over 200 resamples. The measured ECE is 0.1123.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.07,0.13)    44  0.116  0.091  +0.025  ####.|..................................
  [0.13,0.20)   270  0.169  0.189  -0.020  #######|................................
  [0.20,0.27)   279  0.231  0.280  -0.048  #########|#.............................
  [0.27,0.33)   215  0.299  0.423  -0.125  ############|####.......................
  [0.33,0.40)   182  0.368  0.456  -0.089  ###############|##......................
  [0.40,0.47)   158  0.429  0.639  -0.210  #################|########..............
  [0.47,0.53)   106  0.501  0.642  -0.141  ####################|#####..............
  [0.53,0.60)    73  0.562  0.863  -0.301  ######################|############.....
  [0.60,0.67)    61  0.629  0.836  -0.207  #########################|#######.......
  [0.67,0.73)    61  0.699  0.951  -0.252  ############################|#########..
  [0.73,0.80)    33  0.759  0.909  -0.150  ##############################|#####....
  [0.80,0.87)    11  0.829  1.000  -0.171  #################################|######
  [0.87,0.93)     6  0.893  1.000  -0.107  ####################################|###
  [0.93,1.00)     1  0.937  1.000  -0.063  #####################################|##
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
