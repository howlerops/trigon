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
| `intent` | 0.0136 |
| Epochs | 6 |
| Seed | 3 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 6 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 3
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

Model(s): trigon-reference-0.1.0+58b0babb

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.7404 | 0.0177 | 0.0212 | 0.3645 | 35.1 | 63.7 | 609 |
| banking77/calibrated | 5000 | 0.7404 | 0.0177 | 0.0212 | 0.3645 | 35.0 | 81.9 | 609 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0205 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0149 on this run, p95 0.0205
- PASS accuracy_over_baseline: 0.7242 (limit 0.0500) -- model 0.7404 vs marginal predictor 0.0162; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.7242 (limit 0.0500) -- worst is 'intent' at 0.7404 vs its own marginal 0.0162; the pooled gate hides this
- PASS workhorse_ece: 0.0177 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0212 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0177 (limit 0.0500) -- worst is 'choice' at ECE 0.0177 (overconfidence -0.006); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.7404 | 0.0162 | +0.7242 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.740 | 0.734 | -0.006 | 0.0177 | 0.0212 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.740 | 0.734 | -0.006 | 0.0177 | 0.0212 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0149 (95th percentile 0.0205), simulated over 200 resamples. The measured ECE is 0.0177.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.07,0.13)     2  0.125  0.000  +0.125  .....|..................................
  [0.13,0.20)    18  0.171  0.167  +0.005  #######|................................
  [0.20,0.27)    81  0.239  0.136  +0.103  #####.....|.............................
  [0.27,0.33)   171  0.300  0.246  +0.055  ##########..|...........................
  [0.33,0.40)   238  0.368  0.366  +0.002  ###############|........................
  [0.40,0.47)   292  0.434  0.414  +0.020  #################|......................
  [0.47,0.53)   357  0.501  0.493  +0.008  ####################|...................
  [0.53,0.60)   348  0.567  0.560  +0.006  ######################.|................
  [0.60,0.67)   324  0.633  0.636  -0.002  #########################|..............
  [0.67,0.73)   320  0.702  0.706  -0.004  ############################|...........
  [0.73,0.80)   392  0.766  0.781  -0.014  ###############################|........
  [0.80,0.87)   462  0.833  0.864  -0.031  #################################|#.....
  [0.87,0.93)   608  0.903  0.926  -0.023  ####################################|...
  [0.93,1.00)  1387  0.969  0.986  -0.017  #######################################|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
