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
| `intent` | 0.0144 |
| Epochs | 6 |
| Seed | 2 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 6 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 2
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

Model(s): trigon-reference-0.1.0+c6b37ab2

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.7194 | 0.0335 | 0.0353 | 0.3918 | 37.0 | 66.2 | 609 |
| banking77/calibrated | 5000 | 0.7126 | 0.0185 | 0.0183 | 0.3921 | 37.3 | 89.6 | 609 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0178 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0119 on this run, p95 0.0178
- PASS accuracy_over_baseline: 0.6952 (limit 0.0500) -- model 0.7126 vs marginal predictor 0.0174; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.6952 (limit 0.0500) -- worst is 'intent' at 0.7126 vs its own marginal 0.0174; the pooled gate hides this
- PASS workhorse_ece: 0.0185 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0183 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0185 (limit 0.0500) -- worst is 'choice' at ECE 0.0185 (overconfidence +0.010); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.7194 | 0.0174 | +0.7020 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.713 | 0.723 | +0.010 | 0.0185 | 0.0183 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.713 | 0.723 | +0.010 | 0.0185 | 0.0183 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0154 (95th percentile 0.0207), simulated over 200 resamples. The measured ECE is 0.0335.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.13,0.20)    37  0.174  0.135  +0.039  #####..|................................
  [0.20,0.27)   118  0.237  0.229  +0.008  #########|..............................
  [0.27,0.33)   197  0.301  0.279  +0.022  ###########.|...........................
  [0.33,0.40)   305  0.368  0.341  +0.027  ##############.|........................
  [0.40,0.47)   331  0.433  0.384  +0.049  ###############..|......................
  [0.47,0.53)   398  0.499  0.510  -0.011  ####################|...................
  [0.53,0.60)   371  0.569  0.601  -0.032  #######################|................
  [0.60,0.67)   350  0.633  0.637  -0.004  #########################|..............
  [0.67,0.73)   365  0.700  0.762  -0.062  ############################|#..........
  [0.73,0.80)   412  0.768  0.811  -0.043  ###############################|........
  [0.80,0.87)   466  0.835  0.893  -0.058  #################################|##....
  [0.87,0.93)   665  0.902  0.941  -0.039  ####################################|#..
  [0.93,1.00)   985  0.965  0.991  -0.026  #######################################|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
