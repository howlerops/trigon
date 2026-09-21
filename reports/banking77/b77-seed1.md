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
| `intent` | 0.0124 |
| Epochs | 4 |
| Seed | 1 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1
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

Model(s): trigon-reference-0.1.0+4e65e7ac

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.7176 | 0.0361 | 0.0382 | 0.3892 | 40.7 | 125.5 | 608 |
| banking77/calibrated | 5000 | 0.7176 | 0.0361 | 0.0382 | 0.3892 | 35.6 | 57.7 | 608 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0207 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0155 on this run, p95 0.0207
- PASS accuracy_over_baseline: 0.7010 (limit 0.0500) -- model 0.7176 vs marginal predictor 0.0166; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.7010 (limit 0.0500) -- worst is 'intent' at 0.7176 vs its own marginal 0.0166; the pooled gate hides this
- PASS workhorse_ece: 0.0361 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0382 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0361 (limit 0.0500) -- worst is 'choice' at ECE 0.0361 (overconfidence -0.013); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.7176 | 0.0166 | +0.7010 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.718 | 0.705 | -0.013 | 0.0361 | 0.0382 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.718 | 0.705 | -0.013 | 0.0361 | 0.0382 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0155 (95th percentile 0.0207), simulated over 200 resamples. The measured ECE is 0.0361.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.13,0.20)    34  0.178  0.088  +0.089  ####...|................................
  [0.20,0.27)   104  0.239  0.183  +0.056  #######...|.............................
  [0.27,0.33)   188  0.303  0.245  +0.058  ##########..|...........................
  [0.33,0.40)   273  0.367  0.297  +0.071  ############...|........................
  [0.40,0.47)   321  0.435  0.396  +0.039  ################.|......................
  [0.47,0.53)   389  0.499  0.491  +0.008  ####################|...................
  [0.53,0.60)   396  0.567  0.558  +0.009  ######################.|................
  [0.60,0.67)   376  0.632  0.676  -0.043  #########################|#.............
  [0.67,0.73)   350  0.699  0.737  -0.038  ############################|...........
  [0.73,0.80)   405  0.768  0.778  -0.010  ###############################|........
  [0.80,0.87)   496  0.834  0.893  -0.059  #################################|##....
  [0.87,0.93)   634  0.902  0.957  -0.055  ####################################|#..
  [0.93,1.00)  1034  0.966  0.989  -0.023  #######################################|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
