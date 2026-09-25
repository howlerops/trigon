# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 2,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 3,080 |
| — held out of train to reach the floor | 1,920 |
| Questions per request | 1 |
| Labels per question | 77 |

| Question | Marginal predictor |
| --- | ---: |
| `intent` | 0.0130 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.9876 | 4.3675 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 6 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py banking77 -n 2000 --calibration-n 1000 --eval-n 0 --epochs 6 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+406709ed

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.3650 | 0.1036 | 0.1035 | 0.7927 | 12.4 | 15.0 | 609 |
| banking77/calibrated | 5000 | 0.3614 | 0.0230 | 0.0306 | 0.7790 | 11.9 | 14.7 | 609 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0201 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0136 on this run, p95 0.0201
- PASS accuracy_over_baseline: 0.3454 (limit 0.0500) -- model 0.3614 vs marginal predictor 0.0160; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.3454 (limit 0.0500) -- worst is 'intent' at 0.3614 vs its own marginal 0.0160; the pooled gate hides this
- PASS workhorse_ece: 0.0230 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0306 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0230 (limit 0.0500) -- worst is 'choice' at ECE 0.0230 (overconfidence +0.003); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.3614 | 0.0160 | +0.3454 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.361 | 0.364 | +0.003 | 0.0230 | 0.0306 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.361 | 0.364 | +0.003 | 0.0230 | 0.0306 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0145 (95th percentile 0.0200), simulated over 200 resamples. The measured ECE is 0.1036.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.00,0.07)     7  0.057  0.000  +0.057  ..|.....................................
  [0.07,0.13)   487  0.112  0.121  -0.009  ####|...................................
  [0.13,0.20)  1288  0.169  0.202  -0.033  #######|................................
  [0.20,0.27)  1317  0.231  0.326  -0.095  #########|###...........................
  [0.27,0.33)   840  0.297  0.442  -0.145  ############|#####......................
  [0.33,0.40)   477  0.364  0.501  -0.137  ###############|####....................
  [0.40,0.47)   242  0.432  0.686  -0.254  #################|#########.............
  [0.47,0.53)   108  0.498  0.796  -0.299  ####################|###########........
  [0.53,0.60)    94  0.565  0.851  -0.286  #######################|##########......
  [0.60,0.67)    64  0.632  0.953  -0.322  #########################|############..
  [0.67,0.73)    41  0.695  1.000  -0.305  ############################|###########
  [0.73,0.80)    23  0.756  0.870  -0.113  ##############################|####.....
  [0.80,0.87)    11  0.830  1.000  -0.170  #################################|######
  [0.87,0.93)     1  0.871  1.000  -0.129  ###################################|####
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
