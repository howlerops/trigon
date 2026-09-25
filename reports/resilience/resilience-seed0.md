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
| Epochs | 3 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py banking77 -n 2000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+373a46cd

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.1190 | 0.0321 | 0.0364 | 0.9464 | 12.2 | 15.4 | 609 |
| banking77/calibrated | 5000 | 0.1206 | 0.0077 | 0.0191 | 0.9448 | 12.3 | 15.0 | 609 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0113 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0070 on this run, p95 0.0113
- PASS accuracy_over_baseline: 0.1046 (limit 0.0500) -- model 0.1206 vs marginal predictor 0.0160; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1046 (limit 0.0500) -- worst is 'intent' at 0.1206 vs its own marginal 0.0160; the pooled gate hides this
- PASS workhorse_ece: 0.0077 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0191 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0077 (limit 0.0500) -- worst is 'choice' at ECE 0.0077 (overconfidence +0.004); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.1206 | 0.0160 | +0.1046 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.121 | 0.125 | +0.004 | 0.0077 | 0.0191 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.121 | 0.125 | +0.004 | 0.0077 | 0.0191 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0068 (95th percentile 0.0107), simulated over 200 resamples. The measured ECE is 0.0321.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.00,0.07)  1519  0.055  0.066  -0.011  ##|.....................................
  [0.07,0.13)  2333  0.084  0.132  -0.048  ###|#...................................
  [0.13,0.20)   701  0.161  0.154  +0.007  ######|.................................
  [0.20,0.27)   444  0.225  0.171  +0.054  #######..|..............................
  [0.27,0.33)     3  0.271  0.667  -0.396  ###########|###############.............
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
