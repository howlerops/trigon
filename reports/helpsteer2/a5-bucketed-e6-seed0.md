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

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.5962 | 1.1349 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 6 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 6 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+c12bda56

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5717 | 0.0087 | 0.0101 | 0.5625 | 13.8 | 59.3 | 1286 |
| helpsteer2/calibrated | 5000 | 0.5717 | 0.0087 | 0.0101 | 0.5625 | 13.6 | 58.8 | 1286 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0098 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0069 on this run, p95 0.0098
- FAIL accuracy_over_baseline: 0.0193 (limit 0.0500) -- model 0.5717 vs marginal predictor 0.5524; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0042 (limit 0.0500) -- worst is 'correctness' at 0.4742 vs its own marginal 0.4784; the pooled gate hides this
- PASS workhorse_ece: 0.0087 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0101 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0087 (limit 0.0500) -- worst is 'score' at ECE 0.0087 (overconfidence +0.008); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `correctness` | 5,000 | 0.4742 | 0.4784 | -0.0042 ⚠ |
| `helpfulness` | 5,000 | 0.4054 | 0.4094 | -0.0040 ⚠ |
| `coherence` | 5,000 | 0.7120 | 0.7120 | +0.0000 |
| `verbosity` | 5,000 | 0.6534 | 0.6292 | +0.0242 |
| `complexity` | 5,000 | 0.6136 | 0.5330 | +0.0806 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.572 | 0.580 | +0.008 | 0.0087 | 0.0101 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.572 | 0.580 | +0.008 | 0.0087 | 0.0101 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0069 (95th percentile 0.0098), simulated over 200 resamples. The measured ECE is 0.0087.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)    66  0.249  0.227  +0.022  #########.|.............................
  [0.27,0.33)   648  0.310  0.304  +0.006  ############|...........................
  [0.33,0.40)  1984  0.370  0.376  -0.005  ###############|........................
  [0.40,0.47)  3449  0.436  0.434  +0.003  #################|......................
  [0.47,0.53)  4191  0.499  0.489  +0.010  ####################|...................
  [0.53,0.60)  3014  0.565  0.541  +0.023  ######################.|................
  [0.60,0.67)  2946  0.635  0.627  +0.009  #########################|..............
  [0.67,0.73)  4628  0.700  0.693  +0.007  ############################|...........
  [0.73,0.80)  3636  0.763  0.760  +0.003  ##############################.|........
  [0.80,0.87)   438  0.810  0.781  +0.029  ###############################.|.......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
