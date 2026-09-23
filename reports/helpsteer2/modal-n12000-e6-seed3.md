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
| `coherence` | 0.7104 |
| `complexity` | 0.5392 |
| `correctness` | 0.4624 |
| `helpfulness` | 0.3992 |
| `verbosity` | 0.6300 |
| Epochs | 6 |
| Seed | 3 |
| Device | cuda (NVIDIA A10) |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 6 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+da994a4c

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5638 | 0.0139 | 0.0168 | 0.5697 | 14.3 | 63.3 | 1281 |
| helpsteer2/calibrated | 5000 | 0.5638 | 0.0139 | 0.0168 | 0.5697 | 14.2 | 61.8 | 1281 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0097 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0067 on this run, p95 0.0097
- FAIL accuracy_over_baseline: 0.0155 (limit 0.0500) -- model 0.5638 vs marginal predictor 0.5482; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0002 (limit 0.0500) -- worst is 'helpfulness' at 0.3990 vs its own marginal 0.3992; the pooled gate hides this
- PASS workhorse_ece: 0.0139 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0168 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0139 (limit 0.0500) -- worst is 'score' at ECE 0.0139 (overconfidence +0.011); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `helpfulness` | 5,000 | 0.3990 | 0.3992 | -0.0002 ⚠ |
| `coherence` | 5,000 | 0.7104 | 0.7104 | +0.0000 |
| `correctness` | 5,000 | 0.4624 | 0.4624 | +0.0000 |
| `verbosity` | 5,000 | 0.6470 | 0.6300 | +0.0170 |
| `complexity` | 5,000 | 0.6000 | 0.5392 | +0.0608 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.564 | 0.574 | +0.011 | 0.0139 | 0.0168 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.564 | 0.574 | +0.011 | 0.0139 | 0.0168 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0067 (95th percentile 0.0097), simulated over 200 resamples. The measured ECE is 0.0139.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)    44  0.250  0.182  +0.068  #######...|.............................
  [0.27,0.33)   652  0.310  0.305  +0.005  ############|...........................
  [0.33,0.40)  1911  0.371  0.381  -0.011  ###############|........................
  [0.40,0.47)  3612  0.436  0.416  +0.019  #################|......................
  [0.47,0.53)  4352  0.498  0.467  +0.031  ###################.|...................
  [0.53,0.60)  3158  0.566  0.547  +0.020  ######################.|................
  [0.60,0.67)  3171  0.635  0.625  +0.011  #########################|..............
  [0.67,0.73)  4409  0.700  0.700  +0.000  ############################|...........
  [0.73,0.80)  3674  0.761  0.766  -0.005  ##############################|.........
  [0.80,0.87)    17  0.805  0.824  -0.018  ################################|.......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
