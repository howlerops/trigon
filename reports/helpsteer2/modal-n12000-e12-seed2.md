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
| `coherence` | 0.7164 |
| `complexity` | 0.5286 |
| `correctness` | 0.4808 |
| `helpfulness` | 0.4186 |
| `verbosity` | 0.6210 |
| Epochs | 12 |
| Seed | 2 |
| Device | cuda (NVIDIA A10) |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 12 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+4909ba57

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5728 | 0.0222 | 0.0229 | 0.5650 | 14.0 | 60.9 | 1291 |
| helpsteer2/calibrated | 5000 | 0.5728 | 0.0222 | 0.0229 | 0.5650 | 13.9 | 59.3 | 1291 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0100 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0071 on this run, p95 0.0100
- FAIL accuracy_over_baseline: 0.0197 (limit 0.0500) -- model 0.5728 vs marginal predictor 0.5531; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0028 (limit 0.0500) -- worst is 'helpfulness' at 0.4158 vs its own marginal 0.4186; the pooled gate hides this
- PASS workhorse_ece: 0.0222 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0229 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0222 (limit 0.0500) -- worst is 'score' at ECE 0.0222 (overconfidence +0.001); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `helpfulness` | 5,000 | 0.4158 | 0.4186 | -0.0028 ⚠ |
| `correctness` | 5,000 | 0.4792 | 0.4808 | -0.0016 ⚠ |
| `coherence` | 5,000 | 0.7164 | 0.7164 | +0.0000 |
| `verbosity` | 5,000 | 0.6510 | 0.6210 | +0.0300 |
| `complexity` | 5,000 | 0.6014 | 0.5286 | +0.0728 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.573 | 0.573 | +0.001 | 0.0222 | 0.0229 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.573 | 0.573 | +0.001 | 0.0222 | 0.0229 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0071 (95th percentile 0.0100), simulated over 200 resamples. The measured ECE is 0.0222.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   276  0.249  0.239  +0.010  ##########|.............................
  [0.27,0.33)  1463  0.306  0.342  -0.036  ############|#..........................
  [0.33,0.40)  2576  0.369  0.392  -0.024  ###############|........................
  [0.40,0.47)  3207  0.435  0.461  -0.026  #################|......................
  [0.47,0.53)  3258  0.499  0.518  -0.019  ####################|...................
  [0.53,0.60)  2482  0.566  0.571  -0.004  #######################|................
  [0.60,0.67)  2704  0.636  0.634  +0.002  #########################|..............
  [0.67,0.73)  4242  0.702  0.676  +0.026  ###########################.|...........
  [0.73,0.80)  3350  0.762  0.738  +0.024  ##############################|.........
  [0.80,0.87)  1438  0.827  0.768  +0.059  ###############################..|......
  [0.87,0.93)     4  0.869  1.000  -0.131  ###################################|####
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
