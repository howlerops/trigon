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
| `coherence` | 0.7184 |
| `complexity` | 0.5242 |
| `correctness` | 0.4802 |
| `helpfulness` | 0.4194 |
| `verbosity` | 0.6256 |
| Epochs | 12 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 12 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-reference-0.1.0+48edf996

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5738 | 0.0187 | 0.0188 | 0.5641 | 13.3 | 58.9 | 1290 |
| helpsteer2/calibrated | 5000 | 0.5738 | 0.0187 | 0.0188 | 0.5641 | 13.1 | 58.8 | 1290 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0106 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0072 on this run, p95 0.0106
- FAIL accuracy_over_baseline: 0.0202 (limit 0.0500) -- model 0.5738 vs marginal predictor 0.5536; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0038 (limit 0.0500) -- worst is 'correctness' at 0.4764 vs its own marginal 0.4802; the pooled gate hides this
- PASS workhorse_ece: 0.0187 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0188 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0187 (limit 0.0500) -- worst is 'score' at ECE 0.0187 (overconfidence +0.004); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `correctness` | 5,000 | 0.4764 | 0.4802 | -0.0038 ⚠ |
| `coherence` | 5,000 | 0.7184 | 0.7184 | +0.0000 |
| `helpfulness` | 5,000 | 0.4204 | 0.4194 | +0.0010 |
| `verbosity` | 5,000 | 0.6550 | 0.6256 | +0.0294 |
| `complexity` | 5,000 | 0.5986 | 0.5242 | +0.0744 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.574 | 0.578 | +0.004 | 0.0187 | 0.0188 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.574 | 0.578 | +0.004 | 0.0187 | 0.0188 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0072 (95th percentile 0.0106), simulated over 200 resamples. The measured ECE is 0.0187.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   304  0.250  0.319  -0.069  ##########|##...........................
  [0.27,0.33)  1213  0.304  0.334  -0.029  ############|...........................
  [0.33,0.40)  1975  0.369  0.427  -0.058  ###############|#.......................
  [0.40,0.47)  2774  0.436  0.439  -0.004  #################|......................
  [0.47,0.53)  3654  0.500  0.484  +0.016  ###################.|...................
  [0.53,0.60)  3283  0.566  0.559  +0.008  ######################.|................
  [0.60,0.67)  3430  0.634  0.619  +0.015  #########################|..............
  [0.67,0.73)  3948  0.701  0.692  +0.009  ############################|...........
  [0.73,0.80)  3389  0.763  0.746  +0.018  ##############################.|........
  [0.80,0.87)  1026  0.821  0.769  +0.052  ###############################..|......
  [0.87,0.93)     4  0.871  1.000  -0.129  ###################################|####
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
