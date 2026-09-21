# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9789 | 127 |
| 2 | 0.9501 | 132 |
| 3 | 0.9474 | 135 |
| 4 | 0.9451 | 125 |
| 5 | 0.9454 | 125 |
| 6 | 0.9540 | 126 |
| 7 | 0.9511 | 123 |
| 8 | 0.9499 | 121 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **34%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+d9103d17, trigon-reference-0.1.0+d9103d17+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5232 | 0.0274 | 0.0308 | 0.5454 | 3.6 | 5.4 | 114 |
| calibration/temperature-scaled | 6000 | 0.5078 | 0.0419 | 0.0454 | 0.5646 | 3.6 | 5.1 | 114 |
| calibration/int8 | 6000 | 0.4666 | 0.2018 | 0.2026 | 0.7258 | 3.6 | 5.0 | 114 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0094 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0068 on this run, p95 0.0094
- PASS accuracy_over_baseline: 0.1156 (limit 0.0500) -- model 0.5078 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0402 (limit 0.0500) -- worst is 'at_risk' at 0.6270 vs its own marginal 0.6672; the pooled gate hides this
- PASS workhorse_ece: 0.0419 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0454 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0727 (limit 0.0500) -- worst is 'noul' at ECE 0.0727 (overconfidence +0.046); pooled ECE cancels heads that err in opposite directions
- FAIL quantization_ece_delta: 0.1599 (limit 0.0100)

**Blocked**: quantization_ece_delta.

**Advisory, not blocking**: worst_question_over_baseline, worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2525 | 0.2565 | -0.0040 ⚠ |
| `at_risk` | 6,000 | 0.6672 | 0.6672 | +0.0000 |
| `plan` | 6,000 | 0.6500 | 0.2530 | +0.3970 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.508 | 0.542 | +0.034 | 0.0419 | 0.0454 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.650 | 0.672 | +0.022 | 0.0222 | 0.0222 |
| noul | 6000 | 0.627 | 0.673 | +0.046 | 0.0727 | 0.0743 |
| score | 6000 | 0.246 | 0.282 | +0.035 | 0.0350 | 0.0423 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0060 (95th percentile 0.0087), simulated over 40 resamples. The measured ECE is 0.0274.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  3006  0.251  0.252  -0.001  ##########|.............................
  [0.27,0.33)  2994  0.329  0.253  +0.076  ##########...|..........................
  [0.33,0.40)    14  0.363  0.500  -0.137  ###############|####....................
  [0.40,0.47)  2980  0.459  0.454  +0.005  ##################|.....................
  [0.60,0.67)  6000  0.641  0.667  -0.026  ##########################|.............
  [0.67,0.73)     2  0.721  1.000  -0.279  #############################|##########
  [0.73,0.80)     3  0.736  1.000  -0.264  #############################|##########
  [0.80,0.87)  1499  0.851  0.845  +0.006  ##################################|.....
  [0.87,0.93)  1502  0.897  0.845  +0.053  ##################################..|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
