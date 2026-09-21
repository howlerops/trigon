# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-head linear
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9215 | 109 |
| 2 | 0.9190 | 126 |
| 3 | 1.0347 | 152 |
| 4 | 1.0152 | 138 |
| 5 | 1.0129 | 145 |
| 6 | 0.9321 | 165 |
| 7 | 0.9110 | 189 |
| 8 | 0.8980 | 168 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **43%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+0020f70f, trigon-reference-0.1.0+0020f70f+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6107 | 0.0785 | 0.0780 | 0.4729 | 3.5 | 4.5 | 114 |
| calibration/temperature-scaled | 6000 | 0.6087 | 0.0518 | 0.0516 | 0.4673 | 3.5 | 4.5 | 114 |
| calibration/int8 | 6000 | 0.6104 | 0.0588 | 0.0586 | 0.4741 | 3.5 | 4.7 | 114 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0092 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0063 on this run, p95 0.0092
- PASS accuracy_over_baseline: 0.2164 (limit 0.0500) -- model 0.6087 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0168 (limit 0.0500) -- worst is 'size' at 0.2397 vs its own marginal 0.2565; the pooled gate hides this
- FAIL workhorse_ece: 0.0518 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0516 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0885 (limit 0.0500) -- worst is 'score' at ECE 0.0885 (overconfidence +0.089); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0070 (limit 0.0100)

**Blocked**: workhorse_ece, workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline, worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2397 | 0.2565 | -0.0168 ⚠ |
| `at_risk` | 6,000 | 0.7443 | 0.6672 | +0.0772 |
| `plan` | 6,000 | 0.8482 | 0.2530 | +0.5952 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.609 | 0.638 | +0.030 | 0.0518 | 0.0516 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.847 | 0.869 | +0.022 | 0.0220 | 0.0220 |
| noul | 6000 | 0.739 | 0.718 | -0.021 | 0.0450 | 0.0465 |
| score | 6000 | 0.240 | 0.328 | +0.089 | 0.0885 | 0.0887 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0075 (95th percentile 0.0110), simulated over 40 resamples. The measured ECE is 0.0785.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.27,0.33)  3919  0.307  0.235  +0.072  #########...|...........................
  [0.33,0.40)  2082  0.368  0.249  +0.119  ##########.....|........................
  [0.40,0.47)     3  0.405  1.000  -0.595  ################|#######################
  [0.47,0.53)   910  0.506  0.542  -0.036  ####################|#..................
  [0.53,0.60)     3  0.551  0.000  +0.551  ......................|.................
  [0.60,0.67)   938  0.606  0.649  -0.044  ########################|#..............
  [0.67,0.73)  2541  0.720  0.845  -0.125  #############################|####......
  [0.73,0.80)  3021  0.774  0.849  -0.074  ###############################|##......
  [0.80,0.87)  2530  0.854  0.821  +0.033  #################################.|.....
  [0.87,0.93)  2053  0.897  0.810  +0.088  ################################....|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
