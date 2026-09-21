# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 2 --floor-trials 40 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9310 | 158 |
| 2 | 0.8606 | 153 |
| 3 | 0.8607 | 162 |
| 4 | 0.8485 | 167 |
| 5 | 0.8400 | 159 |
| 6 | 0.8372 | 182 |
| 7 | 0.8351 | 154 |
| 8 | 0.8341 | 163 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **54%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+587bc40c

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6194 | 0.0096 | 0.0110 | 0.4629 | 4.3 | 6.3 | 117 |
| calibration/temperature-scaled | 6000 | 0.6194 | 0.0067 | 0.0082 | 0.4628 | 4.2 | 6.7 | 117 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0075 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0052 on this run, p95 0.0075
- PASS accuracy_over_baseline: 0.2279 (limit 0.0500) -- model 0.6194 vs marginal predictor 0.3915; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0023 (limit 0.0500) -- worst is 'size' at 0.2590 vs its own marginal 0.2613; the pooled gate hides this
- PASS workhorse_ece: 0.0067 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0082 (limit 0.0500)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2590 | 0.2613 | -0.0023 ⚠ |
| `at_risk` | 6,000 | 0.7498 | 0.6585 | +0.0913 |
| `plan` | 6,000 | 0.8493 | 0.2547 | +0.5947 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.619 | 0.624 | +0.005 | 0.0067 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0045 (95th percentile 0.0069), simulated over 40 resamples. The measured ECE is 0.0096.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.251  0.259  -0.008  ##########|.............................
  [0.47,0.53)     2  0.504  0.000  +0.504  ....................|...................
  [0.53,0.60)     1  0.563  0.000  +0.563  .......................|................
  [0.60,0.67)  1775  0.645  0.654  -0.008  ##########################|.............
  [0.67,0.73)     5  0.672  0.800  -0.128  ###########################|####........
  [0.73,0.80)  1337  0.792  0.773  +0.020  ###############################.|.......
  [0.80,0.87)  7653  0.838  0.829  +0.009  #################################.|.....
  [0.87,0.93)  1227  0.869  0.860  +0.010  ##################################.|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
