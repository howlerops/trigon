# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 2 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-readout-per-level
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9004 | 82 |
| 2 | 0.8836 | 91 |
| 3 | 0.8809 | 99 |
| 4 | 0.8730 | 126 |
| 5 | 0.9053 | 160 |
| 6 | 0.9117 | 224 |
| 7 | 0.9074 | 194 |
| 8 | 0.9051 | 196 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **42%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+8568f856, trigon-reference-0.1.0+8568f856+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6162 | 0.0298 | 0.0355 | 0.4701 | 4.2 | 7.3 | 147 |
| calibration/temperature-scaled | 6000 | 0.6146 | 0.0215 | 0.0311 | 0.4677 | 4.2 | 7.1 | 147 |
| calibration/int8 | 6000 | 0.6148 | 0.0272 | 0.0349 | 0.4696 | 4.2 | 7.1 | 147 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0077 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0056 on this run, p95 0.0077
- PASS accuracy_over_baseline: 0.2231 (limit 0.0500) -- model 0.6146 vs marginal predictor 0.3915; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0153 (limit 0.0500) -- worst is 'size' at 0.2460 vs its own marginal 0.2613; the pooled gate hides this
- PASS workhorse_ece: 0.0215 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0311 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0338 (limit 0.0500) -- worst is 'score' at ECE 0.0338 (overconfidence +0.034); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0057 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2490 | 0.2613 | -0.0123 ⚠ |
| `at_risk` | 6,000 | 0.7503 | 0.6585 | +0.0918 |
| `plan` | 6,000 | 0.8493 | 0.2547 | +0.5947 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.615 | 0.631 | +0.017 | 0.0215 | 0.0311 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.849 | 0.850 | +0.001 | 0.0127 | 0.0257 |
| noul | 6000 | 0.749 | 0.764 | +0.015 | 0.0287 | 0.0339 |
| score | 6000 | 0.246 | 0.280 | +0.034 | 0.0338 | 0.0347 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0055 (95th percentile 0.0080), simulated over 40 resamples. The measured ECE is 0.0298.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.254  0.249  +0.005  ##########|.............................
  [0.40,0.47)     1  0.459  0.000  +0.459  ..................|.....................
  [0.60,0.67)  1463  0.642  0.855  -0.213  ##########################|#######......
  [0.67,0.73)   956  0.692  0.646  +0.045  ##########################..|...........
  [0.73,0.80)  2061  0.776  0.751  +0.025  ##############################.|........
  [0.80,0.87)  5976  0.817  0.818  -0.001  #################################|......
  [0.87,0.93)  1543  0.901  0.840  +0.061  ##################################..|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
