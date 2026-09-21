# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-readout-per-level
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0899 | 86 |
| 2 | 1.1352 | 107 |
| 3 | 1.1381 | 101 |
| 4 | 1.0876 | 95 |
| 5 | 1.0375 | 113 |
| 6 | 1.0358 | 128 |
| 7 | 1.0349 | 116 |
| 8 | 1.0343 | 114 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **20%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+8fb718b9, trigon-reference-0.1.0+8fb718b9+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4557 | 0.0092 | 0.0633 | 0.5950 | 4.4 | 6.8 | 147 |
| calibration/temperature-scaled | 6000 | 0.4557 | 0.0214 | 0.0308 | 0.5897 | 4.4 | 7.8 | 147 |
| calibration/int8 | 6000 | 0.4557 | 0.0510 | 0.0538 | 0.6025 | 4.3 | 7.3 | 147 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0097 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0061 on this run, p95 0.0097
- PASS accuracy_over_baseline: 0.0634 (limit 0.0500) -- model 0.4557 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0055 (limit 0.0500) -- worst is 'size' at 0.2510 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0214 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0308 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0370 (limit 0.0500) -- worst is 'choice' at ECE 0.0370 (overconfidence +0.037); pooled ECE cancels heads that err in opposite directions
- FAIL quantization_ece_delta: 0.0296 (limit 0.0100)

**Blocked**: quantization_ece_delta.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2510 | 0.2565 | -0.0055 ⚠ |
| `at_risk` | 6,000 | 0.6672 | 0.6672 | +0.0000 |
| `plan` | 6,000 | 0.4488 | 0.2530 | +0.1958 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.456 | 0.461 | +0.005 | 0.0214 | 0.0308 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.449 | 0.486 | +0.037 | 0.0370 | 0.0370 |
| noul | 6000 | 0.667 | 0.644 | -0.024 | 0.0253 | 0.0419 |
| score | 6000 | 0.251 | 0.253 | +0.002 | 0.0020 | 0.0192 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0051 (95th percentile 0.0081), simulated over 40 resamples. The measured ECE is 0.0092.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.253  0.251  +0.002  ##########|.............................
  [0.27,0.33)  4493  0.325  0.316  +0.009  #############|..........................
  [0.60,0.67)  6000  0.653  0.667  -0.014  ##########################|.............
  [0.80,0.87)  1433  0.860  0.844  +0.017  ##################################|.....
  [0.87,0.93)    74  0.868  0.892  -0.024  ###################################|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
