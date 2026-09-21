# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 4000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 20 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9480 | 118 |
| 2 | 0.9425 | 124 |
| 3 | 1.1114 | 139 |
| 4 | 1.0999 | 158 |
| 5 | 1.0992 | 150 |
| 6 | 1.0980 | 134 |
| 7 | 1.0974 | 131 |
| 8 | 1.0970 | 133 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **10%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+f5641141, trigon-reference-0.1.0+f5641141+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 4000 | 0.4173 | 0.0220 | 0.0371 | 0.6187 | 3.7 | 4.7 | 121 |
| calibration/temperature-scaled | 4000 | 0.4173 | 0.0220 | 0.0371 | 0.6187 | 3.8 | 4.8 | 121 |
| calibration/int8 | 4000 | 0.4173 | 0.0220 | 0.0372 | 0.6187 | 3.8 | 5.0 | 121 |

## Release gates

- PASS sample_size: 12000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0114 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0070 on this run, p95 0.0114
- FAIL accuracy_over_baseline: 0.0251 (limit 0.0500) -- model 0.4173 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0005 (limit 0.0500) -- worst is 'plan' at 0.2535 vs its own marginal 0.2540; the pooled gate hides this
- PASS workhorse_ece: 0.0220 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0371 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0442 (limit 0.0500) -- worst is 'choice' at ECE 0.0442 (overconfidence +0.044); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0000 (limit 0.0100)

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `plan` | 4,000 | 0.2535 | 0.2540 | -0.0005 ⚠ |
| `size` | 4,000 | 0.2567 | 0.2565 | +0.0002 |
| `at_risk` | 4,000 | 0.7415 | 0.6660 | +0.0755 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 12000 | 0.417 | 0.430 | +0.013 | 0.0220 | 0.0371 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 4000 | 0.254 | 0.298 | +0.044 | 0.0442 | 0.1756 |
| noul | 4000 | 0.742 | 0.741 | -0.001 | 0.0178 | 0.0298 |
| score | 4000 | 0.257 | 0.253 | -0.004 | 0.0041 | 0.0199 |

## Is this number evidence?

On these 12000 predictions a perfectly calibrated model scores a mean ECE of 0.0070 (95th percentile 0.0114), simulated over 20 resamples. The measured ECE is 0.0220.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  4000  0.253  0.257  -0.004  ##########|.............................
  [0.27,0.33)  4000  0.298  0.254  +0.044  ##########..|...........................
  [0.60,0.67)   965  0.642  0.669  -0.027  ##########################|.............
  [0.67,0.73)   948  0.696  0.659  +0.037  ##########################..|...........
  [0.80,0.87)  2087  0.807  0.812  -0.005  ################################|.......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
