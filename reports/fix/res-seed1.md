# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0971 | 139 |
| 2 | 1.0441 | 149 |
| 3 | 1.0186 | 127 |
| 4 | 1.0380 | 128 |
| 5 | 1.0369 | 128 |
| 6 | 1.0367 | 128 |
| 7 | 1.0357 | 130 |
| 8 | 1.0352 | 133 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **20%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+9f25c435, trigon-reference-0.1.0+9f25c435+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4872 | 0.0258 | 0.0530 | 0.5712 | 3.8 | 5.1 | 114 |
| calibration/temperature-scaled | 6000 | 0.4872 | 0.0258 | 0.0530 | 0.5712 | 3.8 | 4.5 | 114 |
| calibration/int8 | 6000 | 0.4872 | 0.0259 | 0.0534 | 0.5712 | 3.8 | 5.2 | 114 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0078 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0049 on this run, p95 0.0078
- PASS accuracy_over_baseline: 0.0931 (limit 0.0500) -- model 0.4872 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0027 (limit 0.0500) -- worst is 'size' at 0.2585 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0258 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0530 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0437 (limit 0.0500) -- worst is 'score' at ECE 0.0437 (overconfidence +0.044); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0000 (limit 0.0100)

**Blocked**: workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2585 | 0.2558 | +0.0027 |
| `at_risk` | 6,000 | 0.7537 | 0.6687 | +0.0850 |
| `plan` | 6,000 | 0.4493 | 0.2577 | +0.1917 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.487 | 0.492 | +0.005 | 0.0258 | 0.0530 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.449 | 0.451 | +0.002 | 0.0021 | 0.1667 |
| noul | 6000 | 0.754 | 0.724 | -0.030 | 0.0318 | 0.0357 |
| score | 6000 | 0.259 | 0.302 | +0.044 | 0.0437 | 0.0459 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0049 (95th percentile 0.0078), simulated over 40 resamples. The measured ECE is 0.0258.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.27,0.33) 10541  0.312  0.286  +0.026  ###########.|...........................
  [0.47,0.53)     4  0.509  0.500  +0.009  ####################|...................
  [0.53,0.60)  1301  0.557  0.654  -0.098  ######################|###..............
  [0.60,0.67)   444  0.636  0.624  +0.012  #########################|..............
  [0.73,0.80)  4251  0.784  0.798  -0.014  ###############################|........
  [0.80,0.87)  1459  0.846  0.846  +0.001  ##################################|.....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
