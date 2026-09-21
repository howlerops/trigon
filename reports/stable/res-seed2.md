# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 2 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9375 | 118 |
| 2 | 0.9143 | 133 |
| 3 | 0.9562 | 106 |
| 4 | 0.9457 | 94 |
| 5 | 0.9783 | 100 |
| 6 | 0.9776 | 112 |
| 7 | 0.9719 | 130 |
| 8 | 0.9425 | 84 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **36%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+2aabdfbc, trigon-reference-0.1.0+2aabdfbc+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5891 | 0.0487 | 0.0563 | 0.4993 | 4.5 | 6.8 | 144 |
| calibration/temperature-scaled | 6000 | 0.5909 | 0.0095 | 0.0167 | 0.4886 | 4.5 | 6.9 | 144 |
| calibration/int8 | 6000 | 0.5743 | 0.0215 | 0.0336 | 0.4923 | 4.5 | 6.7 | 144 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0089 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0060 on this run, p95 0.0089
- PASS accuracy_over_baseline: 0.1994 (limit 0.0500) -- model 0.5909 vs marginal predictor 0.3915; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0012 (limit 0.0500) -- worst is 'size' at 0.2602 vs its own marginal 0.2613; the pooled gate hides this
- PASS workhorse_ece: 0.0095 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0167 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0219 (limit 0.0500) -- worst is 'noul' at ECE 0.0219 (overconfidence -0.007); pooled ECE cancels heads that err in opposite directions
- FAIL quantization_ece_delta: 0.0120 (limit 0.0100)

**Blocked**: quantization_ece_delta.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2602 | 0.2613 | -0.0012 ⚠ |
| `at_risk` | 6,000 | 0.6585 | 0.6585 | +0.0000 |
| `plan` | 6,000 | 0.8487 | 0.2547 | +0.5940 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.591 | 0.589 | -0.002 | 0.0095 | 0.0167 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.849 | 0.850 | +0.001 | 0.0062 | 0.0234 |
| noul | 6000 | 0.664 | 0.657 | -0.007 | 0.0219 | 0.0225 |
| score | 6000 | 0.260 | 0.260 | -0.000 | 0.0003 | 0.0177 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0050 (95th percentile 0.0071), simulated over 40 resamples. The measured ECE is 0.0487.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.260  0.260  -0.000  ##########|.............................
  [0.40,0.47)     4  0.418  0.000  +0.418  .................|......................
  [0.60,0.67)  7464  0.624  0.697  -0.073  #########################|##............
  [0.73,0.80)  2290  0.782  0.861  -0.079  ###############################|##......
  [0.80,0.87)   703  0.802  0.822  -0.021  ################################|.......
  [0.87,0.93)  1539  0.928  0.840  +0.088  ##################################...|..
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
