# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-head linear
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9891 | 116 |
| 2 | 0.9452 | 116 |
| 3 | 1.0036 | 132 |
| 4 | 0.9552 | 132 |
| 5 | 0.9481 | 141 |
| 6 | 0.9465 | 122 |
| 7 | 0.9418 | 117 |
| 8 | 0.9406 | 118 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **36%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+6937d066, trigon-reference-0.1.0+6937d066+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5201 | 0.0078 | 0.0779 | 0.5439 | 3.7 | 5.1 | 121 |
| calibration/temperature-scaled | 6000 | 0.5201 | 0.0078 | 0.0779 | 0.5439 | 3.6 | 4.1 | 121 |
| calibration/int8 | 6000 | 0.5201 | 0.0077 | 0.0777 | 0.5439 | 3.6 | 4.8 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0088 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0059 on this run, p95 0.0088
- PASS accuracy_over_baseline: 0.1261 (limit 0.0500) -- model 0.5201 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0098 (limit 0.0500) -- worst is 'size' at 0.2460 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0078 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0779 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0137 (limit 0.0500) -- worst is 'score' at ECE 0.0137 (overconfidence +0.014); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0001 (limit 0.0100)

**Blocked**: workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2460 | 0.2558 | -0.0098 ⚠ |
| `at_risk` | 6,000 | 0.6687 | 0.6687 | +0.0000 |
| `plan` | 6,000 | 0.6457 | 0.2577 | +0.3880 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.520 | 0.522 | +0.002 | 0.0078 | 0.0779 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.646 | 0.642 | -0.004 | 0.0039 | 0.2094 |
| noul | 6000 | 0.669 | 0.665 | -0.004 | 0.0059 | 0.0149 |
| score | 6000 | 0.246 | 0.260 | +0.014 | 0.0137 | 0.0281 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0059 (95th percentile 0.0088), simulated over 40 resamples. The measured ECE is 0.0078.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.260  0.246  +0.014  ##########|.............................
  [0.40,0.47)  3060  0.448  0.453  -0.005  ##################|.....................
  [0.60,0.67)  2940  0.656  0.666  -0.010  ##########################|.............
  [0.67,0.73)  3060  0.674  0.671  +0.002  ###########################|............
  [0.80,0.87)  2940  0.844  0.846  -0.003  ##################################|.....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
