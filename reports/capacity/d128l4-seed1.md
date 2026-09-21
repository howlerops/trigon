# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 4000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 128 --layers 4 --noise 0.2 --seed 1 --floor-trials 20 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0326 | 138 |
| 2 | 1.0085 | 166 |
| 3 | 1.0036 | 166 |
| 4 | 1.0029 | 156 |
| 5 | 1.0006 | 154 |
| 6 | 0.9999 | 154 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **26%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+2747e27a, trigon-reference-0.1.0+2747e27a+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 4000 | 0.4833 | 0.0086 | 0.0250 | 0.5683 | 5.7 | 9.8 | 121 |
| calibration/temperature-scaled | 4000 | 0.4803 | 0.0203 | 0.0213 | 0.5706 | 5.6 | 7.7 | 121 |
| calibration/int8 | 4000 | 0.4799 | 0.0252 | 0.0290 | 0.5740 | 5.5 | 7.4 | 121 |

## Release gates

- PASS sample_size: 12000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0115 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0081 on this run, p95 0.0115
- PASS accuracy_over_baseline: 0.0859 (limit 0.0500) -- model 0.4803 vs marginal predictor 0.3943; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0160 (limit 0.0500) -- worst is 'size' at 0.2422 vs its own marginal 0.2582; the pooled gate hides this
- PASS workhorse_ece: 0.0203 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0213 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0311 (limit 0.0500) -- worst is 'choice' at ECE 0.0311 (overconfidence +0.024); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0050 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 4,000 | 0.2400 | 0.2582 | -0.0182 ⚠ |
| `at_risk` | 4,000 | 0.7538 | 0.6677 | +0.0860 |
| `plan` | 4,000 | 0.4562 | 0.2570 | +0.1992 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 12000 | 0.480 | 0.497 | +0.017 | 0.0203 | 0.0213 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 4000 | 0.445 | 0.469 | +0.024 | 0.0311 | 0.0692 |
| noul | 4000 | 0.754 | 0.755 | +0.001 | 0.0044 | 0.0289 |
| score | 4000 | 0.242 | 0.267 | +0.025 | 0.0252 | 0.0256 |

## Is this number evidence?

On these 12000 predictions a perfectly calibrated model scores a mean ECE of 0.0069 (95th percentile 0.0106), simulated over 20 resamples. The measured ECE is 0.0086.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  4000  0.257  0.240  +0.017  ##########|.............................
  [0.27,0.33)  2992  0.320  0.324  -0.004  #############|..........................
  [0.60,0.67)  1168  0.641  0.647  -0.006  ##########################|.............
  [0.73,0.80)   297  0.799  0.785  +0.015  ###############################.|.......
  [0.80,0.87)  3543  0.817  0.813  +0.003  #################################|......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
