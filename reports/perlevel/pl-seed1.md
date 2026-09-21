# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-readout-per-level
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9911 | 85 |
| 2 | 0.9235 | 87 |
| 3 | 0.9202 | 88 |
| 4 | 0.9167 | 84 |
| 5 | 0.9152 | 84 |
| 6 | 0.9140 | 86 |
| 7 | 0.9111 | 96 |
| 8 | 0.9090 | 82 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **41%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+74b3f33c, trigon-reference-0.1.0+74b3f33c+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5486 | 0.0056 | 0.0232 | 0.5166 | 4.1 | 7.4 | 147 |
| calibration/temperature-scaled | 6000 | 0.5486 | 0.0056 | 0.0232 | 0.5166 | 4.1 | 7.1 | 147 |
| calibration/int8 | 6000 | 0.5482 | 0.0059 | 0.0238 | 0.5166 | 4.1 | 6.9 | 147 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0087 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0055 on this run, p95 0.0087
- PASS accuracy_over_baseline: 0.1546 (limit 0.0500) -- model 0.5486 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0078 (limit 0.0500) -- worst is 'size' at 0.2480 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0056 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0232 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0088 (limit 0.0500) -- worst is 'noul' at ECE 0.0088 (overconfidence -0.002); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0003 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2480 | 0.2558 | -0.0078 ⚠ |
| `at_risk` | 6,000 | 0.7535 | 0.6687 | +0.0848 |
| `plan` | 6,000 | 0.6443 | 0.2577 | +0.3867 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.549 | 0.548 | -0.001 | 0.0056 | 0.0232 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.644 | 0.640 | -0.004 | 0.0052 | 0.0659 |
| noul | 6000 | 0.753 | 0.751 | -0.002 | 0.0088 | 0.0182 |
| score | 6000 | 0.248 | 0.252 | +0.004 | 0.0058 | 0.0231 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0055 (95th percentile 0.0087), simulated over 40 resamples. The measured ECE is 0.0056.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  5948  0.251  0.247  +0.005  ##########|.............................
  [0.27,0.33)    52  0.277  0.404  -0.127  ###########|####........................
  [0.40,0.47)  3031  0.444  0.450  -0.006  ##################|.....................
  [0.47,0.53)    29  0.478  0.483  -0.005  ###################|....................
  [0.60,0.67)  1749  0.639  0.646  -0.007  ##########################|.............
  [0.73,0.80)  1059  0.770  0.792  -0.022  ###############################|........
  [0.80,0.87)  6059  0.823  0.822  +0.002  #################################|......
  [0.87,0.93)    73  0.882  0.849  +0.033  ##################################.|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
