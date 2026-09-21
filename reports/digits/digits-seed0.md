# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0029 | 116 |
| 2 | 1.0232 | 110 |
| 3 | 1.0152 | 112 |
| 4 | 1.0805 | 132 |
| 5 | 1.0017 | 128 |
| 6 | 0.9571 | 124 |
| 7 | 0.9036 | 107 |
| 8 | 0.8787 | 106 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **46%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+27964533, trigon-reference-0.1.0+27964533+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5888 | 0.0153 | 0.0226 | 0.4915 | 3.8 | 5.1 | 121 |
| calibration/temperature-scaled | 6000 | 0.5881 | 0.0258 | 0.0298 | 0.4934 | 3.8 | 4.9 | 121 |
| calibration/int8 | 6000 | 0.5888 | 0.0249 | 0.0284 | 0.4919 | 3.8 | 4.7 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0087 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0060 on this run, p95 0.0087
- PASS accuracy_over_baseline: 0.1959 (limit 0.0500) -- model 0.5881 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0055 (limit 0.0500) -- worst is 'size' at 0.2510 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0258 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0298 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0441 (limit 0.0500) -- worst is 'noul' at ECE 0.0441 (overconfidence -0.015); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0009 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2510 | 0.2565 | -0.0055 ⚠ |
| `at_risk` | 6,000 | 0.6672 | 0.6672 | +0.0000 |
| `plan` | 6,000 | 0.8482 | 0.2530 | +0.5952 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.588 | 0.591 | +0.003 | 0.0258 | 0.0298 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.848 | 0.867 | +0.019 | 0.0331 | 0.0331 |
| noul | 6000 | 0.665 | 0.650 | -0.015 | 0.0441 | 0.0441 |
| score | 6000 | 0.251 | 0.256 | +0.005 | 0.0047 | 0.0260 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0056 (95th percentile 0.0087), simulated over 40 resamples. The measured ECE is 0.0153.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.256  0.251  +0.005  ##########|.............................
  [0.60,0.67)  3009  0.651  0.667  -0.017  ##########################|.............
  [0.67,0.73)  3729  0.685  0.702  -0.017  ###########################|............
  [0.73,0.80)   764  0.739  0.846  -0.107  ##############################|###......
  [0.80,0.87)  3399  0.858  0.851  +0.007  ##################################|.....
  [0.87,0.93)  1099  0.870  0.844  +0.026  ##################################.|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
