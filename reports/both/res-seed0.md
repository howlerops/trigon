# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9738 | 130 |
| 2 | 0.9846 | 122 |
| 3 | 0.9840 | 115 |
| 4 | 0.9741 | 119 |
| 5 | 0.9326 | 110 |
| 6 | 0.9129 | 110 |
| 7 | 0.9097 | 114 |
| 8 | 0.9083 | 113 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **41%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+add7bf93, trigon-reference-0.1.0+add7bf93+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5552 | 0.0117 | 0.0143 | 0.5104 | 3.8 | 5.1 | 121 |
| calibration/temperature-scaled | 6000 | 0.5552 | 0.0117 | 0.0143 | 0.5104 | 3.8 | 5.0 | 121 |
| calibration/int8 | 6000 | 0.5552 | 0.0116 | 0.0144 | 0.5104 | 3.8 | 5.2 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0090 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0065 on this run, p95 0.0090
- PASS accuracy_over_baseline: 0.1630 (limit 0.0500) -- model 0.5552 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0072 (limit 0.0500) -- worst is 'size' at 0.2493 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0117 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0143 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0185 (limit 0.0500) -- worst is 'noul' at ECE 0.0185 (overconfidence +0.001); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0001 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2493 | 0.2565 | -0.0072 ⚠ |
| `at_risk` | 6,000 | 0.7627 | 0.6672 | +0.0955 |
| `plan` | 6,000 | 0.6537 | 0.2530 | +0.4007 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.555 | 0.558 | +0.003 | 0.0117 | 0.0143 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.654 | 0.657 | +0.003 | 0.0116 | 0.0193 |
| noul | 6000 | 0.763 | 0.763 | +0.001 | 0.0185 | 0.0287 |
| score | 6000 | 0.249 | 0.254 | +0.005 | 0.0050 | 0.0221 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0065 (95th percentile 0.0090), simulated over 40 resamples. The measured ECE is 0.0117.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.254  0.249  +0.005  ##########|.............................
  [0.40,0.47)  2670  0.450  0.460  -0.010  ##################|.....................
  [0.47,0.53)   302  0.472  0.434  +0.038  #################..|....................
  [0.53,0.60)     3  0.560  0.000  +0.560  ......................|.................
  [0.60,0.67)   760  0.649  0.622  +0.027  #########################.|.............
  [0.67,0.73)  1091  0.692  0.677  +0.015  ###########################.|...........
  [0.73,0.80)  2070  0.790  0.816  -0.025  ################################|.......
  [0.80,0.87)  4772  0.838  0.829  +0.009  #################################.|.....
  [0.87,0.93)   332  0.874  0.852  +0.021  ##################################.|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
