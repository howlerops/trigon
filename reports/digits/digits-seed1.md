# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0000 | 105 |
| 2 | 1.0080 | 103 |
| 3 | 1.0206 | 102 |
| 4 | 1.0127 | 105 |
| 5 | 1.0083 | 110 |
| 6 | 1.0142 | 106 |
| 7 | 1.0341 | 104 |
| 8 | 1.0336 | 100 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **20%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+8cd66aff, trigon-reference-0.1.0+8cd66aff+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5198 | 0.0296 | 0.0452 | 0.5595 | 3.8 | 5.3 | 121 |
| calibration/temperature-scaled | 6000 | 0.5212 | 0.0365 | 0.0396 | 0.5577 | 3.8 | 5.1 | 121 |
| calibration/int8 | 6000 | 0.5057 | 0.0505 | 0.0480 | 0.5642 | 3.8 | 5.5 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0099 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0064 on this run, p95 0.0099
- PASS accuracy_over_baseline: 0.1271 (limit 0.0500) -- model 0.5212 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0088 (limit 0.0500) -- worst is 'size' at 0.2470 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0365 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0396 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0487 (limit 0.0500) -- worst is 'noul' at ECE 0.0487 (overconfidence +0.049); pooled ECE cancels heads that err in opposite directions
- FAIL quantization_ece_delta: 0.0141 (limit 0.0100)

**Blocked**: quantization_ece_delta.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2447 | 0.2558 | -0.0112 ⚠ |
| `at_risk` | 6,000 | 0.6687 | 0.6687 | +0.0000 |
| `plan` | 6,000 | 0.6462 | 0.2577 | +0.3885 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.521 | 0.557 | +0.035 | 0.0365 | 0.0396 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.646 | 0.672 | +0.026 | 0.0371 | 0.0404 |
| noul | 6000 | 0.670 | 0.719 | +0.049 | 0.0487 | 0.0487 |
| score | 6000 | 0.247 | 0.279 | +0.032 | 0.0345 | 0.0324 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0055 (95th percentile 0.0087), simulated over 40 resamples. The measured ECE is 0.0296.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.254  0.245  +0.010  ##########|.............................
  [0.33,0.40)  2988  0.381  0.446  -0.066  ###############|##......................
  [0.60,0.67)  6000  0.653  0.669  -0.016  ##########################|.............
  [0.73,0.80)  1459  0.749  0.846  -0.096  ##############################|###......
  [0.80,0.87)  1553  0.817  0.844  -0.027  #################################|......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
