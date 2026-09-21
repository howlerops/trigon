# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 4000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 256 --layers 4 --noise 0.2 --seed 1 --floor-trials 20 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0029 | 352 |
| 2 | 0.9940 | 429 |
| 3 | 1.0381 | 366 |
| 4 | 1.0090 | 386 |
| 5 | 1.0032 | 396 |
| 6 | 1.0013 | 388 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **26%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+a208a564, trigon-reference-0.1.0+a208a564+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 4000 | 0.5889 | 0.0490 | 0.0623 | 0.5185 | 8.4 | 10.1 | 121 |
| calibration/temperature-scaled | 4000 | 0.5923 | 0.0139 | 0.0202 | 0.4856 | 8.5 | 10.5 | 121 |
| calibration/int8 | 4000 | 0.5923 | 0.0244 | 0.0314 | 0.4909 | 8.4 | 11.8 | 121 |

## Release gates

- PASS sample_size: 12000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0081 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0052 on this run, p95 0.0081
- PASS accuracy_over_baseline: 0.1979 (limit 0.0500) -- model 0.5923 vs marginal predictor 0.3943; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0070 (limit 0.0500) -- worst is 'size' at 0.2512 vs its own marginal 0.2582; the pooled gate hides this
- PASS workhorse_ece: 0.0139 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0202 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0383 (limit 0.0500) -- worst is 'noul' at ECE 0.0383 (overconfidence +0.035); pooled ECE cancels heads that err in opposite directions
- FAIL quantization_ece_delta: 0.0105 (limit 0.0100)

**Blocked**: quantization_ece_delta.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 4,000 | 0.2512 | 0.2582 | -0.0070 ⚠ |
| `at_risk` | 4,000 | 0.6677 | 0.6677 | +0.0000 |
| `plan` | 4,000 | 0.8478 | 0.2570 | +0.5907 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 12000 | 0.592 | 0.605 | +0.012 | 0.0139 | 0.0202 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 4000 | 0.848 | 0.851 | +0.003 | 0.0047 | 0.0122 |
| noul | 4000 | 0.678 | 0.712 | +0.035 | 0.0383 | 0.0383 |
| score | 4000 | 0.251 | 0.251 | -0.000 | 0.0005 | 0.0278 |

## Is this number evidence?

On these 12000 predictions a perfectly calibrated model scores a mean ECE of 0.0059 (95th percentile 0.0089), simulated over 20 resamples. The measured ECE is 0.0490.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  4000  0.251  0.251  -0.000  ##########|.............................
  [0.40,0.47)  1008  0.442  0.849  -0.407  ##################|###############......
  [0.60,0.67)  1030  0.655  0.653  +0.002  ##########################|.............
  [0.67,0.73)  3077  0.672  0.678  -0.006  ###########################|............
  [0.73,0.80)   906  0.765  0.843  -0.078  ###############################|##......
  [0.80,0.87)   949  0.815  0.857  -0.041  #################################|......
  [0.87,0.93)  1030  0.888  0.844  +0.044  ##################################..|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
