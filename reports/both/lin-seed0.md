# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-head linear
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9138 | 118 |
| 2 | 0.9431 | 118 |
| 3 | 1.0015 | 127 |
| 4 | 0.9283 | 128 |
| 5 | 0.9133 | 114 |
| 6 | 0.8766 | 138 |
| 7 | 0.8482 | 233 |
| 8 | 0.8421 | 248 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **52%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+c299aa4a, trigon-reference-0.1.0+c299aa4a+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6167 | 0.0820 | 0.0792 | 0.4712 | 3.6 | 4.9 | 121 |
| calibration/temperature-scaled | 6000 | 0.6167 | 0.0567 | 0.0557 | 0.4654 | 3.5 | 4.1 | 121 |
| calibration/int8 | 6000 | 0.6167 | 0.0573 | 0.0565 | 0.4656 | 3.5 | 4.6 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0087 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0058 on this run, p95 0.0087
- PASS accuracy_over_baseline: 0.2245 (limit 0.0500) -- model 0.6167 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0172 (limit 0.0500) -- worst is 'size' at 0.2393 vs its own marginal 0.2565; the pooled gate hides this
- FAIL workhorse_ece: 0.0567 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0557 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0691 (limit 0.0500) -- worst is 'score' at ECE 0.0691 (overconfidence +0.069); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0006 (limit 0.0100)

**Blocked**: workhorse_ece, workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline, worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2393 | 0.2565 | -0.0172 ⚠ |
| `at_risk` | 6,000 | 0.7627 | 0.6672 | +0.0955 |
| `plan` | 6,000 | 0.8482 | 0.2530 | +0.5952 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.617 | 0.653 | +0.036 | 0.0567 | 0.0557 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.848 | 0.867 | +0.019 | 0.0418 | 0.0196 |
| noul | 6000 | 0.763 | 0.784 | +0.021 | 0.0591 | 0.0562 |
| score | 6000 | 0.239 | 0.308 | +0.069 | 0.0691 | 0.0691 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0065 (95th percentile 0.0090), simulated over 40 resamples. The measured ECE is 0.0820.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.27,0.33)  6000  0.308  0.239  +0.069  ##########..|...........................
  [0.53,0.60)  1241  0.585  0.628  -0.043  #######################|#...............
  [0.60,0.67)  1024  0.630  0.772  -0.142  #########################|#####.........
  [0.67,0.73)  1902  0.703  0.838  -0.135  ############################|#####......
  [0.73,0.80)  3684  0.753  0.852  -0.099  ##############################|###......
  [0.80,0.87)  1774  0.860  0.815  +0.046  #################################.|.....
  [0.87,0.93)  2375  0.875  0.808  +0.067  ################################...|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
