# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9966 | 94 |
| 2 | 0.9429 | 140 |
| 3 | 0.9512 | 99 |
| 4 | 0.9222 | 79 |
| 5 | 0.9172 | 80 |
| 6 | 0.9155 | 81 |
| 7 | 0.9139 | 82 |
| 8 | 0.9125 | 92 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **41%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+071c60fd, trigon-reference-0.1.0+071c60fd+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5462 | 0.0144 | 0.0391 | 0.5154 | 4.5 | 14.8 | 144 |
| calibration/temperature-scaled | 6000 | 0.5768 | 0.0226 | 0.0230 | 0.5063 | 4.3 | 7.4 | 144 |
| calibration/int8 | 6000 | 0.5742 | 0.0168 | 0.0262 | 0.5113 | 4.3 | 7.3 | 144 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0112 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0082 on this run, p95 0.0112
- PASS accuracy_over_baseline: 0.1828 (limit 0.0500) -- model 0.5768 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0290 (limit 0.0500) -- worst is 'size' at 0.2848 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0226 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0230 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0253 (limit 0.0500) -- worst is 'score' at ECE 0.0253 (overconfidence +0.018); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0058 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2438 | 0.2558 | -0.0120 ⚠ |
| `at_risk` | 6,000 | 0.7535 | 0.6687 | +0.0848 |
| `plan` | 6,000 | 0.6413 | 0.2577 | +0.3837 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.577 | 0.586 | +0.009 | 0.0226 | 0.0230 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.692 | 0.702 | +0.010 | 0.0234 | 0.0234 |
| noul | 6000 | 0.753 | 0.754 | +0.000 | 0.0189 | 0.0268 |
| score | 6000 | 0.285 | 0.303 | +0.018 | 0.0253 | 0.0423 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0071 (95th percentile 0.0104), simulated over 40 resamples. The measured ECE is 0.0144.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  5565  0.261  0.243  +0.018  ##########|.............................
  [0.27,0.33)   435  0.275  0.255  +0.020  ##########.|............................
  [0.40,0.47)  3012  0.442  0.436  +0.006  #################.|.....................
  [0.60,0.67)   886  0.616  0.644  -0.028  #########################|..............
  [0.67,0.73)   863  0.675  0.647  +0.028  ##########################.|............
  [0.73,0.80)  2174  0.785  0.800  -0.014  ###############################|........
  [0.80,0.87)  4559  0.834  0.825  +0.008  #################################|......
  [0.87,0.93)   506  0.872  0.840  +0.032  ##################################.|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
