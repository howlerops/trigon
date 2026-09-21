# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 4000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 256 --layers 4 --noise 0.2 --seed 0 --floor-trials 20 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9648 | 335 |
| 2 | 1.0343 | 463 |
| 3 | 1.1238 | 634 |
| 4 | 1.1238 | 493 |
| 5 | 1.1092 | 540 |
| 6 | 1.0784 | 410 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **13%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+b110633e, trigon-reference-0.1.0+b110633e+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 4000 | 0.4569 | 0.1461 | 0.1351 | 0.6144 | 8.3 | 11.1 | 121 |
| calibration/temperature-scaled | 4000 | 0.5202 | 0.0288 | 0.0334 | 0.5670 | 8.3 | 10.4 | 121 |
| calibration/int8 | 4000 | 0.5202 | 0.0327 | 0.0359 | 0.5696 | 8.4 | 10.8 | 121 |

## Release gates

- PASS sample_size: 12000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0104 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0067 on this run, p95 0.0104
- PASS accuracy_over_baseline: 0.1281 (limit 0.0500) -- model 0.5202 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0055 (limit 0.0500) -- worst is 'size' at 0.2510 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0288 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0334 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0625 (limit 0.0500) -- worst is 'choice' at ECE 0.0625 (overconfidence -0.061); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0039 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline, worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 4,000 | 0.2510 | 0.2565 | -0.0055 ⚠ |
| `at_risk` | 4,000 | 0.6660 | 0.6660 | +0.0000 |
| `plan` | 4,000 | 0.4537 | 0.2540 | +0.1997 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 12000 | 0.520 | 0.493 | -0.028 | 0.0288 | 0.0334 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 4000 | 0.644 | 0.583 | -0.061 | 0.0625 | 0.0625 |
| noul | 4000 | 0.666 | 0.644 | -0.022 | 0.0251 | 0.0251 |
| score | 4000 | 0.251 | 0.250 | -0.001 | 0.0007 | 0.0245 |

## Is this number evidence?

On these 12000 predictions a perfectly calibrated model scores a mean ECE of 0.0080 (95th percentile 0.0114), simulated over 20 resamples. The measured ECE is 0.1461.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  4000  0.250  0.251  -0.001  ##########|.............................
  [0.33,0.40)  1006  0.373  0.078  +0.295  ###............|........................
  [0.40,0.47)   961  0.406  0.856  -0.451  ################|#################......
  [0.53,0.60)   688  0.595  0.837  -0.242  ########################|########.......
  [0.60,0.67)  1345  0.643  0.251  +0.392  ##########................|.............
  [0.67,0.73)   319  0.732  0.661  +0.071  ##########################...|..........
  [0.73,0.80)  3681  0.749  0.666  +0.082  ###########################...|.........
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
