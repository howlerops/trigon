# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 3 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0574 | 83 |
| 2 | 1.0797 | 99 |
| 3 | 1.0311 | 91 |
| 4 | 1.0286 | 94 |
| 5 | 1.0282 | 80 |
| 6 | 1.0272 | 80 |
| 7 | 1.0160 | 80 |
| 8 | 0.9923 | 82 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **27%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+f86d3ebe, trigon-reference-0.1.0+f86d3ebe+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4838 | 0.0191 | 0.0209 | 0.5746 | 4.1 | 11.0 | 144 |
| calibration/temperature-scaled | 6000 | 0.4829 | 0.0180 | 0.0192 | 0.5752 | 4.4 | 13.6 | 144 |
| calibration/int8 | 6000 | 0.4833 | 0.0178 | 0.0190 | 0.5751 | 4.1 | 8.2 | 144 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0100 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0061 on this run, p95 0.0100
- PASS accuracy_over_baseline: 0.0889 (limit 0.0500) -- model 0.4829 vs marginal predictor 0.3940; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0140 (limit 0.0500) -- worst is 'size' at 0.2520 vs its own marginal 0.2660; the pooled gate hides this
- PASS workhorse_ece: 0.0180 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0192 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0266 (limit 0.0500) -- worst is 'noul' at ECE 0.0266 (overconfidence +0.009); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0002 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2520 | 0.2660 | -0.0140 ⚠ |
| `at_risk` | 6,000 | 0.7513 | 0.6573 | +0.0940 |
| `plan` | 6,000 | 0.4482 | 0.2587 | +0.1895 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.483 | 0.493 | +0.010 | 0.0180 | 0.0192 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.448 | 0.460 | +0.012 | 0.0179 | 0.0264 |
| noul | 6000 | 0.749 | 0.758 | +0.009 | 0.0266 | 0.0266 |
| score | 6000 | 0.252 | 0.261 | +0.009 | 0.0095 | 0.0206 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0062 (95th percentile 0.0099), simulated over 40 resamples. The measured ECE is 0.0191.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.261  0.252  +0.009  ##########|.............................
  [0.27,0.33)  4134  0.324  0.328  -0.004  #############|..........................
  [0.33,0.40)   366  0.334  0.306  +0.028  ############.|..........................
  [0.53,0.60)   267  0.593  0.655  -0.063  ########################|#..............
  [0.60,0.67)  1513  0.636  0.659  -0.023  #########################|..............
  [0.80,0.87)  4677  0.822  0.790  +0.032  ################################.|......
  [0.87,0.93)  1043  0.883  0.826  +0.056  #################################..|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
