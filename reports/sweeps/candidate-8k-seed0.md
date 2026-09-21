# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9443 | 143 |
| 2 | 1.0715 | 143 |
| 3 | 1.1053 | 148 |
| 4 | 1.1051 | 145 |
| 5 | 1.1044 | 130 |
| 6 | 1.0862 | 147 |
| 7 | 1.0868 | 160 |
| 8 | 1.0457 | 131 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **18%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+6c907b98

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5537 | 0.0303 | 0.0364 | 0.5154 | 4.0 | 8.3 | 117 |
| calibration/temperature-scaled | 6000 | 0.5537 | 0.0102 | 0.0188 | 0.5125 | 3.9 | 8.7 | 117 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0093 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0061 on this run, p95 0.0093
- PASS accuracy_over_baseline: 0.1614 (limit 0.0500) -- model 0.5537 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0062 (limit 0.0500) -- worst is 'size' at 0.2503 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0102 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0188 (limit 0.0500)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2503 | 0.2565 | -0.0062 ⚠ |
| `at_risk` | 6,000 | 0.7627 | 0.6672 | +0.0955 |
| `plan` | 6,000 | 0.6480 | 0.2530 | +0.3950 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.554 | 0.548 | -0.005 | 0.0102 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0062 (95th percentile 0.0094), simulated over 40 resamples. The measured ECE is 0.0303.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.254  0.250  +0.004  ##########|.............................
  [0.40,0.47)  3025  0.434  0.450  -0.016  #################|......................
  [0.67,0.73)  1851  0.680  0.655  +0.025  ##########################.|............
  [0.73,0.80)  2091  0.762  0.847  -0.085  ##############################|###......
  [0.80,0.87)   884  0.843  0.854  -0.011  ##################################|.....
  [0.87,0.93)  4149  0.869  0.811  +0.058  ################################...|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
