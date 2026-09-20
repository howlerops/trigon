# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 2500 --eval-n 6000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 100 --option-scoring dot_product --match-normalize
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1582 | 52 |
| 2 | 1.1426 | 52 |
| 3 | 1.1412 | 56 |
| 4 | 1.1406 | 55 |
| 5 | 1.1395 | 54 |
| 6 | 1.1398 | 54 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **3%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+4f28dd5f

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.3887 | 0.0201 | 0.0330 | 0.6487 | 4.0 | 6.2 | 122 |
| calibration/temperature-scaled | 6000 | 0.3887 | 0.0184 | 0.0326 | 0.6486 | 4.0 | 8.5 | 122 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0076 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0040 on this run, p95 0.0076
- FAIL accuracy_over_baseline: -0.0036 (limit 0.0500) -- model 0.3887 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS workhorse_ece: 0.0184 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0326 (limit 0.0500)

**Blocked**: accuracy_over_baseline.

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.389 | 0.395 | +0.006 | 0.0184 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0040 (95th percentile 0.0075), simulated over 100 resamples. The measured ECE is 0.0201.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.27,0.33) 12000  0.267  0.249  +0.018  ##########.|............................
  [0.60,0.67)  6000  0.642  0.667  -0.025  ##########################|.............
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
