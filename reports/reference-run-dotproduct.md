# Reference run

`2500` training cases, `6` epochs, lr `0.01`, d_model `128`, layers `2`, option scoring `dot_product`, label noise `0.2`.

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1531 | 77 |
| 2 | 1.1419 | 76 |
| 3 | 1.1409 | 74 |
| 4 | 1.1406 | 76 |
| 5 | 1.1395 | 79 |
| 6 | 1.1398 | 77 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **3%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.3887 | 0.0174 | 0.0436 | 0.6486 | 5.6 | 7.2 | 122 |
| calibration/temperature-scaled | 6000 | 0.3887 | 0.0177 | 0.0441 | 0.6486 | 5.6 | 7.1 | 122 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0071 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0045 on this run, p95 0.0071
- FAIL accuracy_over_baseline: -0.0036 (limit 0.0500) -- model 0.3887 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS workhorse_ece: 0.0177 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0441 (limit 0.0500)

**Blocked**: accuracy_over_baseline.

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.389 | 0.394 | +0.005 | 0.0177 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0036 (95th percentile 0.0063), simulated over 60 resamples. The measured ECE is 0.0174.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27) 12000  0.264  0.249  +0.015  ##########.|............................
  [0.60,0.67)  6000  0.644  0.667  -0.023  ##########################|.............
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
