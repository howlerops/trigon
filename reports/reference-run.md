# Reference run

`2500` training cases, `6` epochs, lr `0.01`, d_model `128`, layers `2`, option scoring `readout_per_option`, label noise `0.2`.

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1578 | 88 |
| 2 | 1.0564 | 95 |
| 3 | 1.0429 | 101 |
| 4 | 1.0405 | 101 |
| 5 | 1.0372 | 99 |
| 6 | 1.0366 | 97 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **20%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4561 | 0.0112 | 0.0155 | 0.5947 | 6.4 | 8.1 | 125 |
| calibration/temperature-scaled | 6000 | 0.4561 | 0.0111 | 0.0158 | 0.5947 | 6.5 | 8.1 | 125 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0080 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0049 on this run, p95 0.0080
- PASS accuracy_over_baseline: 0.0639 (limit 0.0500) -- model 0.4561 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS workhorse_ece: 0.0111 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0158 (limit 0.0500)

All gates passed.

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.456 | 0.455 | -0.002 | 0.0111 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0051 (95th percentile 0.0075), simulated over 60 resamples. The measured ECE is 0.0112.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.254  0.251  +0.003  ##########|.............................
  [0.27,0.33)  4482  0.322  0.315  +0.007  #############|..........................
  [0.60,0.67)  6000  0.644  0.667  -0.024  ##########################|.............
  [0.80,0.87)  1518  0.857  0.848  +0.008  ##################################|.....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
