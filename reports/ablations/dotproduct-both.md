# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 2500 --eval-n 6000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 100 --option-scoring dot_product --match-normalize --match-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1766 | 55 |
| 2 | 1.0495 | 59 |
| 3 | 1.0410 | 58 |
| 4 | 1.0396 | 58 |
| 5 | 1.0368 | 56 |
| 6 | 1.0359 | 56 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **20%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+bc86a568

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4561 | 0.0214 | 0.0356 | 0.5951 | 4.0 | 6.2 | 122 |
| calibration/temperature-scaled | 6000 | 0.4561 | 0.0194 | 0.0357 | 0.5949 | 4.0 | 5.8 | 122 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0091 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0059 on this run, p95 0.0091
- PASS accuracy_over_baseline: 0.0639 (limit 0.0500) -- model 0.4561 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS workhorse_ece: 0.0194 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0357 (limit 0.0500)

All gates passed.

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.456 | 0.463 | +0.007 | 0.0194 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0059 (95th percentile 0.0092), simulated over 100 resamples. The measured ECE is 0.0214.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  4482  0.264  0.252  +0.012  ##########.|............................
  [0.27,0.33)  1518  0.272  0.248  +0.023  ##########.|............................
  [0.33,0.40)  4482  0.344  0.315  +0.029  #############.|.........................
  [0.60,0.67)  6000  0.641  0.667  -0.026  ##########################|.............
  [0.80,0.87)  1518  0.855  0.848  +0.007  ##################################|.....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
