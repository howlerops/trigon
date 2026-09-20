# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 2500 --eval-n 6000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 100 --option-scoring dot_product --match-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1863 | 53 |
| 2 | 1.1451 | 52 |
| 3 | 1.0778 | 57 |
| 4 | 0.9086 | 58 |
| 5 | 0.8859 | 56 |
| 6 | 0.8772 | 56 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **47%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+929398dc

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5889 | 0.0211 | 0.0243 | 0.4906 | 3.9 | 6.2 | 122 |
| calibration/temperature-scaled | 6000 | 0.5889 | 0.0184 | 0.0227 | 0.4904 | 3.9 | 8.5 | 122 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0094 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0056 on this run, p95 0.0094
- PASS accuracy_over_baseline: 0.1967 (limit 0.0500) -- model 0.5889 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS workhorse_ece: 0.0184 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0227 (limit 0.0500)

All gates passed.

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.589 | 0.588 | -0.001 | 0.0184 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0054 (95th percentile 0.0089), simulated over 100 resamples. The measured ECE is 0.0211.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  1473  0.261  0.239  +0.022  ##########|.............................
  [0.27,0.33)  4527  0.274  0.255  +0.019  ##########.|............................
  [0.60,0.67)  4498  0.628  0.667  -0.039  #########################|#.............
  [0.67,0.73)  1502  0.691  0.668  +0.023  ###########################.|...........
  [0.73,0.80)     2  0.799  0.500  +0.299  ####################............|.......
  [0.80,0.87)  5998  0.840  0.848  -0.009  ##################################|.....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
