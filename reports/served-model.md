# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 1200 --eval-n 6000 --epochs 5 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 100 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1569 | 27 |
| 2 | 1.1436 | 28 |
| 3 | 1.1423 | 28 |
| 4 | 1.1404 | 28 |
| 5 | 1.1393 | 27 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **3%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+87314780

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.3914 | 0.0115 | 0.0503 | 0.6485 | 3.8 | 4.6 | 125 |
| calibration/temperature-scaled | 6000 | 0.3914 | 0.0171 | 0.0503 | 0.6490 | 3.9 | 5.7 | 125 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0081 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0045 on this run, p95 0.0081
- FAIL accuracy_over_baseline: -0.0008 (limit 0.0500) -- model 0.3914 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS workhorse_ece: 0.0171 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0503 (limit 0.0500)

**Blocked**: accuracy_over_baseline, workhorse_adaptive_ece.

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.391 | 0.395 | +0.003 | 0.0171 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0039 (95th percentile 0.0081), simulated over 100 resamples. The measured ECE is 0.0115.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27) 12000  0.260  0.254  +0.007  ##########|.............................
  [0.60,0.67)  6000  0.647  0.667  -0.021  ##########################|.............
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
