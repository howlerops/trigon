# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9312 | 141 |
| 2 | 0.9506 | 128 |
| 3 | 1.0230 | 149 |
| 4 | 1.0167 | 157 |
| 5 | 1.0078 | 134 |
| 6 | 1.0044 | 149 |
| 7 | 1.0030 | 156 |
| 8 | 1.0016 | 136 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **26%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+585308ce

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5866 | 0.0431 | 0.0822 | 0.4943 | 4.0 | 8.5 | 117 |
| calibration/temperature-scaled | 6000 | 0.5866 | 0.0677 | 0.0682 | 0.4914 | 4.0 | 6.7 | 117 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0094 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0059 on this run, p95 0.0094
- PASS accuracy_over_baseline: 0.1926 (limit 0.0500) -- model 0.5866 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0112 (limit 0.0500) -- worst is 'size' at 0.2447 vs its own marginal 0.2558; the pooled gate hides this
- FAIL workhorse_ece: 0.0677 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0682 (limit 0.0500)

**Blocked**: workhorse_ece, workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2447 | 0.2558 | -0.0112 ⚠ |
| `at_risk` | 6,000 | 0.6687 | 0.6687 | +0.0000 |
| `plan` | 6,000 | 0.8465 | 0.2577 | +0.5888 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.587 | 0.579 | -0.007 | 0.0677 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0053 (95th percentile 0.0079), simulated over 40 resamples. The measured ECE is 0.0431.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.253  0.245  +0.008  ##########|.............................
  [0.60,0.67)  6261  0.637  0.685  -0.049  #########################|#.............
  [0.67,0.73)  1246  0.691  0.804  -0.113  ############################|###........
  [0.73,0.80)  3006  0.777  0.845  -0.068  ###############################|##......
  [0.80,0.87)     6  0.801  0.833  -0.033  ################################|.......
  [0.87,0.93)  1481  0.900  0.847  +0.053  ##################################..|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
