# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 1200 --eval-n 2000 --epochs 3 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 2 --floor-trials 20 --option-scoring auto --validation-fraction 0.0
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0962 | 19 |
| 2 | 1.0779 | 20 |
| 3 | 1.0692 | 19 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **14%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+dc2bef85

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 2000 | 0.4470 | 0.0095 | 0.0560 | 0.6112 | 3.9 | 6.2 | 117 |
| calibration/temperature-scaled | 2000 | 0.4470 | 0.0095 | 0.0582 | 0.6113 | 3.9 | 5.5 | 117 |

## Release gates

- PASS sample_size: 6000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0127 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0087 on this run, p95 0.0127
- PASS accuracy_over_baseline: 0.0517 (limit 0.0500) -- model 0.4470 vs marginal predictor 0.3953; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0215 (limit 0.0500) -- worst is 'size' at 0.2420 vs its own marginal 0.2635; the pooled gate hides this
- PASS workhorse_ece: 0.0095 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0582 (limit 0.0500)

**Blocked**: workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 2,000 | 0.2420 | 0.2635 | -0.0215 ⚠ |
| `at_risk` | 2,000 | 0.6640 | 0.6640 | +0.0000 |
| `plan` | 2,000 | 0.4350 | 0.2585 | +0.1765 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 6000 | 0.447 | 0.445 | -0.002 | 0.0095 |

## Is this number evidence?

On these 6000 predictions a perfectly calibrated model scores a mean ECE of 0.0086 (95th percentile 0.0123), simulated over 20 resamples. The measured ECE is 0.0095.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  3509  0.259  0.270  -0.011  ##########|.............................
  [0.60,0.67)  1509  0.666  0.670  -0.004  ###########################|............
  [0.67,0.73)   491  0.667  0.646  +0.021  ##########################.|............
  [0.80,0.87)   491  0.834  0.829  +0.005  #################################|......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
