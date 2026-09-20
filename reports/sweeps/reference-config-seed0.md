# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 1200 --eval-n 2000 --epochs 3 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 20 --option-scoring auto --validation-fraction 0.0
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1534 | 19 |
| 2 | 1.1427 | 19 |
| 3 | 1.1399 | 18 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **3%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+c26d5c29

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 2000 | 0.3903 | 0.0074 | 0.0448 | 0.6494 | 3.9 | 5.5 | 117 |
| calibration/temperature-scaled | 2000 | 0.3903 | 0.0151 | 0.0450 | 0.6498 | 3.9 | 6.9 | 117 |

## Release gates

- PASS sample_size: 6000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0135 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0061 on this run, p95 0.0135
- FAIL accuracy_over_baseline: 0.0000 (limit 0.0500) -- model 0.3903 vs marginal predictor 0.3903; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0000 (limit 0.0500) -- worst is 'at_risk' at 0.6625 vs its own marginal 0.6625; the pooled gate hides this
- PASS workhorse_ece: 0.0151 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0450 (limit 0.0500)

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `at_risk` | 2,000 | 0.6625 | 0.6625 | +0.0000 |
| `plan` | 2,000 | 0.2540 | 0.2540 | +0.0000 |
| `size` | 2,000 | 0.2545 | 0.2545 | +0.0000 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 6000 | 0.390 | 0.395 | +0.005 | 0.0151 |

## Is this number evidence?

On these 6000 predictions a perfectly calibrated model scores a mean ECE of 0.0082 (95th percentile 0.0128), simulated over 20 resamples. The measured ECE is 0.0074.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  2000  0.253  0.254  -0.001  ##########|.............................
  [0.27,0.33)  2000  0.270  0.255  +0.016  ##########.|............................
  [0.67,0.73)  2000  0.668  0.662  +0.006  ##########################.|............
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
