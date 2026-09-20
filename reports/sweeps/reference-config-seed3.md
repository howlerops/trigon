# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 1200 --eval-n 2000 --epochs 3 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 3 --floor-trials 20 --option-scoring auto --validation-fraction 0.0
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1283 | 19 |
| 2 | 1.0487 | 19 |
| 3 | 1.0438 | 18 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **19%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+492dfaa5

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 2000 | 0.4483 | 0.0160 | 0.0661 | 0.6047 | 3.9 | 5.5 | 117 |
| calibration/temperature-scaled | 2000 | 0.4483 | 0.0223 | 0.0688 | 0.6054 | 3.8 | 5.6 | 117 |

## Release gates

- PASS sample_size: 6000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0133 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0081 on this run, p95 0.0133
- PASS accuracy_over_baseline: 0.0510 (limit 0.0500) -- model 0.4483 vs marginal predictor 0.3973; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0300 (limit 0.0500) -- worst is 'size' at 0.2475 vs its own marginal 0.2775; the pooled gate hides this
- PASS workhorse_ece: 0.0223 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0688 (limit 0.0500)

**Blocked**: workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 2,000 | 0.2475 | 0.2775 | -0.0300 ⚠ |
| `at_risk` | 2,000 | 0.6525 | 0.6525 | +0.0000 |
| `plan` | 2,000 | 0.4450 | 0.2620 | +0.1830 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 6000 | 0.448 | 0.460 | +0.011 | 0.0223 |

## Is this number evidence?

On these 6000 predictions a perfectly calibrated model scores a mean ECE of 0.0074 (95th percentile 0.0133), simulated over 20 resamples. The measured ECE is 0.0160.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.27,0.33)  3504  0.294  0.281  +0.013  ###########.|...........................
  [0.60,0.67)  2000  0.633  0.652  -0.019  #########################|..............
  [0.80,0.87)   496  0.830  0.806  +0.024  ################################.|......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
