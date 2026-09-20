# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 1200 --eval-n 2000 --epochs 3 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 20 --option-scoring auto --validation-fraction 0.0
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.1092 | 19 |
| 2 | 0.9801 | 20 |
| 3 | 0.9689 | 19 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **31%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+5c32a846

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 2000 | 0.5225 | 0.0186 | 0.0626 | 0.5445 | 3.9 | 6.0 | 117 |
| calibration/temperature-scaled | 2000 | 0.5225 | 0.0122 | 0.0635 | 0.5441 | 3.9 | 5.6 | 117 |

## Release gates

- PASS sample_size: 6000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0120 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0088 on this run, p95 0.0120
- PASS accuracy_over_baseline: 0.1213 (limit 0.0500) -- model 0.5225 vs marginal predictor 0.4012; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0215 (limit 0.0500) -- worst is 'size' at 0.2475 vs its own marginal 0.2690; the pooled gate hides this
- PASS workhorse_ece: 0.0122 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0635 (limit 0.0500)

**Blocked**: workhorse_adaptive_ece.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 2,000 | 0.2475 | 0.2690 | -0.0215 ⚠ |
| `at_risk` | 2,000 | 0.6695 | 0.6695 | +0.0000 |
| `plan` | 2,000 | 0.6505 | 0.2650 | +0.3855 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 6000 | 0.522 | 0.518 | -0.004 | 0.0122 |

## Is this number evidence?

On these 6000 predictions a perfectly calibrated model scores a mean ECE of 0.0083 (95th percentile 0.0115), simulated over 20 resamples. The measured ECE is 0.0186.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  2000  0.253  0.247  +0.006  ##########|.............................
  [0.40,0.47)   986  0.420  0.458  -0.038  #################|......................
  [0.60,0.67)  2000  0.640  0.669  -0.029  ##########################|.............
  [0.80,0.87)  1014  0.835  0.837  -0.003  #################################|......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
