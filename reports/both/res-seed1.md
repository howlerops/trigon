# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 1 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0117 | 113 |
| 2 | 1.0089 | 99 |
| 3 | 1.0044 | 97 |
| 4 | 1.0039 | 103 |
| 5 | 1.0017 | 104 |
| 6 | 1.0010 | 102 |
| 7 | 1.0003 | 101 |
| 8 | 0.9994 | 100 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **26%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+94d1d47e, trigon-reference-0.1.0+94d1d47e+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4817 | 0.0090 | 0.0121 | 0.5689 | 3.6 | 4.7 | 121 |
| calibration/temperature-scaled | 6000 | 0.4813 | 0.0148 | 0.0181 | 0.5698 | 3.6 | 4.5 | 121 |
| calibration/int8 | 6000 | 0.4813 | 0.0165 | 0.0193 | 0.5718 | 3.6 | 4.3 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0088 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0056 on this run, p95 0.0088
- PASS accuracy_over_baseline: 0.0873 (limit 0.0500) -- model 0.4813 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0133 (limit 0.0500) -- worst is 'size' at 0.2425 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0148 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0181 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0299 (limit 0.0500) -- worst is 'score' at ECE 0.0299 (overconfidence +0.030); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0017 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2435 | 0.2558 | -0.0123 ⚠ |
| `at_risk` | 6,000 | 0.7535 | 0.6687 | +0.0848 |
| `plan` | 6,000 | 0.4480 | 0.2577 | +0.1903 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.481 | 0.493 | +0.012 | 0.0148 | 0.0181 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.448 | 0.454 | +0.006 | 0.0064 | 0.0214 |
| noul | 6000 | 0.753 | 0.753 | -0.000 | 0.0080 | 0.0244 |
| score | 6000 | 0.242 | 0.272 | +0.030 | 0.0299 | 0.0319 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0059 (95th percentile 0.0078), simulated over 40 resamples. The measured ECE is 0.0090.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.256  0.243  +0.013  ##########|.............................
  [0.27,0.33)  4519  0.325  0.317  +0.007  #############|..........................
  [0.60,0.67)  1703  0.643  0.642  +0.000  ##########################|.............
  [0.67,0.73)    46  0.667  0.761  -0.094  ###########################|##..........
  [0.73,0.80)  2674  0.793  0.801  -0.008  ################################|.......
  [0.80,0.87)  3058  0.828  0.819  +0.009  #################################|......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
