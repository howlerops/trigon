# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 3 --floor-trials 40 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9073 | 140 |
| 2 | 0.8377 | 133 |
| 3 | 0.8359 | 129 |
| 4 | 0.8334 | 126 |
| 5 | 0.8314 | 125 |
| 6 | 0.8290 | 125 |
| 7 | 0.8278 | 125 |
| 8 | 0.8265 | 126 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **55%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+94e16f25, trigon-reference-0.1.0+94e16f25+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6081 | 0.0121 | 0.0188 | 0.4734 | 3.9 | 6.3 | 117 |
| calibration/temperature-scaled | 6000 | 0.6081 | 0.0121 | 0.0188 | 0.4734 | 4.0 | 6.4 | 117 |
| calibration/int8 | 6000 | 0.6081 | 0.0122 | 0.0190 | 0.4734 | 4.0 | 6.2 | 117 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0088 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0053 on this run, p95 0.0088
- PASS accuracy_over_baseline: 0.2141 (limit 0.0500) -- model 0.6081 vs marginal predictor 0.3940; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0230 (limit 0.0500) -- worst is 'size' at 0.2430 vs its own marginal 0.2660; the pooled gate hides this
- PASS workhorse_ece: 0.0121 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0188 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0217 (limit 0.0500) -- worst is 'choice' at ECE 0.0217 (overconfidence +0.011); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0001 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2430 | 0.2660 | -0.0230 ⚠ |
| `at_risk` | 6,000 | 0.7513 | 0.6573 | +0.0940 |
| `plan` | 6,000 | 0.8298 | 0.2587 | +0.5712 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.608 | 0.617 | +0.009 | 0.0121 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.830 | 0.841 | +0.011 | 0.0217 |
| noul | 6000 | 0.751 | 0.758 | +0.007 | 0.0148 |
| score | 6000 | 0.243 | 0.252 | +0.009 | 0.0095 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0053 (95th percentile 0.0088), simulated over 40 resamples. The measured ECE is 0.0121.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.252  0.243  +0.009  ##########|.............................
  [0.60,0.67)  1780  0.645  0.658  -0.013  ##########################|.............
  [0.67,0.73)    10  0.704  1.000  -0.296  ############################|###########
  [0.80,0.87)  8713  0.818  0.813  +0.004  #################################|......
  [0.87,0.93)  1497  0.879  0.814  +0.066  #################################..|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
