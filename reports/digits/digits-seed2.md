# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 2 --floor-trials 40 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 1.0305 | 111 |
| 2 | 1.1080 | 104 |
| 3 | 1.1067 | 108 |
| 4 | 1.1058 | 104 |
| 5 | 1.1057 | 105 |
| 6 | 1.1048 | 113 |
| 7 | 1.1044 | 118 |
| 8 | 1.1042 | 116 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **9%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+064615c1, trigon-reference-0.1.0+064615c1+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.4206 | 0.0051 | 0.0221 | 0.6223 | 3.6 | 4.8 | 121 |
| calibration/temperature-scaled | 6000 | 0.4338 | 0.0136 | 0.0203 | 0.6216 | 3.6 | 4.7 | 121 |
| calibration/int8 | 6000 | 0.4167 | 0.0203 | 0.0346 | 0.6252 | 3.6 | 4.5 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0081 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0057 on this run, p95 0.0081
- FAIL accuracy_over_baseline: 0.0423 (limit 0.0500) -- model 0.4338 vs marginal predictor 0.3915; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0022 (limit 0.0500) -- worst is 'size' at 0.2592 vs its own marginal 0.2613; the pooled gate hides this
- PASS workhorse_ece: 0.0136 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0203 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0292 (limit 0.0500) -- worst is 'choice' at ECE 0.0292 (overconfidence -0.015); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0067 (limit 0.0100)

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `plan` | 6,000 | 0.2522 | 0.2547 | -0.0025 ⚠ |
| `size` | 6,000 | 0.2592 | 0.2613 | -0.0022 ⚠ |
| `at_risk` | 6,000 | 0.7503 | 0.6585 | +0.0918 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.434 | 0.432 | -0.002 | 0.0136 | 0.0203 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.292 | 0.277 | -0.015 | 0.0292 | 0.0626 |
| noul | 6000 | 0.750 | 0.758 | +0.007 | 0.0136 | 0.0224 |
| score | 6000 | 0.259 | 0.261 | +0.002 | 0.0019 | 0.0137 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0046 (95th percentile 0.0071), simulated over 40 resamples. The measured ECE is 0.0051.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27) 12000  0.257  0.256  +0.001  ##########|.............................
  [0.60,0.67)  1783  0.644  0.655  -0.011  ##########################|.............
  [0.73,0.80)    41  0.798  0.780  +0.017  ###############################.|.......
  [0.80,0.87)  4176  0.806  0.791  +0.015  ################################|.......
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
