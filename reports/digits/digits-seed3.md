# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 3 --floor-trials 40 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.8804 | 106 |
| 2 | 0.9381 | 105 |
| 3 | 1.1068 | 100 |
| 4 | 1.1055 | 99 |
| 5 | 1.1055 | 101 |
| 6 | 1.1048 | 102 |
| 7 | 1.1045 | 100 |
| 8 | 1.1043 | 100 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **9%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+1c126875, trigon-reference-0.1.0+1c126875+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6114 | 0.0422 | 0.0472 | 0.4801 | 3.6 | 4.9 | 121 |
| calibration/temperature-scaled | 6000 | 0.6114 | 0.0422 | 0.0472 | 0.4801 | 3.7 | 5.1 | 121 |
| calibration/int8 | 6000 | 0.6114 | 0.0421 | 0.0465 | 0.4801 | 3.6 | 5.0 | 121 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0068 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0042 on this run, p95 0.0068
- PASS accuracy_over_baseline: 0.2174 (limit 0.0500) -- model 0.6114 vs marginal predictor 0.3940; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0128 (limit 0.0500) -- worst is 'size' at 0.2532 vs its own marginal 0.2660; the pooled gate hides this
- PASS workhorse_ece: 0.0422 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0472 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0946 (limit 0.0500) -- worst is 'choice' at ECE 0.0946 (overconfidence +0.095); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0001 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline, worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2532 | 0.2660 | -0.0128 ⚠ |
| `at_risk` | 6,000 | 0.7513 | 0.6573 | +0.0940 |
| `plan` | 6,000 | 0.8298 | 0.2587 | +0.5712 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.611 | 0.652 | +0.041 | 0.0422 | 0.0472 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.830 | 0.924 | +0.095 | 0.0946 | 0.0946 |
| noul | 6000 | 0.751 | 0.781 | +0.030 | 0.0295 | 0.0443 |
| score | 6000 | 0.253 | 0.251 | -0.002 | 0.0025 | 0.0190 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0042 (95th percentile 0.0068), simulated over 40 resamples. The measured ECE is 0.0422.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.251  0.253  -0.002  ##########|.............................
  [0.73,0.80)  6000  0.781  0.751  +0.030  ##############################.|........
  [0.87,0.93)  4388  0.916  0.833  +0.083  #################################....|..
  [0.93,1.00)  1612  0.947  0.820  +0.127  #################################.....|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
