# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 3 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-readout-per-level
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9515 | 96 |
| 2 | 0.9875 | 102 |
| 3 | 1.0423 | 124 |
| 4 | 1.1059 | 107 |
| 5 | 1.1055 | 118 |
| 6 | 1.1048 | 118 |
| 7 | 1.1045 | 105 |
| 8 | 1.1043 | 100 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **9%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+6c4b75e9, trigon-reference-0.1.0+6c4b75e9+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5486 | 0.0140 | 0.0526 | 0.5272 | 4.2 | 6.7 | 147 |
| calibration/temperature-scaled | 6000 | 0.5884 | 0.0122 | 0.0355 | 0.5020 | 4.1 | 7.2 | 147 |
| calibration/int8 | 6000 | 0.5883 | 0.0173 | 0.0512 | 0.5043 | 4.1 | 6.4 | 147 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0072 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0052 on this run, p95 0.0072
- PASS accuracy_over_baseline: 0.1944 (limit 0.0500) -- model 0.5884 vs marginal predictor 0.3940; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0112 (limit 0.0500) -- worst is 'size' at 0.2548 vs its own marginal 0.2660; the pooled gate hides this
- PASS workhorse_ece: 0.0122 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0355 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0360 (limit 0.0500) -- worst is 'choice' at ECE 0.0360 (overconfidence +0.009); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0050 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2548 | 0.2660 | -0.0112 ⚠ |
| `at_risk` | 6,000 | 0.7513 | 0.6573 | +0.0940 |
| `plan` | 6,000 | 0.6395 | 0.2587 | +0.3808 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.588 | 0.595 | +0.007 | 0.0122 | 0.0355 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.766 | 0.775 | +0.009 | 0.0360 | 0.0714 |
| noul | 6000 | 0.745 | 0.759 | +0.014 | 0.0317 | 0.0311 |
| score | 6000 | 0.255 | 0.252 | -0.003 | 0.0027 | 0.0544 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0064 (95th percentile 0.0098), simulated over 40 resamples. The measured ECE is 0.0140.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.252  0.255  -0.003  ##########|.............................
  [0.47,0.53)  3016  0.489  0.453  +0.036  ##################..|...................
  [0.60,0.67)   792  0.660  0.654  +0.006  ##########################|.............
  [0.67,0.73)   451  0.716  0.652  +0.064  ##########################...|..........
  [0.73,0.80)  3740  0.764  0.771  -0.007  ###############################|........
  [0.80,0.87)  3469  0.819  0.824  -0.005  #################################|......
  [0.87,0.93)   532  0.889  0.797  +0.092  ################################....|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
