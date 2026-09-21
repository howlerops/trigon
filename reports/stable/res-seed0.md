# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 0 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9719 | 84 |
| 2 | 1.0921 | 88 |
| 3 | 1.1050 | 87 |
| 4 | 1.0893 | 97 |
| 5 | 1.0700 | 90 |
| 6 | 1.0429 | 86 |
| 7 | 1.0262 | 85 |
| 8 | 1.0186 | 86 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **23%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+88cf3548, trigon-reference-0.1.0+88cf3548+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5505 | 0.0697 | 0.0689 | 0.5217 | 4.3 | 13.1 | 144 |
| calibration/temperature-scaled | 6000 | 0.5579 | 0.0412 | 0.0428 | 0.5132 | 4.3 | 13.0 | 144 |
| calibration/int8 | 6000 | 0.5582 | 0.0416 | 0.0445 | 0.5153 | 4.2 | 7.6 | 144 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0113 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0078 on this run, p95 0.0113
- PASS accuracy_over_baseline: 0.1657 (limit 0.0500) -- model 0.5579 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0130 (limit 0.0500) -- worst is 'size' at 0.2435 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0412 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0428 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0566 (limit 0.0500) -- worst is 'score' at ECE 0.0566 (overconfidence +0.057); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0004 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline, worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2415 | 0.2565 | -0.0150 ⚠ |
| `at_risk` | 6,000 | 0.7620 | 0.6672 | +0.0948 |
| `plan` | 6,000 | 0.6480 | 0.2530 | +0.3950 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.558 | 0.562 | +0.004 | 0.0412 | 0.0428 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.673 | 0.659 | -0.015 | 0.0285 | 0.0287 |
| noul | 6000 | 0.757 | 0.726 | -0.031 | 0.0406 | 0.0416 |
| score | 6000 | 0.243 | 0.300 | +0.057 | 0.0566 | 0.0570 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0065 (95th percentile 0.0092), simulated over 40 resamples. The measured ECE is 0.0697.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.27,0.33)  6000  0.306  0.241  +0.065  ##########..|...........................
  [0.40,0.47)  3025  0.428  0.450  -0.023  #################|......................
  [0.47,0.53)   764  0.520  0.628  -0.108  #####################|###...............
  [0.53,0.60)  1087  0.554  0.670  -0.116  ######################|####.............
  [0.67,0.73)  1037  0.727  0.849  -0.121  #############################|####......
  [0.73,0.80)  1490  0.770  0.851  -0.081  ###############################|##......
  [0.80,0.87)   448  0.819  0.844  -0.025  #################################|......
  [0.87,0.93)  4149  0.891  0.811  +0.080  ################################....|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
