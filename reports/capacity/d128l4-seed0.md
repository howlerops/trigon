# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 4000 --epochs 6 --lr 0.01 --accumulate 16 --d-model 128 --layers 4 --noise 0.2 --seed 0 --floor-trials 20 --calibration-n 1000 --option-scoring auto
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9486 | 162 |
| 2 | 1.0239 | 208 |
| 3 | 0.9872 | 211 |
| 4 | 0.9855 | 207 |
| 5 | 0.9769 | 196 |
| 6 | 0.9607 | 181 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **33%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+d7fc93c5, trigon-reference-0.1.0+d7fc93c5+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 4000 | 0.5104 | 0.0158 | 0.0257 | 0.5530 | 5.5 | 9.7 | 121 |
| calibration/temperature-scaled | 4000 | 0.5104 | 0.0208 | 0.0257 | 0.5530 | 5.5 | 7.8 | 121 |
| calibration/int8 | 4000 | 0.5104 | 0.0220 | 0.0266 | 0.5535 | 5.5 | 8.4 | 121 |

## Release gates

- PASS sample_size: 12000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0091 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0066 on this run, p95 0.0091
- PASS accuracy_over_baseline: 0.1182 (limit 0.0500) -- model 0.5104 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0025 (limit 0.0500) -- worst is 'size' at 0.2540 vs its own marginal 0.2565; the pooled gate hides this
- PASS workhorse_ece: 0.0208 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0257 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0374 (limit 0.0500) -- worst is 'choice' at ECE 0.0374 (overconfidence +0.037); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0013 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 4,000 | 0.2540 | 0.2565 | -0.0025 ⚠ |
| `at_risk` | 4,000 | 0.6660 | 0.6660 | +0.0000 |
| `plan` | 4,000 | 0.6112 | 0.2540 | +0.3572 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 12000 | 0.510 | 0.517 | +0.006 | 0.0208 | 0.0257 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 4000 | 0.611 | 0.649 | +0.037 | 0.0374 | 0.0374 |
| noul | 4000 | 0.666 | 0.644 | -0.022 | 0.0221 | 0.0221 |
| score | 4000 | 0.254 | 0.257 | +0.003 | 0.0030 | 0.0245 |

## Is this number evidence?

On these 12000 predictions a perfectly calibrated model scores a mean ECE of 0.0087 (95th percentile 0.0121), simulated over 20 resamples. The measured ECE is 0.0158.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  4000  0.257  0.254  +0.003  ##########|.............................
  [0.27,0.33)   155  0.305  0.052  +0.254  ##..........|...........................
  [0.33,0.40)    22  0.338  0.045  +0.293  ##............|.........................
  [0.40,0.47)  1676  0.459  0.448  +0.011  ##################|.....................
  [0.47,0.53)   331  0.469  0.423  +0.046  #################..|....................
  [0.53,0.60)     2  0.573  1.000  -0.427  #######################|################
  [0.60,0.67)  1994  0.647  0.670  -0.023  ##########################|.............
  [0.67,0.73)  2026  0.673  0.664  +0.009  ###########################|............
  [0.73,0.80)   254  0.786  0.850  -0.064  ###############################|##......
  [0.80,0.87)  1540  0.838  0.850  -0.012  ##################################|.....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
