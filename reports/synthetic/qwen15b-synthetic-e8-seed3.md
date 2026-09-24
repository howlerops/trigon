# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.0001 --accumulate 8 --d-model 192 --layers 3 --noise 0.2 --seed 3 --floor-trials 100 --calibration-n 1000 --option-scoring auto --backbone qwen2.5-1.5b --lora-rank 16
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.7751 | 301 |
| 2 | 0.5814 | 302 |
| 3 | 0.5641 | 301 |
| 4 | 0.5491 | 300 |
| 5 | 0.5261 | 299 |
| 6 | 0.4698 | 300 |
| 7 | 0.3498 | 299 |
| 8 | 0.2517 | 300 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **151%** of that gap.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+16704926, trigon-qwen2.5-1.5b-0.1.0+16704926+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.8251 | 0.0380 | 0.0407 | 0.3034 | 49.4 | 55.7 | 113 |
| calibration/temperature-scaled | 6000 | 0.8249 | 0.0106 | 0.0190 | 0.3007 | 49.9 | 56.1 | 113 |
| calibration/int8 | 6000 | 0.8249 | 0.0114 | 0.0193 | 0.3007 | 49.5 | 56.4 | 113 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0072 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0043 on this run, p95 0.0072
- PASS accuracy_over_baseline: 0.4309 (limit 0.0500) -- model 0.8249 vs marginal predictor 0.3940; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1355 (limit 0.0500) -- worst is 'at_risk' at 0.7928 vs its own marginal 0.6573; the pooled gate hides this
- PASS workhorse_ece: 0.0106 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0190 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0262 (limit 0.0500) -- worst is 'noul' at ECE 0.0262 (overconfidence +0.000); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0008 (limit 0.0100)

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `calibration/temperature-scaled`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `at_risk` | 6,000 | 0.7928 | 0.6573 | +0.1355 |
| `plan` | 6,000 | 0.8298 | 0.2587 | +0.5712 |
| `size` | 6,000 | 0.8522 | 0.2660 | +0.5862 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.825 | 0.827 | +0.002 | 0.0106 | 0.0190 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.830 | 0.822 | -0.008 | 0.0076 | 0.0076 |
| noul | 6000 | 0.793 | 0.793 | +0.000 | 0.0262 | 0.0274 |
| score | 6000 | 0.852 | 0.864 | +0.012 | 0.0221 | 0.0221 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0048 (95th percentile 0.0074), simulated over 100 resamples. The measured ECE is 0.0380.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.53,0.60)    10  0.589  0.600  -0.011  ########################|...............
  [0.60,0.67)   335  0.648  0.815  -0.167  ##########################|######.......
  [0.67,0.73)  1117  0.700  0.818  -0.119  ############################|####.......
  [0.73,0.80)  3144  0.779  0.812  -0.034  ###############################|........
  [0.80,0.87)  8800  0.833  0.821  +0.012  #################################|......
  [0.87,0.93)  3506  0.894  0.841  +0.053  ##################################..|...
  [0.93,1.00)  1088  0.947  0.858  +0.088  ##################################....|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
