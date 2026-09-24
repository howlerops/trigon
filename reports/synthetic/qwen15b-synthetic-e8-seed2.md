# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.0001 --accumulate 8 --d-model 192 --layers 3 --noise 0.2 --seed 2 --floor-trials 100 --calibration-n 1000 --option-scoring auto --backbone qwen2.5-1.5b --lora-rank 16
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.8326 | 300 |
| 2 | 0.5951 | 305 |
| 3 | 0.5707 | 301 |
| 4 | 0.5608 | 301 |
| 5 | 0.5513 | 299 |
| 6 | 0.5267 | 300 |
| 7 | 0.4725 | 300 |
| 8 | 0.4077 | 299 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **125%** of that gap.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+af5eaceb, trigon-qwen2.5-1.5b-0.1.0+af5eaceb+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.8253 | 0.0341 | 0.0346 | 0.2999 | 51.6 | 54.4 | 113 |
| calibration/temperature-scaled | 6000 | 0.8252 | 0.0114 | 0.0127 | 0.3000 | 51.3 | 54.3 | 113 |
| calibration/int8 | 6000 | 0.8245 | 0.0116 | 0.0114 | 0.3010 | 51.1 | 53.3 | 113 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0064 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0037 on this run, p95 0.0064
- PASS accuracy_over_baseline: 0.4337 (limit 0.0500) -- model 0.8252 vs marginal predictor 0.3915; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1360 (limit 0.0500) -- worst is 'at_risk' at 0.7945 vs its own marginal 0.6585; the pooled gate hides this
- PASS workhorse_ece: 0.0114 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0127 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0254 (limit 0.0500) -- worst is 'noul' at ECE 0.0254 (overconfidence +0.023); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0002 (limit 0.0100)

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `calibration/temperature-scaled`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `at_risk` | 6,000 | 0.7945 | 0.6585 | +0.1360 |
| `size` | 6,000 | 0.8325 | 0.2613 | +0.5712 |
| `plan` | 6,000 | 0.8487 | 0.2547 | +0.5940 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.825 | 0.836 | +0.011 | 0.0114 | 0.0127 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.849 | 0.850 | +0.002 | 0.0066 | 0.0134 |
| noul | 6000 | 0.794 | 0.818 | +0.023 | 0.0254 | 0.0286 |
| score | 6000 | 0.833 | 0.839 | +0.006 | 0.0065 | 0.0065 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0043 (95th percentile 0.0074), simulated over 100 resamples. The measured ECE is 0.0341.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.40,0.47)    24  0.436  0.458  -0.023  #################|......................
  [0.47,0.53)    33  0.498  0.273  +0.225  ###########.........|...................
  [0.53,0.60)    20  0.570  0.500  +0.070  ####################...|................
  [0.60,0.67)    24  0.627  0.625  +0.002  #########################|..............
  [0.67,0.73)    31  0.716  0.677  +0.039  ###########################..|..........
  [0.73,0.80)  2367  0.780  0.801  -0.020  ###############################|........
  [0.80,0.87)  8572  0.837  0.819  +0.018  #################################|......
  [0.87,0.93)  6465  0.903  0.847  +0.056  ##################################..|...
  [0.93,1.00)   464  0.944  0.856  +0.088  ##################################....|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
