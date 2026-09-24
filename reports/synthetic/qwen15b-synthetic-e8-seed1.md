# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.0001 --accumulate 8 --d-model 192 --layers 3 --noise 0.2 --seed 1 --floor-trials 100 --calibration-n 1000 --option-scoring auto --backbone qwen2.5-1.5b --lora-rank 16
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.8096 | 301 |
| 2 | 0.5967 | 301 |
| 3 | 0.5742 | 300 |
| 4 | 0.5615 | 301 |
| 5 | 0.5372 | 301 |
| 6 | 0.4755 | 301 |
| 7 | 0.3358 | 302 |
| 8 | 0.2140 | 303 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **158%** of that gap.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+967776f4, trigon-qwen2.5-1.5b-0.1.0+967776f4+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.8299 | 0.0337 | 0.0358 | 0.2952 | 48.7 | 55.4 | 113 |
| calibration/temperature-scaled | 6000 | 0.8299 | 0.0307 | 0.0324 | 0.2948 | 48.3 | 53.5 | 113 |
| calibration/int8 | 6000 | 0.8298 | 0.0305 | 0.0326 | 0.2949 | 48.8 | 54.3 | 113 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0083 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0049 on this run, p95 0.0083
- PASS accuracy_over_baseline: 0.4359 (limit 0.0500) -- model 0.8299 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1268 (limit 0.0500) -- worst is 'at_risk' at 0.7955 vs its own marginal 0.6687; the pooled gate hides this
- PASS workhorse_ece: 0.0307 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0324 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0344 (limit 0.0500) -- worst is 'noul' at ECE 0.0344 (overconfidence -0.034); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0002 (limit 0.0100)

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `calibration/temperature-scaled`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `at_risk` | 6,000 | 0.7955 | 0.6687 | +0.1268 |
| `plan` | 6,000 | 0.8465 | 0.2577 | +0.5888 |
| `size` | 6,000 | 0.8478 | 0.2558 | +0.5920 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.830 | 0.826 | -0.004 | 0.0307 | 0.0324 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.847 | 0.850 | +0.003 | 0.0240 | 0.0240 |
| noul | 6000 | 0.795 | 0.761 | -0.034 | 0.0344 | 0.0427 |
| score | 6000 | 0.848 | 0.866 | +0.018 | 0.0337 | 0.0404 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0048 (95th percentile 0.0083), simulated over 100 resamples. The measured ECE is 0.0337.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.40,0.47)     2  0.459  1.000  -0.541  ##################|#####################
  [0.47,0.53)    10  0.501  0.700  -0.199  ####################|#######............
  [0.53,0.60)    18  0.580  0.833  -0.253  #######################|#########.......
  [0.60,0.67)   106  0.641  0.802  -0.160  ##########################|#####........
  [0.67,0.73)  1073  0.713  0.801  -0.088  #############################|##........
  [0.73,0.80)  4786  0.770  0.801  -0.031  ###############################|........
  [0.80,0.87)  5452  0.834  0.840  -0.006  #################################|......
  [0.87,0.93)  6377  0.893  0.847  +0.046  ##################################..|...
  [0.93,1.00)   176  0.939  0.881  +0.058  ###################################...|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
