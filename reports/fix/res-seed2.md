# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.01 --accumulate 16 --d-model 128 --layers 2 --noise 0.2 --seed 2 --floor-trials 40 --calibration-n 1000 --option-scoring auto --score-residual
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.9559 | 128 |
| 2 | 0.9299 | 123 |
| 3 | 0.9724 | 122 |
| 4 | 0.9787 | 127 |
| 5 | 0.9764 | 129 |
| 6 | 0.9748 | 133 |
| 7 | 0.9733 | 134 |
| 8 | 0.9727 | 133 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **31%** of that gap.
# Eval report

Model(s): trigon-reference-0.1.0+a3ba5687, trigon-reference-0.1.0+a3ba5687+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5547 | 0.0456 | 0.0562 | 0.5205 | 3.7 | 4.6 | 114 |
| calibration/temperature-scaled | 6000 | 0.5528 | 0.0325 | 0.0297 | 0.5212 | 3.6 | 4.8 | 114 |
| calibration/int8 | 6000 | 0.5542 | 0.0560 | 0.0550 | 0.5492 | 3.6 | 4.7 | 114 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0091 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0060 on this run, p95 0.0091
- PASS accuracy_over_baseline: 0.1613 (limit 0.0500) -- model 0.5528 vs marginal predictor 0.3915; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0045 (limit 0.0500) -- worst is 'size' at 0.2568 vs its own marginal 0.2613; the pooled gate hides this
- PASS workhorse_ece: 0.0325 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0297 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0491 (limit 0.0500) -- worst is 'noul' at ECE 0.0491 (overconfidence +0.049); pooled ECE cancels heads that err in opposite directions
- FAIL quantization_ece_delta: 0.0235 (limit 0.0100)

**Blocked**: quantization_ece_delta.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2612 | 0.2613 | -0.0002 ⚠ |
| `at_risk` | 6,000 | 0.7503 | 0.6585 | +0.0918 |
| `plan` | 6,000 | 0.6525 | 0.2547 | +0.3978 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.553 | 0.577 | +0.024 | 0.0325 | 0.0297 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.651 | 0.653 | +0.001 | 0.0254 | 0.0228 |
| noul | 6000 | 0.750 | 0.799 | +0.049 | 0.0491 | 0.0491 |
| score | 6000 | 0.257 | 0.280 | +0.023 | 0.0230 | 0.0272 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0067 (95th percentile 0.0099), simulated over 40 resamples. The measured ECE is 0.0456.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  1783  0.261  0.258  +0.003  ##########|.............................
  [0.27,0.33)  3162  0.308  0.256  +0.052  ##########..|...........................
  [0.33,0.40)  1055  0.340  0.282  +0.058  ###########...|.........................
  [0.40,0.47)  2959  0.421  0.450  -0.029  #################|......................
  [0.67,0.73)  1783  0.694  0.655  +0.040  ##########################..|...........
  [0.73,0.80)  1055  0.795  0.858  -0.062  ################################|#......
  [0.80,0.87)  4660  0.843  0.798  +0.045  ################################..|.....
  [0.87,0.93)  1085  0.927  0.837  +0.090  #################################....|..
  [0.93,1.00)   458  0.976  0.847  +0.129  ##################################.....|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
