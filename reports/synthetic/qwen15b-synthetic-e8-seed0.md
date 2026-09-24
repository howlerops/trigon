# Reference run

Everything below is reproducible from this command — the run is seeded end to end, so it returns the same weights and the same gate verdicts:

```bash
trigon train -n 8000 --eval-n 6000 --epochs 8 --lr 0.0001 --accumulate 8 --d-model 192 --layers 3 --noise 0.2 --seed 0 --floor-trials 100 --calibration-n 1000 --option-scoring auto --backbone qwen2.5-1.5b --lora-rank 16
```

| Epoch | Mean loss | Seconds |
| ---: | ---: | ---: |
| 1 | 0.7721 | 295 |
| 2 | 0.5964 | 296 |
| 3 | 0.5701 | 295 |
| 4 | 0.5590 | 295 |
| 5 | 0.5428 | 296 |
| 6 | 0.5060 | 296 |
| 7 | 0.4303 | 296 |
| 8 | 0.3424 | 296 |

Chance is `1.1552` and the Bayes-optimal loss for this generator is `0.5585` — the label noise puts a floor under how well anything can do. The run closed **136%** of that gap.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+1016c13d, trigon-qwen2.5-1.5b-0.1.0+1016c13d+int8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.8361 | 0.0569 | 0.0618 | 0.2922 | 48.8 | 54.4 | 113 |
| calibration/temperature-scaled | 6000 | 0.8361 | 0.0408 | 0.0444 | 0.2895 | 48.7 | 54.7 | 113 |
| calibration/int8 | 6000 | 0.8360 | 0.0417 | 0.0452 | 0.2897 | 48.9 | 55.1 | 113 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0080 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0049 on this run, p95 0.0080
- PASS accuracy_over_baseline: 0.4438 (limit 0.0500) -- model 0.8361 vs marginal predictor 0.3922; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1423 (limit 0.0500) -- worst is 'at_risk' at 0.8095 vs its own marginal 0.6672; the pooled gate hides this
- PASS workhorse_ece: 0.0408 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0444 (limit 0.0500)
- FAIL (advisory) worst_primitive_workhorse_ece: 0.0551 (limit 0.0500) -- worst is 'score' at ECE 0.0551 (overconfidence -0.007); pooled ECE cancels heads that err in opposite directions
- PASS quantization_ece_delta: 0.0009 (limit 0.0100)

All blocking gates passed.

**Advisory, not blocking**: worst_primitive_workhorse_ece. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `calibration/temperature-scaled`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `at_risk` | 6,000 | 0.8095 | 0.6672 | +0.1423 |
| `size` | 6,000 | 0.8505 | 0.2565 | +0.5940 |
| `plan` | 6,000 | 0.8482 | 0.2530 | +0.5952 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.836 | 0.826 | -0.010 | 0.0408 | 0.0444 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.848 | 0.869 | +0.021 | 0.0207 | 0.0274 |
| noul | 6000 | 0.809 | 0.766 | -0.043 | 0.0508 | 0.0508 |
| score | 6000 | 0.851 | 0.844 | -0.007 | 0.0551 | 0.0551 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0050 (95th percentile 0.0082), simulated over 100 resamples. The measured ECE is 0.0569.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.40,0.47)     4  0.442  0.250  +0.192  ##########........|.....................
  [0.47,0.53)     1  0.505  0.000  +0.505  ....................|...................
  [0.53,0.60)     5  0.552  0.200  +0.352  ########..............|.................
  [0.60,0.67)   247  0.650  0.846  -0.196  ##########################|#######......
  [0.67,0.73)  3351  0.712  0.822  -0.110  ############################|####.......
  [0.73,0.80)  3602  0.759  0.833  -0.075  ##############################|##.......
  [0.80,0.87)  4489  0.842  0.831  +0.010  #################################.|.....
  [0.87,0.93)  6289  0.895  0.849  +0.046  ##################################..|...
  [0.93,1.00)    12  0.934  0.917  +0.018  #####################################|..
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
