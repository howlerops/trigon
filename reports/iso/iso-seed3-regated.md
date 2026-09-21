# Re-gated: `iso-seed3.pt`

Calibration refitted on the seed-3 calibration split and the gates re-run, against weights trained earlier. Reproduce with:

```bash
python scripts/regate.py reports/iso/iso-seed3.pt --out reports/iso -n 8000 --calibration-n 1000 --eval-n 6000 --noise 0.2 --floor-trials 40
```

# Eval report

Model(s): trigon-reference-0.1.0+94e16f25

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.6081 | 0.0121 | 0.0188 | 0.4734 | 3.3 | 224.1 | 117 |
| calibration/calibrated | 6000 | 0.6081 | 0.0087 | 0.0153 | 0.4737 | 3.3 | 7.6 | 117 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0090 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0047 on this run, p95 0.0090
- PASS accuracy_over_baseline: 0.2141 (limit 0.0500) -- model 0.6081 vs marginal predictor 0.3940; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0230 (limit 0.0500) -- worst is 'size' at 0.2430 vs its own marginal 0.2660; the pooled gate hides this
- PASS workhorse_ece: 0.0087 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0153 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0148 (limit 0.0500) -- worst is 'noul' at ECE 0.0148 (overconfidence +0.007); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2430 | 0.2660 | -0.0230 ⚠ |
| `at_risk` | 6,000 | 0.7513 | 0.6573 | +0.0940 |
| `plan` | 6,000 | 0.8298 | 0.2587 | +0.5712 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.608 | 0.612 | +0.004 | 0.0087 | 0.0153 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.830 | 0.824 | -0.006 | 0.0064 | 0.0126 |
| noul | 6000 | 0.751 | 0.758 | +0.007 | 0.0148 | 0.0189 |
| score | 6000 | 0.243 | 0.252 | +0.009 | 0.0095 | 0.0279 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0053 (95th percentile 0.0088), simulated over 40 resamples. The measured ECE is 0.0121.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.252  0.243  +0.009  ##########|.............................
  [0.60,0.67)  1780  0.645  0.658  -0.013  ##########################|.............
  [0.67,0.73)    10  0.704  1.000  -0.296  ############################|###########
  [0.80,0.87)  8713  0.818  0.813  +0.004  #################################|......
  [0.87,0.93)  1497  0.879  0.814  +0.066  #################################..|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
