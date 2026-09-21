# Re-gated: `iso-seed1.pt`

Calibration refitted on the seed-1 calibration split and the gates re-run, against weights trained earlier. Reproduce with:

```bash
python scripts/regate.py reports/iso/iso-seed1.pt --out reports/iso -n 8000 --calibration-n 1000 --eval-n 6000 --noise 0.2 --floor-trials 40
```

# Eval report

Model(s): trigon-reference-0.1.0+585308ce

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration/uncalibrated | 6000 | 0.5866 | 0.0431 | 0.0822 | 0.4943 | 3.2 | 6.8 | 117 |
| calibration/calibrated | 6000 | 0.6144 | 0.0235 | 0.0262 | 0.4671 | 3.2 | 6.7 | 117 |

## Release gates

- PASS sample_size: 18000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0080 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0050 on this run, p95 0.0080
- PASS accuracy_over_baseline: 0.2203 (limit 0.0500) -- model 0.6144 vs marginal predictor 0.3941; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0112 (limit 0.0500) -- worst is 'size' at 0.2447 vs its own marginal 0.2558; the pooled gate hides this
- PASS workhorse_ece: 0.0235 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0262 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0467 (limit 0.0500) -- worst is 'noul' at ECE 0.0467 (overconfidence +0.032); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `size` | 6,000 | 0.2447 | 0.2558 | -0.0112 ⚠ |
| `at_risk` | 6,000 | 0.6687 | 0.6687 | +0.0000 |
| `plan` | 6,000 | 0.8465 | 0.2577 | +0.5888 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accounts | 18000 | 0.614 | 0.629 | +0.015 | 0.0235 | 0.0262 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 6000 | 0.847 | 0.850 | +0.004 | 0.0240 | 0.0240 |
| noul | 6000 | 0.752 | 0.784 | +0.032 | 0.0467 | 0.0467 |
| score | 6000 | 0.245 | 0.253 | +0.008 | 0.0081 | 0.0216 |

## Is this number evidence?

On these 18000 predictions a perfectly calibrated model scores a mean ECE of 0.0053 (95th percentile 0.0079), simulated over 40 resamples. The measured ECE is 0.0431.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)  6000  0.253  0.245  +0.008  ##########|.............................
  [0.60,0.67)  6261  0.637  0.685  -0.049  #########################|#.............
  [0.67,0.73)  1246  0.691  0.804  -0.113  ############################|###........
  [0.73,0.80)  3006  0.777  0.845  -0.068  ###############################|##......
  [0.80,0.87)     6  0.801  0.833  -0.033  ################################|.......
  [0.87,0.93)  1481  0.900  0.847  +0.053  ##################################..|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
