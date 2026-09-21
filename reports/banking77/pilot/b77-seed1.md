# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 4,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases (the corpus's own test split) | 1,500 |
| Options | 77 |
| Marginal predictor accuracy | 0.0153 |
| Epochs | 4 |
| Seed | 1 |

```
python scripts/train_corpus.py banking77 -n 4000 --calibration-n 1000 --eval-n 1500 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-reference-0.1.0+706d730b

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 1500 | 0.4193 | 0.1290 | 0.1305 | 0.7504 | 36.6 | 79.0 | 608 |
| banking77/calibrated | 1500 | 0.4167 | 0.0351 | 0.0334 | 0.7240 | 36.5 | 56.2 | 608 |

## Release gates

- FAIL sample_size: 1500.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- FAIL gate_is_testable: 0.0362 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0249 on this run, p95 0.0362
- PASS accuracy_over_baseline: 0.4000 (limit 0.0500) -- model 0.4167 vs marginal predictor 0.0167; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.4000 (limit 0.0500) -- worst is 'intent' at 0.4167 vs its own marginal 0.0167; the pooled gate hides this
- PASS workhorse_ece: 0.0351 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0334 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0351 (limit 0.0500) -- worst is 'choice' at ECE 0.0351 (overconfidence +0.026); pooled ECE cancels heads that err in opposite directions

**Blocked**: sample_size, gate_is_testable.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 1,500 | 0.4193 | 0.0167 | +0.4027 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 1500 | 0.417 | 0.443 | +0.026 | 0.0351 | 0.0334 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 1500 | 0.417 | 0.443 | +0.026 | 0.0351 | 0.0334 |

## Is this number evidence?

On these 1500 predictions a perfectly calibrated model scores a mean ECE of 0.0283 (95th percentile 0.0384), simulated over 200 resamples. The measured ECE is 0.1290.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.00,0.07)     1  0.063  0.000  +0.063  ...|....................................
  [0.07,0.13)   116  0.111  0.121  -0.010  ####|...................................
  [0.13,0.20)   370  0.169  0.205  -0.036  #######|................................
  [0.20,0.27)   361  0.233  0.316  -0.083  #########|###...........................
  [0.27,0.33)   252  0.299  0.448  -0.149  ############|#####......................
  [0.33,0.40)   151  0.363  0.662  -0.299  ###############|##########..............
  [0.40,0.47)    74  0.427  0.730  -0.303  #################|###########...........
  [0.47,0.53)    33  0.503  0.788  -0.285  ####################|###########........
  [0.53,0.60)    46  0.567  0.891  -0.324  #######################|############....
  [0.60,0.67)    23  0.627  1.000  -0.373  #########################|##############
  [0.67,0.73)    21  0.701  0.952  -0.251  ############################|#########..
  [0.73,0.80)    19  0.760  0.947  -0.188  ##############################|#######..
  [0.80,0.87)    18  0.833  0.833  -0.000  #################################|......
  [0.87,0.93)    15  0.875  1.000  -0.125  ###################################|####
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
