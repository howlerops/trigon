# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 7,083 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 3,080 |
| — held out of train to reach the floor | 1,920 |
| Questions per request | 1 |
| Labels per question | 77 |

| Question | Marginal predictor |
| --- | ---: |
| `intent` | 0.0136 |
| Epochs | 4 |
| Seed | 3 |
| Device | cuda (NVIDIA A10G) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

`MIN_CALIBRATION_SAMPLES` is 5,000 and this
corpus's test split is smaller, so the evaluation set is topped up
from rows held out of train that neither training nor calibration
saw. They are held-out data by the only definition that matters and
they come from the train distribution, which is why the split is
reported above rather than summed into one number.

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+424e6cce

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.9132 | 0.0569 | 0.0568 | 0.1426 | 76.7 | 98.8 | 298 |
| banking77/calibrated | 5000 | 0.9126 | 0.0116 | 0.0178 | 0.1335 | 76.1 | 90.1 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0105 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0068 on this run, p95 0.0105
- PASS accuracy_over_baseline: 0.8964 (limit 0.0500) -- model 0.9126 vs marginal predictor 0.0162; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.8964 (limit 0.0500) -- worst is 'intent' at 0.9126 vs its own marginal 0.0162; the pooled gate hides this
- PASS workhorse_ece: 0.0116 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0178 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0116 (limit 0.0500) -- worst is 'choice' at ECE 0.0116 (overconfidence +0.009); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.9126 | 0.0162 | +0.8964 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.913 | 0.921 | +0.009 | 0.0116 | 0.0178 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.913 | 0.921 | +0.009 | 0.0116 | 0.0178 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0050 (95th percentile 0.0070), simulated over 200 resamples. The measured ECE is 0.0569.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)     3  0.247  0.333  -0.086  ##########|##...........................
  [0.27,0.33)     3  0.303  0.333  -0.030  ############|...........................
  [0.33,0.40)     8  0.358  0.000  +0.358  ..............|.........................
  [0.40,0.47)    24  0.431  0.167  +0.264  #######..........|......................
  [0.47,0.53)    38  0.508  0.316  +0.192  #############.......|...................
  [0.53,0.60)    44  0.569  0.386  +0.182  ###############........|................
  [0.60,0.67)    45  0.639  0.333  +0.306  #############.............|.............
  [0.67,0.73)    62  0.699  0.500  +0.199  ####################........|...........
  [0.73,0.80)    59  0.766  0.559  +0.206  ######################.........|........
  [0.80,0.87)    88  0.837  0.636  +0.201  #########################........|......
  [0.87,0.93)   146  0.906  0.623  +0.282  #########################...........|...
  [0.93,1.00)  4480  0.997  0.961  +0.036  ######################################..
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
