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
| `intent` | 0.0130 |
| Epochs | 4 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+e219ccc8

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.9118 | 0.0489 | 0.0470 | 0.1412 | 60.0 | 81.5 | 298 |
| banking77/calibrated | 5000 | 0.9118 | 0.0489 | 0.0470 | 0.1412 | 59.5 | 86.4 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0085 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0059 on this run, p95 0.0085
- PASS accuracy_over_baseline: 0.8958 (limit 0.0500) -- model 0.9118 vs marginal predictor 0.0160; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.8958 (limit 0.0500) -- worst is 'intent' at 0.9118 vs its own marginal 0.0160; the pooled gate hides this
- PASS workhorse_ece: 0.0489 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0470 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0489 (limit 0.0500) -- worst is 'choice' at ECE 0.0489 (overconfidence +0.047); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.9118 | 0.0160 | +0.8958 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.912 | 0.959 | +0.047 | 0.0489 | 0.0470 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.912 | 0.959 | +0.047 | 0.0489 | 0.0470 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0059 (95th percentile 0.0085), simulated over 200 resamples. The measured ECE is 0.0489.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)     4  0.243  0.250  -0.007  ##########|.............................
  [0.27,0.33)    13  0.306  0.308  -0.001  ############|...........................
  [0.33,0.40)    16  0.365  0.562  -0.197  ###############|######..................
  [0.40,0.47)    20  0.431  0.500  -0.069  #################|##....................
  [0.47,0.53)    59  0.506  0.288  +0.217  ############........|...................
  [0.53,0.60)    65  0.565  0.400  +0.165  ################.......|................
  [0.60,0.67)    70  0.632  0.471  +0.160  ###################......|..............
  [0.67,0.73)    71  0.698  0.507  +0.191  ####################........|...........
  [0.73,0.80)    66  0.767  0.636  +0.131  #########################......|........
  [0.80,0.87)   110  0.830  0.682  +0.148  ###########################......|......
  [0.87,0.93)   175  0.904  0.697  +0.207  ############################........|...
  [0.93,1.00)  4331  0.996  0.966  +0.030  #######################################.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
