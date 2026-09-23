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
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+649d6cda

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.9242 | 0.0499 | 0.0497 | 0.1264 | 58.4 | 65.6 | 298 |
| banking77/calibrated | 5000 | 0.9228 | 0.0077 | 0.0077 | 0.1192 | 59.1 | 64.5 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0103 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0066 on this run, p95 0.0103
- PASS accuracy_over_baseline: 0.9068 (limit 0.0500) -- model 0.9228 vs marginal predictor 0.0160; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.9068 (limit 0.0500) -- worst is 'intent' at 0.9228 vs its own marginal 0.0160; the pooled gate hides this
- PASS workhorse_ece: 0.0077 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0077 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0077 (limit 0.0500) -- worst is 'choice' at ECE 0.0077 (overconfidence +0.001); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.9228 | 0.0160 | +0.9068 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.923 | 0.924 | +0.001 | 0.0077 | 0.0077 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.923 | 0.924 | +0.001 | 0.0077 | 0.0077 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0050 (95th percentile 0.0071), simulated over 200 resamples. The measured ECE is 0.0499.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)     1  0.257  0.000  +0.257  ..........|.............................
  [0.27,0.33)     5  0.301  0.400  -0.099  ############|###........................
  [0.33,0.40)     9  0.361  0.222  +0.139  #########.....|.........................
  [0.40,0.47)    12  0.436  0.250  +0.186  ##########.......|......................
  [0.47,0.53)    37  0.506  0.297  +0.209  ############........|...................
  [0.53,0.60)    38  0.566  0.447  +0.119  ##################.....|................
  [0.60,0.67)    45  0.633  0.533  +0.100  #####################....|..............
  [0.67,0.73)    51  0.701  0.471  +0.230  ###################.........|...........
  [0.73,0.80)    54  0.765  0.537  +0.228  #####################..........|........
  [0.80,0.87)    64  0.827  0.531  +0.295  #####################............|......
  [0.87,0.93)   118  0.907  0.644  +0.263  ##########################..........|...
  [0.93,1.00)  4566  0.997  0.963  +0.034  #######################################.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
