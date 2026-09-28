# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 1,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 3,080 |
| — held out of train to reach the floor | 1,920 |
| Questions per request | 1 |
| Labels per question | 77 |

| Question | Marginal predictor |
| --- | ---: |
| `intent` | 0.0162 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.9880 | 4.3879 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 4 |
| Seed | 3 |
| Device | cuda (NVIDIA A10G) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py banking77 -n 1000 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+5fc935d1

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.6302 | 0.0876 | 0.0873 | 0.5099 | 78.1 | 108.2 | 298 |
| banking77/calibrated | 5000 | 0.6302 | 0.0247 | 0.0171 | 0.4995 | 77.7 | 85.2 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0222 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0167 on this run, p95 0.0222
- PASS accuracy_over_baseline: 0.6140 (limit 0.0500) -- model 0.6302 vs marginal predictor 0.0162; calibration cannot reject a model that ignores the state
- PASS worst_question_over_baseline: 0.6140 (limit 0.0500) -- worst is 'intent' at 0.6302 vs its own marginal 0.0162; the pooled gate hides this
- PASS workhorse_ece: 0.0247 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0171 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0247 (limit 0.0500) -- worst is 'choice' at ECE 0.0247 (overconfidence +0.009); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.6302 | 0.0162 | +0.6140 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.630 | 0.639 | +0.009 | 0.0247 | 0.0171 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.630 | 0.639 | +0.009 | 0.0247 | 0.0171 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0151 (95th percentile 0.0209), simulated over 200 resamples. The measured ECE is 0.0876.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.07,0.13)    11  0.119  0.182  -0.063  #####|#.................................
  [0.13,0.20)    64  0.175  0.109  +0.066  ####...|................................
  [0.20,0.27)   123  0.236  0.187  +0.049  #######..|..............................
  [0.27,0.33)   213  0.300  0.254  +0.046  ##########..|...........................
  [0.33,0.40)   281  0.371  0.299  +0.072  ############...|........................
  [0.40,0.47)   310  0.433  0.339  +0.094  ##############...|......................
  [0.47,0.53)   345  0.500  0.406  +0.094  ################....|...................
  [0.53,0.60)   315  0.566  0.463  +0.103  ###################....|................
  [0.60,0.67)   341  0.634  0.534  +0.100  #####################....|..............
  [0.67,0.73)   300  0.700  0.610  +0.090  ########################....|...........
  [0.73,0.80)   361  0.767  0.615  +0.152  #########################......|........
  [0.80,0.87)   383  0.834  0.687  +0.147  ###########################......|......
  [0.87,0.93)   525  0.902  0.770  +0.132  ###############################.....|...
  [0.93,1.00)  1428  0.979  0.936  +0.043  #####################################..|
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
