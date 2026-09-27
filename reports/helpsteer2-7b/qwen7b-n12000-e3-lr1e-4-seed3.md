# helpsteer2

**HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0. https://huggingface.co/datasets/nvidia/HelpSteer2**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 12,000 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 1,038 |
| — held out of train to reach the floor | 3,962 |
| Questions per request | 5 |
| Labels per question | 5 |

| Question | Marginal predictor |
| --- | ---: |
| `coherence` | 0.7104 |
| `complexity` | 0.5392 |
| `correctness` | 0.4624 |
| `helpfulness` | 0.3992 |
| `verbosity` | 0.6300 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6006 | 1.1468 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 3 |
| Device | cuda (NVIDIA A100-SXM4-80GB) |
| Model | qwen2.5-7b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --backbone qwen2.5-7b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-7b-0.1.0+7c2524d0

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5866 | 0.0374 | 0.0373 | 0.5317 | 78.2 | 159.8 | 582 |
| helpsteer2/calibrated | 5000 | 0.5866 | 0.0374 | 0.0373 | 0.5317 | 77.3 | 158.9 | 582 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0103 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0073 on this run, p95 0.0103
- FAIL accuracy_over_baseline: 0.0383 (limit 0.0500) -- model 0.5866 vs marginal predictor 0.5482; calibration cannot reject a model that ignores the state
- FAIL worst_question_over_baseline: 0.0012 (limit 0.0500) -- worst is 'coherence' at 0.7116 vs its own marginal 0.7104; the pooled gate hides this
- PASS workhorse_ece: 0.0374 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0373 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0374 (limit 0.0500) -- worst is 'score' at ECE 0.0374 (overconfidence +0.037); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline, worst_question_over_baseline.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7116 | 0.7104 | +0.0012 |
| `correctness` | 5,000 | 0.4894 | 0.4624 | +0.0270 |
| `helpfulness` | 5,000 | 0.4340 | 0.3992 | +0.0348 |
| `verbosity` | 5,000 | 0.6762 | 0.6300 | +0.0462 |
| `complexity` | 5,000 | 0.6216 | 0.5392 | +0.0824 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.587 | 0.624 | +0.037 | 0.0374 | 0.0373 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.587 | 0.624 | +0.037 | 0.0374 | 0.0373 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0073 (95th percentile 0.0103), simulated over 200 resamples. The measured ECE is 0.0374.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   191  0.253  0.178  +0.075  #######...|.............................
  [0.27,0.33)  1317  0.303  0.269  +0.034  ###########.|...........................
  [0.33,0.40)  1562  0.368  0.316  +0.051  #############..|........................
  [0.40,0.47)  1846  0.432  0.427  +0.005  #################|......................
  [0.47,0.53)  2446  0.504  0.463  +0.040  ###################.|...................
  [0.53,0.60)  3043  0.568  0.532  +0.035  #####################..|................
  [0.60,0.67)  3364  0.634  0.580  +0.054  #######################..|..............
  [0.67,0.73)  3559  0.700  0.654  +0.046  ##########################..|...........
  [0.73,0.80)  3700  0.767  0.730  +0.037  #############################..|........
  [0.80,0.87)  2945  0.830  0.804  +0.026  ################################.|......
  [0.87,0.93)  1019  0.889  0.867  +0.022  ###################################.|...
  [0.93,1.00)     8  0.937  1.000  -0.063  #####################################|##
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
