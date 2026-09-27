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
| `coherence` | 0.7164 |
| `complexity` | 0.5286 |
| `correctness` | 0.4808 |
| `helpfulness` | 0.4186 |
| `verbosity` | 0.6210 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.5983 | 1.1427 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 2 |
| Device | cuda (NVIDIA A100-SXM4-80GB) |
| Model | qwen2.5-7b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --backbone qwen2.5-7b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-7b-0.1.0+0517e6ff

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5930 | 0.0286 | 0.0281 | 0.5334 | 77.5 | 156.3 | 587 |
| helpsteer2/calibrated | 5000 | 0.5930 | 0.0100 | 0.0133 | 0.5320 | 77.8 | 156.7 | 587 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0101 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0071 on this run, p95 0.0101
- FAIL accuracy_over_baseline: 0.0400 (limit 0.0500) -- model 0.5930 vs marginal predictor 0.5531; calibration cannot reject a model that ignores the state
- FAIL worst_question_over_baseline: -0.0010 (limit 0.0500) -- worst is 'coherence' at 0.7154 vs its own marginal 0.7164; the pooled gate hides this
- PASS workhorse_ece: 0.0100 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0133 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0100 (limit 0.0500) -- worst is 'score' at ECE 0.0100 (overconfidence -0.000); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline, worst_question_over_baseline.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7154 | 0.7164 | -0.0010 ⚠ |
| `correctness` | 5,000 | 0.5092 | 0.4808 | +0.0284 |
| `helpfulness` | 5,000 | 0.4546 | 0.4186 | +0.0360 |
| `verbosity` | 5,000 | 0.6702 | 0.6210 | +0.0492 |
| `complexity` | 5,000 | 0.6158 | 0.5286 | +0.0872 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.593 | 0.593 | -0.000 | 0.0100 | 0.0133 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.593 | 0.593 | -0.000 | 0.0100 | 0.0133 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0071 (95th percentile 0.0105), simulated over 200 resamples. The measured ECE is 0.0286.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   561  0.249  0.228  +0.021  #########.|.............................
  [0.27,0.33)  1166  0.301  0.298  +0.003  ############|...........................
  [0.33,0.40)  1392  0.368  0.371  -0.003  ###############|........................
  [0.40,0.47)  2123  0.437  0.447  -0.010  #################|......................
  [0.47,0.53)  4252  0.502  0.516  -0.014  ####################|...................
  [0.53,0.60)  5005  0.566  0.601  -0.035  #######################|................
  [0.60,0.67)  4590  0.633  0.667  -0.034  #########################|#.............
  [0.67,0.73)  3094  0.697  0.751  -0.054  ############################|#..........
  [0.73,0.80)  2233  0.763  0.814  -0.050  ###############################|#.......
  [0.80,0.87)   550  0.820  0.818  +0.001  #################################|......
  [0.87,0.93)    32  0.892  0.875  +0.017  ###################################.|...
  [0.93,1.00)     2  0.936  1.000  -0.064  #####################################|##
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
