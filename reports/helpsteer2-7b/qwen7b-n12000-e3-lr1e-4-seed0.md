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
| `coherence` | 0.7120 |
| `complexity` | 0.5330 |
| `correctness` | 0.4784 |
| `helpfulness` | 0.4094 |
| `verbosity` | 0.6292 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.5962 | 1.1349 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 0 |
| Device | cuda (NVIDIA A100 80GB PCIe) |
| Model | qwen2.5-7b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --backbone qwen2.5-7b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-7b-0.1.0+e2fb31e7

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5920 | 0.0465 | 0.0465 | 0.5308 | 65.8 | 164.5 | 583 |
| helpsteer2/calibrated | 5000 | 0.5920 | 0.0149 | 0.0149 | 0.5279 | 63.0 | 160.6 | 583 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0100 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0074 on this run, p95 0.0100
- FAIL accuracy_over_baseline: 0.0396 (limit 0.0500) -- model 0.5920 vs marginal predictor 0.5524; calibration cannot reject a model that ignores the state
- FAIL worst_question_over_baseline: 0.0006 (limit 0.0500) -- worst is 'coherence' at 0.7126 vs its own marginal 0.7120; the pooled gate hides this
- PASS workhorse_ece: 0.0149 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0149 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0149 (limit 0.0500) -- worst is 'score' at ECE 0.0149 (overconfidence +0.015); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline, worst_question_over_baseline.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7126 | 0.7120 | +0.0006 |
| `correctness` | 5,000 | 0.5038 | 0.4784 | +0.0254 |
| `helpfulness` | 5,000 | 0.4374 | 0.4094 | +0.0280 |
| `verbosity` | 5,000 | 0.6810 | 0.6292 | +0.0518 |
| `complexity` | 5,000 | 0.6254 | 0.5330 | +0.0924 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.592 | 0.607 | +0.015 | 0.0149 | 0.0149 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.592 | 0.607 | +0.015 | 0.0149 | 0.0149 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0073 (95th percentile 0.0102), simulated over 200 resamples. The measured ECE is 0.0465.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   446  0.249  0.231  +0.018  #########.|.............................
  [0.27,0.33)   938  0.299  0.278  +0.021  ###########.|...........................
  [0.33,0.40)   963  0.368  0.318  +0.051  #############..|........................
  [0.40,0.47)  1369  0.437  0.402  +0.035  ################.|......................
  [0.47,0.53)  2792  0.503  0.464  +0.039  ###################.|...................
  [0.53,0.60)  3310  0.567  0.522  +0.045  #####################..|................
  [0.60,0.67)  3583  0.633  0.575  +0.059  #######################..|..............
  [0.67,0.73)  3475  0.700  0.646  +0.054  ##########################..|...........
  [0.73,0.80)  3389  0.767  0.724  +0.043  #############################..|........
  [0.80,0.87)  3148  0.833  0.787  +0.046  ###############################..|......
  [0.87,0.93)  1558  0.890  0.832  +0.058  #################################...|...
  [0.93,1.00)    29  0.941  0.828  +0.113  #################################.....|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
