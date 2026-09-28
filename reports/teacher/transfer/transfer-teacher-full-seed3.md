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

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.9874 | 4.3634 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 4 |
| Seed | 3 |
| Device | cuda (NVIDIA A10G) |
| Model | qwen2.5-1.5b, LoRA rank 16 |
| Initialised from | `/runs/teacher-init/qwen15b-e3-seed3.pt` |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000 --init-weights /runs/teacher-init/qwen15b-e3-seed3.pt
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+4fad0e49.init.cf72125a

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.0476 | 0.0226 | 0.0190 | 0.9640 | 79.2 | 82.7 | 298 |
| banking77/calibrated | 5000 | 0.0476 | 0.0226 | 0.0190 | 0.9640 | 79.7 | 87.1 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0079 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0044 on this run, p95 0.0079
- FAIL accuracy_over_baseline: 0.0314 (limit 0.0500) -- model 0.0476 vs marginal predictor 0.0162; calibration cannot reject a model that ignores the state
- FAIL worst_question_over_baseline: 0.0314 (limit 0.0500) -- worst is 'intent' at 0.0476 vs its own marginal 0.0162; the pooled gate hides this
- PASS workhorse_ece: 0.0226 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0190 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0226 (limit 0.0500) -- worst is 'choice' at ECE 0.0226 (overconfidence +0.014); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline, worst_question_over_baseline.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.0476 | 0.0162 | +0.0314 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.048 | 0.061 | +0.014 | 0.0226 | 0.0190 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.048 | 0.061 | +0.014 | 0.0226 | 0.0190 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0044 (95th percentile 0.0079), simulated over 200 resamples. The measured ECE is 0.0226.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.00,0.07)  4062  0.053  0.033  +0.020  #.|.....................................
  [0.07,0.13)   756  0.085  0.071  +0.013  ###|....................................
  [0.13,0.20)   182  0.146  0.269  -0.123  ######|####.............................
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
