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
| Epochs | 3 |
| Seed | 3 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+7501551c

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5770 | 0.0153 | 0.0171 | 0.5493 | 54.8 | 148.3 | 582 |
| helpsteer2/calibrated | 5000 | 0.5770 | 0.0153 | 0.0171 | 0.5493 | 54.2 | 147.4 | 582 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0103 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0072 on this run, p95 0.0103
- FAIL accuracy_over_baseline: 0.0288 (limit 0.0500) -- model 0.5770 vs marginal predictor 0.5482; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0006 (limit 0.0500) -- worst is 'coherence' at 0.7098 vs its own marginal 0.7104; the pooled gate hides this
- PASS workhorse_ece: 0.0153 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0171 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0153 (limit 0.0500) -- worst is 'score' at ECE 0.0153 (overconfidence +0.013); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7098 | 0.7104 | -0.0006 ⚠ |
| `correctness` | 5,000 | 0.4692 | 0.4624 | +0.0068 |
| `helpfulness` | 5,000 | 0.4120 | 0.3992 | +0.0128 |
| `verbosity` | 5,000 | 0.6654 | 0.6300 | +0.0354 |
| `complexity` | 5,000 | 0.6288 | 0.5392 | +0.0896 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.577 | 0.590 | +0.013 | 0.0153 | 0.0171 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.577 | 0.590 | +0.013 | 0.0153 | 0.0171 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0072 (95th percentile 0.0103), simulated over 200 resamples. The measured ECE is 0.0153.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   671  0.243  0.253  -0.010  ##########|.............................
  [0.27,0.33)  1201  0.302  0.300  +0.002  ############|...........................
  [0.33,0.40)  1565  0.368  0.386  -0.018  ###############|........................
  [0.40,0.47)  1993  0.436  0.426  +0.010  #################|......................
  [0.47,0.53)  3328  0.502  0.495  +0.007  ####################|...................
  [0.53,0.60)  3548  0.567  0.532  +0.035  #####################..|................
  [0.60,0.67)  3536  0.634  0.618  +0.016  #########################|..............
  [0.67,0.73)  3979  0.700  0.686  +0.015  ###########################.|...........
  [0.73,0.80)  3661  0.764  0.757  +0.007  ##############################.|........
  [0.80,0.87)  1351  0.824  0.804  +0.020  ################################.|......
  [0.87,0.93)   167  0.883  0.820  +0.062  #################################..|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
