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
| Epochs | 3 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+4e0e460f

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5583 | 0.0225 | 0.0284 | 0.5847 | 58.5 | 147.0 | 583 |
| helpsteer2/calibrated | 5000 | 0.5583 | 0.0225 | 0.0284 | 0.5847 | 58.5 | 146.7 | 583 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0093 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0065 on this run, p95 0.0093
- FAIL accuracy_over_baseline: 0.0059 (limit 0.0500) -- model 0.5583 vs marginal predictor 0.5524; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0000 (limit 0.0500) -- worst is 'coherence' at 0.7120 vs its own marginal 0.7120; the pooled gate hides this
- PASS workhorse_ece: 0.0225 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0284 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0225 (limit 0.0500) -- worst is 'score' at ECE 0.0225 (overconfidence +0.005); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7120 | 0.7120 | +0.0000 |
| `correctness` | 5,000 | 0.4784 | 0.4784 | +0.0000 |
| `helpfulness` | 5,000 | 0.4094 | 0.4094 | +0.0000 |
| `verbosity` | 5,000 | 0.6350 | 0.6292 | +0.0058 |
| `complexity` | 5,000 | 0.5568 | 0.5330 | +0.0238 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.558 | 0.564 | +0.005 | 0.0225 | 0.0284 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.558 | 0.564 | +0.005 | 0.0225 | 0.0284 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0065 (95th percentile 0.0093), simulated over 200 resamples. The measured ECE is 0.0225.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)    28  0.252  0.607  -0.355  ##########|#############................
  [0.27,0.33)   397  0.312  0.393  -0.081  ############|###........................
  [0.33,0.40)  2204  0.372  0.383  -0.011  ###############|........................
  [0.40,0.47)  3753  0.435  0.410  +0.025  ################.|......................
  [0.47,0.53)  4637  0.497  0.471  +0.026  ###################.|...................
  [0.53,0.60)  3989  0.565  0.570  -0.005  #######################|................
  [0.60,0.67)  3692  0.632  0.667  -0.034  #########################|#.............
  [0.67,0.73)  3038  0.706  0.702  +0.004  ############################|...........
  [0.73,0.80)  3262  0.757  0.720  +0.037  #############################.|.........
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
