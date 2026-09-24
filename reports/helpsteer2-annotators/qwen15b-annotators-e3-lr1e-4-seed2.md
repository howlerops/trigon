# helpsteer2-annotators

**HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0. https://huggingface.co/datasets/nvidia/HelpSteer2**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 16,834 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,818 |
| — from the corpus's own test split | 5,818 |
| — held out of train to reach the floor | 0 |
| Questions per request | 5 |
| Labels per question | 5 |

| Question | Marginal predictor |
| --- | ---: |
| `coherence` | 0.6927 |
| `complexity` | 0.4697 |
| `correctness` | 0.4648 |
| `helpfulness` | 0.4129 |
| `verbosity` | 0.5877 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6340 | 1.2460 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 2 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2-annotators -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+76ae7216

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators/uncalibrated | 5818 | 0.5435 | 0.0083 | 0.0101 | 0.5996 | 57.3 | 151.9 | 591 |
| helpsteer2-annotators/calibrated | 5818 | 0.5435 | 0.0083 | 0.0101 | 0.5996 | 57.9 | 151.3 | 591 |

## Release gates

- PASS sample_size: 29090.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0096 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0067 on this run, p95 0.0096
- FAIL accuracy_over_baseline: 0.0180 (limit 0.0500) -- model 0.5435 vs marginal predictor 0.5255; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0005 (limit 0.0500) -- worst is 'coherence' at 0.6922 vs its own marginal 0.6927; the pooled gate hides this
- PASS workhorse_ece: 0.0083 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0101 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0083 (limit 0.0500) -- worst is 'score' at ECE 0.0083 (overconfidence +0.004); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2-annotators/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,818 | 0.6922 | 0.6927 | -0.0005 ⚠ |
| `correctness` | 5,818 | 0.4756 | 0.4648 | +0.0108 |
| `helpfulness` | 5,818 | 0.4270 | 0.4129 | +0.0141 |
| `verbosity` | 5,818 | 0.6024 | 0.5877 | +0.0148 |
| `complexity` | 5,818 | 0.5205 | 0.4697 | +0.0507 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators | 29090 | 0.544 | 0.547 | +0.004 | 0.0083 | 0.0101 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 29090 | 0.544 | 0.547 | +0.004 | 0.0083 | 0.0101 |

## Is this number evidence?

On these 29090 predictions a perfectly calibrated model scores a mean ECE of 0.0067 (95th percentile 0.0096), simulated over 200 resamples. The measured ECE is 0.0083.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   440  0.247  0.284  -0.037  ##########|.............................
  [0.27,0.33)  1339  0.304  0.311  -0.007  ############|...........................
  [0.33,0.40)  2306  0.370  0.381  -0.011  ###############|........................
  [0.40,0.47)  4047  0.436  0.421  +0.015  #################|......................
  [0.47,0.53)  5458  0.500  0.485  +0.014  ###################.|...................
  [0.53,0.60)  4889  0.566  0.561  +0.005  ######################.|................
  [0.60,0.67)  4249  0.632  0.636  -0.003  #########################|..............
  [0.67,0.73)  4026  0.699  0.698  +0.001  ############################|...........
  [0.73,0.80)  2063  0.760  0.760  -0.000  ##############################|.........
  [0.80,0.87)   268  0.818  0.799  +0.020  ################################.|......
  [0.87,0.93)     5  0.878  0.600  +0.278  ########################...........|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
