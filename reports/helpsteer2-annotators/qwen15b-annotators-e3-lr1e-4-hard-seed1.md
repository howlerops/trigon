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
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2-annotators -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000 --hard-labels
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+f6584f43

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators/uncalibrated | 5818 | 0.5466 | 0.0800 | 0.0800 | 0.6084 | 57.6 | 154.1 | 591 |
| helpsteer2-annotators/calibrated | 5818 | 0.5466 | 0.0065 | 0.0104 | 0.5989 | 56.2 | 152.8 | 591 |

## Release gates

- PASS sample_size: 29090.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0100 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0066 on this run, p95 0.0100
- FAIL accuracy_over_baseline: 0.0211 (limit 0.0500) -- model 0.5466 vs marginal predictor 0.5255; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0009 (limit 0.0500) -- worst is 'coherence' at 0.6918 vs its own marginal 0.6927; the pooled gate hides this
- PASS workhorse_ece: 0.0065 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0104 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0065 (limit 0.0500) -- worst is 'score' at ECE 0.0065 (overconfidence +0.005); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2-annotators/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,818 | 0.6918 | 0.6927 | -0.0009 ⚠ |
| `verbosity` | 5,818 | 0.6030 | 0.5877 | +0.0153 |
| `correctness` | 5,818 | 0.4809 | 0.4648 | +0.0162 |
| `helpfulness` | 5,818 | 0.4319 | 0.4129 | +0.0191 |
| `complexity` | 5,818 | 0.5254 | 0.4697 | +0.0557 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators | 29090 | 0.547 | 0.551 | +0.005 | 0.0065 | 0.0104 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 29090 | 0.547 | 0.551 | +0.005 | 0.0065 | 0.0104 |

## Is this number evidence?

On these 29090 predictions a perfectly calibrated model scores a mean ECE of 0.0067 (95th percentile 0.0096), simulated over 200 resamples. The measured ECE is 0.0800.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   340  0.248  0.221  +0.028  #########.|.............................
  [0.27,0.33)   887  0.303  0.259  +0.043  ##########..|...........................
  [0.33,0.40)  1359  0.369  0.351  +0.018  ##############.|........................
  [0.40,0.47)  2095  0.435  0.385  +0.051  ###############..|......................
  [0.47,0.53)  3486  0.501  0.449  +0.053  ##################..|...................
  [0.53,0.60)  4220  0.567  0.497  +0.070  ####################...|................
  [0.60,0.67)  4516  0.633  0.550  +0.083  ######################...|..............
  [0.67,0.73)  4079  0.699  0.589  +0.110  ########################....|...........
  [0.73,0.80)  3528  0.766  0.672  +0.094  ###########################....|........
  [0.80,0.87)  3015  0.832  0.730  +0.102  #############################....|......
  [0.87,0.93)  1493  0.891  0.756  +0.135  ##############################......|...
  [0.93,1.00)    72  0.946  0.847  +0.099  ##################################....|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
