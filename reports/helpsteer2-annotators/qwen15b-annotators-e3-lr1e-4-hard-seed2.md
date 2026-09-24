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
python scripts/train_corpus.py helpsteer2-annotators -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000 --hard-labels
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+18a19144

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators/uncalibrated | 5818 | 0.5396 | 0.0447 | 0.0450 | 0.6105 | 54.4 | 151.9 | 591 |
| helpsteer2-annotators/calibrated | 5818 | 0.5387 | 0.0126 | 0.0147 | 0.6083 | 54.9 | 154.2 | 591 |

## Release gates

- PASS sample_size: 29090.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0088 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0058 on this run, p95 0.0088
- FAIL accuracy_over_baseline: 0.0131 (limit 0.0500) -- model 0.5387 vs marginal predictor 0.5255; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0015 (limit 0.0500) -- worst is 'correctness' at 0.4632 vs its own marginal 0.4648; the pooled gate hides this
- PASS workhorse_ece: 0.0126 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0147 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0126 (limit 0.0500) -- worst is 'score' at ECE 0.0126 (overconfidence -0.003); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2-annotators/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `correctness` | 5,818 | 0.4632 | 0.4648 | -0.0015 ⚠ |
| `coherence` | 5,818 | 0.6928 | 0.6927 | +0.0002 |
| `helpfulness` | 5,818 | 0.4144 | 0.4129 | +0.0015 |
| `verbosity` | 5,818 | 0.5987 | 0.5877 | +0.0110 |
| `complexity` | 5,818 | 0.5242 | 0.4697 | +0.0545 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators | 29090 | 0.539 | 0.535 | -0.003 | 0.0126 | 0.0147 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 29090 | 0.539 | 0.535 | -0.003 | 0.0126 | 0.0147 |

## Is this number evidence?

On these 29090 predictions a perfectly calibrated model scores a mean ECE of 0.0068 (95th percentile 0.0096), simulated over 200 resamples. The measured ECE is 0.0447.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   437  0.250  0.233  +0.017  #########.|.............................
  [0.27,0.33)  1629  0.302  0.293  +0.009  ############|...........................
  [0.33,0.40)  2445  0.368  0.376  -0.008  ###############|........................
  [0.40,0.47)  3105  0.434  0.429  +0.005  #################|......................
  [0.47,0.53)  4163  0.500  0.482  +0.019  ###################.|...................
  [0.53,0.60)  3904  0.566  0.525  +0.041  #####################..|................
  [0.60,0.67)  3505  0.633  0.591  +0.043  ########################.|..............
  [0.67,0.73)  3472  0.700  0.616  +0.083  #########################...|...........
  [0.73,0.80)  3235  0.766  0.695  +0.071  ############################...|........
  [0.80,0.87)  2605  0.830  0.731  +0.100  #############################....|......
  [0.87,0.93)   578  0.886  0.758  +0.129  ##############################.....|....
  [0.93,1.00)    12  0.945  0.750  +0.195  ##############################........|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
