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
python scripts/train_corpus.py helpsteer2-annotators -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+c5a2c647

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators/uncalibrated | 5818 | 0.5464 | 0.0271 | 0.0269 | 0.5982 | 57.8 | 149.3 | 591 |
| helpsteer2-annotators/calibrated | 5818 | 0.5464 | 0.0271 | 0.0269 | 0.5982 | 57.7 | 149.2 | 591 |

## Release gates

- PASS sample_size: 29090.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0094 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0063 on this run, p95 0.0094
- FAIL accuracy_over_baseline: 0.0209 (limit 0.0500) -- model 0.5464 vs marginal predictor 0.5255; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0000 (limit 0.0500) -- worst is 'coherence' at 0.6927 vs its own marginal 0.6927; the pooled gate hides this
- PASS workhorse_ece: 0.0271 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0269 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0271 (limit 0.0500) -- worst is 'score' at ECE 0.0271 (overconfidence +0.027); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2-annotators/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,818 | 0.6927 | 0.6927 | +0.0000 |
| `correctness` | 5,818 | 0.4761 | 0.4648 | +0.0113 |
| `helpfulness` | 5,818 | 0.4264 | 0.4129 | +0.0136 |
| `verbosity` | 5,818 | 0.6055 | 0.5877 | +0.0179 |
| `complexity` | 5,818 | 0.5313 | 0.4697 | +0.0615 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators | 29090 | 0.546 | 0.573 | +0.027 | 0.0271 | 0.0269 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 29090 | 0.546 | 0.573 | +0.027 | 0.0271 | 0.0269 |

## Is this number evidence?

On these 29090 predictions a perfectly calibrated model scores a mean ECE of 0.0063 (95th percentile 0.0094), simulated over 200 resamples. The measured ECE is 0.0271.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   254  0.247  0.256  -0.009  ##########|.............................
  [0.27,0.33)   889  0.304  0.297  +0.007  ############|...........................
  [0.33,0.40)  1826  0.370  0.325  +0.045  #############..|........................
  [0.40,0.47)  3368  0.436  0.418  +0.018  #################|......................
  [0.47,0.53)  5102  0.500  0.466  +0.034  ###################.|...................
  [0.53,0.60)  5107  0.566  0.537  +0.030  #####################..|................
  [0.60,0.67)  4519  0.632  0.605  +0.027  ########################.|..............
  [0.67,0.73)  4142  0.700  0.682  +0.018  ###########################.|...........
  [0.73,0.80)  3258  0.763  0.727  +0.035  #############################..|........
  [0.80,0.87)   588  0.818  0.816  +0.002  #################################|......
  [0.87,0.93)    37  0.888  0.919  -0.031  ####################################|...
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
