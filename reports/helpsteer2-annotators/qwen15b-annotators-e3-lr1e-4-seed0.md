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
| ignores its input | 0.6340 | 1.2461 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2-annotators -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
```

**One seed is one sample from a distribution nobody measured.**
See `CLAUDE.md`: certify on the median and the spread, never on a
single draw. This report is a measurement, not a certification.
# Eval report

Model(s): trigon-qwen2.5-1.5b-0.1.0+1b2315cb

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators/uncalibrated | 5818 | 0.5496 | 0.0082 | 0.0119 | 0.5923 | 57.3 | 150.2 | 591 |
| helpsteer2-annotators/calibrated | 5818 | 0.5496 | 0.0082 | 0.0119 | 0.5923 | 56.1 | 149.1 | 591 |

## Release gates

- PASS sample_size: 29090.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0094 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0065 on this run, p95 0.0094
- FAIL accuracy_over_baseline: 0.0241 (limit 0.0500) -- model 0.5496 vs marginal predictor 0.5255; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0002 (limit 0.0500) -- worst is 'coherence' at 0.6925 vs its own marginal 0.6927; the pooled gate hides this
- PASS workhorse_ece: 0.0082 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0119 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0082 (limit 0.0500) -- worst is 'score' at ECE 0.0082 (overconfidence +0.008); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2-annotators/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,818 | 0.6925 | 0.6927 | -0.0002 ⚠ |
| `correctness` | 5,818 | 0.4777 | 0.4648 | +0.0129 |
| `helpfulness` | 5,818 | 0.4323 | 0.4129 | +0.0194 |
| `verbosity` | 5,818 | 0.6110 | 0.5877 | +0.0234 |
| `complexity` | 5,818 | 0.5345 | 0.4697 | +0.0648 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2-annotators | 29090 | 0.550 | 0.557 | +0.008 | 0.0082 | 0.0119 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 29090 | 0.550 | 0.557 | +0.008 | 0.0082 | 0.0119 |

## Is this number evidence?

On these 29090 predictions a perfectly calibrated model scores a mean ECE of 0.0065 (95th percentile 0.0094), simulated over 200 resamples. The measured ECE is 0.0082.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   506  0.244  0.215  +0.028  #########.|.............................
  [0.27,0.33)  1154  0.303  0.292  +0.011  ############|...........................
  [0.33,0.40)  2088  0.370  0.360  +0.010  ##############.|........................
  [0.40,0.47)  3466  0.436  0.429  +0.007  #################|......................
  [0.47,0.53)  5182  0.500  0.487  +0.013  ###################.|...................
  [0.53,0.60)  5241  0.566  0.557  +0.009  ######################.|................
  [0.60,0.67)  4867  0.632  0.627  +0.005  #########################|..............
  [0.67,0.73)  3765  0.699  0.697  +0.002  ############################|...........
  [0.73,0.80)  2329  0.761  0.756  +0.005  ##############################|.........
  [0.80,0.87)   395  0.823  0.833  -0.010  #################################|......
  [0.87,0.93)    85  0.890  0.941  -0.051  ####################################|#..
  [0.93,1.00)    12  0.938  0.917  +0.022  #####################################.|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
