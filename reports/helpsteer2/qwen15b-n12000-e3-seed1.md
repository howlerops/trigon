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
| `coherence` | 0.7184 |
| `complexity` | 0.5242 |
| `correctness` | 0.4802 |
| `helpfulness` | 0.4194 |
| `verbosity` | 0.6256 |
| Epochs | 3 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+6bc83d52

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5744 | 0.0120 | 0.0135 | 0.5490 | 59.5 | 146.8 | 586 |
| helpsteer2/calibrated | 5000 | 0.5744 | 0.0120 | 0.0135 | 0.5490 | 59.6 | 145.7 | 586 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0102 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0073 on this run, p95 0.0102
- FAIL accuracy_over_baseline: 0.0209 (limit 0.0500) -- model 0.5744 vs marginal predictor 0.5536; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0126 (limit 0.0500) -- worst is 'correctness' at 0.4676 vs its own marginal 0.4802; the pooled gate hides this
- PASS workhorse_ece: 0.0120 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0135 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0120 (limit 0.0500) -- worst is 'score' at ECE 0.0120 (overconfidence +0.005); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `correctness` | 5,000 | 0.4676 | 0.4802 | -0.0126 ⚠ |
| `helpfulness` | 5,000 | 0.4102 | 0.4194 | -0.0092 ⚠ |
| `coherence` | 5,000 | 0.7186 | 0.7184 | +0.0002 |
| `verbosity` | 5,000 | 0.6596 | 0.6256 | +0.0340 |
| `complexity` | 5,000 | 0.6162 | 0.5242 | +0.0920 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.574 | 0.579 | +0.005 | 0.0120 | 0.0135 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.574 | 0.579 | +0.005 | 0.0120 | 0.0135 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0073 (95th percentile 0.0102), simulated over 200 resamples. The measured ECE is 0.0120.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   409  0.248  0.220  +0.028  #########.|.............................
  [0.27,0.33)  1313  0.303  0.299  +0.004  ############|...........................
  [0.33,0.40)  1938  0.369  0.385  -0.016  ###############|........................
  [0.40,0.47)  2615  0.435  0.439  -0.004  #################|......................
  [0.47,0.53)  3710  0.501  0.490  +0.010  ####################|...................
  [0.53,0.60)  3323  0.566  0.548  +0.018  ######################.|................
  [0.60,0.67)  3180  0.635  0.632  +0.003  #########################|..............
  [0.67,0.73)  3807  0.700  0.708  -0.008  ############################|...........
  [0.73,0.80)  3076  0.763  0.770  -0.006  ###############################|........
  [0.80,0.87)  1428  0.827  0.774  +0.053  ###############################..|......
  [0.87,0.93)   201  0.884  0.836  +0.048  #################################..|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
