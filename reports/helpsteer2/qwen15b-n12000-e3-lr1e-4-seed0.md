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
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+366f8130

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5818 | 0.0577 | 0.0576 | 0.5502 | 58.8 | 150.9 | 583 |
| helpsteer2/calibrated | 5000 | 0.5765 | 0.0179 | 0.0242 | 0.5489 | 58.7 | 149.7 | 583 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0095 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0065 on this run, p95 0.0095
- FAIL accuracy_over_baseline: 0.0241 (limit 0.0500) -- model 0.5765 vs marginal predictor 0.5524; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0028 (limit 0.0500) -- worst is 'helpfulness' at 0.4066 vs its own marginal 0.4094; the pooled gate hides this
- PASS workhorse_ece: 0.0179 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0242 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0179 (limit 0.0500) -- worst is 'score' at ECE 0.0179 (overconfidence +0.018); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `helpfulness` | 5,000 | 0.4066 | 0.4094 | -0.0028 ⚠ |
| `coherence` | 5,000 | 0.7118 | 0.7120 | -0.0002 ⚠ |
| `correctness` | 5,000 | 0.4816 | 0.4784 | +0.0032 |
| `verbosity` | 5,000 | 0.6680 | 0.6292 | +0.0388 |
| `complexity` | 5,000 | 0.6144 | 0.5330 | +0.0814 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.576 | 0.594 | +0.018 | 0.0179 | 0.0242 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.576 | 0.594 | +0.018 | 0.0179 | 0.0242 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0071 (95th percentile 0.0099), simulated over 200 resamples. The measured ECE is 0.0577.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   111  0.245  0.261  -0.016  ##########|.............................
  [0.27,0.33)   471  0.305  0.261  +0.044  ##########..|...........................
  [0.33,0.40)   937  0.370  0.302  +0.068  ############...|........................
  [0.40,0.47)  1671  0.436  0.379  +0.057  ###############..|......................
  [0.47,0.53)  3409  0.503  0.453  +0.050  ##################..|...................
  [0.53,0.60)  3866  0.566  0.511  +0.056  ####################...|................
  [0.60,0.67)  3340  0.632  0.559  +0.073  ######################...|..............
  [0.67,0.73)  2971  0.701  0.641  +0.060  ##########################..|...........
  [0.73,0.80)  3669  0.768  0.727  +0.041  #############################..|........
  [0.80,0.87)  3799  0.831  0.766  +0.065  ###############################..|......
  [0.87,0.93)   755  0.883  0.809  +0.074  ################################...|....
  [0.93,1.00)     1  0.934  1.000  -0.066  #####################################|##
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
