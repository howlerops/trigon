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
| `coherence` | 0.7164 |
| `complexity` | 0.5286 |
| `correctness` | 0.4808 |
| `helpfulness` | 0.4186 |
| `verbosity` | 0.6210 |
| Epochs | 3 |
| Seed | 2 |
| Device | cuda (NVIDIA A10G) |
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+1668aa27

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5818 | 0.0105 | 0.0098 | 0.5454 | 75.4 | 165.6 | 587 |
| helpsteer2/calibrated | 5000 | 0.5818 | 0.0105 | 0.0098 | 0.5454 | 75.3 | 165.6 | 587 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0102 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0071 on this run, p95 0.0102
- FAIL accuracy_over_baseline: 0.0287 (limit 0.0500) -- model 0.5818 vs marginal predictor 0.5531; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0008 (limit 0.0500) -- worst is 'coherence' at 0.7172 vs its own marginal 0.7164; the pooled gate hides this
- PASS workhorse_ece: 0.0105 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0098 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0105 (limit 0.0500) -- worst is 'score' at ECE 0.0105 (overconfidence +0.006); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7172 | 0.7164 | +0.0008 |
| `helpfulness` | 5,000 | 0.4238 | 0.4186 | +0.0052 |
| `correctness` | 5,000 | 0.4866 | 0.4808 | +0.0058 |
| `verbosity` | 5,000 | 0.6650 | 0.6210 | +0.0440 |
| `complexity` | 5,000 | 0.6162 | 0.5286 | +0.0876 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.582 | 0.588 | +0.006 | 0.0105 | 0.0098 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.582 | 0.588 | +0.006 | 0.0105 | 0.0098 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0071 (95th percentile 0.0102), simulated over 200 resamples. The measured ECE is 0.0105.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.20,0.27)   466  0.248  0.212  +0.035  ########..|.............................
  [0.27,0.33)  1041  0.301  0.303  -0.002  ############|...........................
  [0.33,0.40)  1588  0.368  0.367  +0.001  ###############|........................
  [0.40,0.47)  2328  0.436  0.443  -0.007  #################|......................
  [0.47,0.53)  3701  0.501  0.481  +0.020  ###################.|...................
  [0.53,0.60)  3664  0.567  0.575  -0.008  #######################|................
  [0.60,0.67)  3469  0.634  0.637  -0.003  #########################|..............
  [0.67,0.73)  3833  0.700  0.698  +0.003  ############################|...........
  [0.73,0.80)  3395  0.764  0.751  +0.013  ##############################.|........
  [0.80,0.87)  1416  0.824  0.785  +0.039  ###############################..|......
  [0.87,0.93)    99  0.878  0.859  +0.019  ##################################.|....
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
