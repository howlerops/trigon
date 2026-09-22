# helpsteer2

**HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0. https://huggingface.co/datasets/nvidia/HelpSteer2**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 1,400 |
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
| Epochs | 2 |
| Seed | 3 |

```
python scripts/train_corpus.py helpsteer2 -n 1400 --calibration-n 1000 --eval-n 0 --epochs 2 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 3
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

Model(s): trigon-reference-0.1.0+2bee8875

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5482 | 0.0317 | 0.0429 | 0.6001 | 44.0 | 319.2 | 1281 |
| helpsteer2/calibrated | 5000 | 0.5463 | 0.0108 | 0.0118 | 0.5980 | 43.9 | 319.0 | 1281 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0084 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0053 on this run, p95 0.0084
- FAIL accuracy_over_baseline: -0.0020 (limit 0.0500) -- model 0.5463 vs marginal predictor 0.5482; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: -0.0122 (limit 0.0500) -- worst is 'helpfulness' at 0.3870 vs its own marginal 0.3992; the pooled gate hides this
- PASS workhorse_ece: 0.0108 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0118 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0108 (limit 0.0500) -- worst is 'score' at ECE 0.0108 (overconfidence +0.002); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `helpfulness` | 5,000 | 0.3870 | 0.3992 | -0.0122 ⚠ |
| `coherence` | 5,000 | 0.7104 | 0.7104 | +0.0000 |
| `correctness` | 5,000 | 0.4624 | 0.4624 | +0.0000 |
| `verbosity` | 5,000 | 0.6300 | 0.6300 | +0.0000 |
| `complexity` | 5,000 | 0.5416 | 0.5392 | +0.0024 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.546 | 0.548 | +0.002 | 0.0108 | 0.0118 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.546 | 0.548 | +0.002 | 0.0108 | 0.0118 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0053 (95th percentile 0.0085), simulated over 200 resamples. The measured ECE is 0.0317.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.40,0.47)  5001  0.446  0.399  +0.047  ################..|.....................
  [0.47,0.53)  6510  0.499  0.452  +0.048  ##################..|...................
  [0.53,0.60)  3534  0.541  0.586  -0.046  ######################|.................
  [0.60,0.67)  4955  0.628  0.635  -0.006  #########################|..............
  [0.67,0.73)  5000  0.699  0.710  -0.011  ############################|...........
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
