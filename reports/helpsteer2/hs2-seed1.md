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
| `coherence` | 0.7184 |
| `complexity` | 0.5242 |
| `correctness` | 0.4802 |
| `helpfulness` | 0.4194 |
| `verbosity` | 0.6256 |
| Epochs | 2 |
| Seed | 1 |

```
python scripts/train_corpus.py helpsteer2 -n 1400 --calibration-n 1000 --eval-n 0 --epochs 2 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1
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

Model(s): trigon-reference-0.1.0+7b3a9cc9

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2/uncalibrated | 5000 | 0.5559 | 0.0144 | 0.0258 | 0.5924 | 45.0 | 301.7 | 1290 |
| helpsteer2/calibrated | 5000 | 0.5559 | 0.0144 | 0.0258 | 0.5924 | 45.0 | 298.5 | 1290 |

## Release gates

- PASS sample_size: 25000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0090 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0056 on this run, p95 0.0090
- FAIL accuracy_over_baseline: 0.0023 (limit 0.0500) -- model 0.5559 vs marginal predictor 0.5536; calibration cannot reject a model that ignores the state
- FAIL (advisory) worst_question_over_baseline: 0.0000 (limit 0.0500) -- worst is 'coherence' at 0.7184 vs its own marginal 0.7184; the pooled gate hides this
- PASS workhorse_ece: 0.0144 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0258 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0144 (limit 0.0500) -- worst is 'score' at ECE 0.0144 (overconfidence -0.014); pooled ECE cancels heads that err in opposite directions

**Blocked**: accuracy_over_baseline.

**Advisory, not blocking**: worst_question_over_baseline. Reported so the run does not read as clean when it is not — see `docs/evals.md` for what turns each one on.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `helpsteer2/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `coherence` | 5,000 | 0.7184 | 0.7184 | +0.0000 |
| `correctness` | 5,000 | 0.4802 | 0.4802 | +0.0000 |
| `helpfulness` | 5,000 | 0.4194 | 0.4194 | +0.0000 |
| `verbosity` | 5,000 | 0.6256 | 0.6256 | +0.0000 |
| `complexity` | 5,000 | 0.5358 | 0.5242 | +0.0116 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helpsteer2 | 25000 | 0.556 | 0.542 | -0.014 | 0.0144 | 0.0258 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score | 25000 | 0.556 | 0.542 | -0.014 | 0.0144 | 0.0258 |

## Is this number evidence?

On these 25000 predictions a perfectly calibrated model scores a mean ECE of 0.0056 (95th percentile 0.0090), simulated over 200 resamples. The measured ECE is 0.0144.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)  2422  0.393  0.397  -0.004  ################|.......................
  [0.40,0.47)  7841  0.439  0.462  -0.023  ##################|.....................
  [0.47,0.53)  2453  0.510  0.514  -0.004  ####################|...................
  [0.53,0.60)  2420  0.545  0.557  -0.012  ######################|.................
  [0.60,0.67)  5555  0.650  0.648  +0.002  ##########################|.............
  [0.67,0.73)  4309  0.691  0.721  -0.029  ############################|...........
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```
