# banking77

**Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. https://github.com/PolyAI-LDN/task-specific-datasets**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 7,083 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 3,080 |
| — held out of train to reach the floor | 1,920 |
| Questions per request | 1 |
| Labels per question | 77 |

| Question | Marginal predictor |
| --- | ---: |
| `intent` | 0.0144 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.9873 | 4.3595 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 4 |
| Seed | 2 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+8006141f

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.8720 | 0.0370 | 0.0354 | 0.1929 | 59.9 | 71.6 | 298 |
| banking77/calibrated | 5000 | 0.8720 | 0.0370 | 0.0354 | 0.1929 | 59.9 | 66.1 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0126 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0092 on this run, p95 0.0126
- PASS accuracy_over_baseline: 0.8546 (limit 0.0500) -- model 0.8720 vs marginal predictor 0.0174; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.8546 (limit 0.0500) -- worst is 'intent' at 0.8720 vs its own marginal 0.0174; the pooled gate hides this
- PASS workhorse_ece: 0.0370 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0354 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0370 (limit 0.0500) -- worst is 'choice' at ECE 0.0370 (overconfidence +0.035); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.8720 | 0.0174 | +0.8546 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.872 | 0.907 | +0.035 | 0.0370 | 0.0354 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.872 | 0.907 | +0.035 | 0.0370 | 0.0354 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0092 (95th percentile 0.0126), simulated over 200 resamples. The measured ECE is 0.0370.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.13,0.20)     4  0.161  0.000  +0.161  ......|.................................
  [0.20,0.27)    18  0.244  0.222  +0.022  #########.|.............................
  [0.27,0.33)    40  0.303  0.250  +0.053  ##########..|...........................
  [0.33,0.40)    49  0.368  0.449  -0.081  ###############|##......................
  [0.40,0.47)    85  0.436  0.400  +0.036  ################.|......................
  [0.47,0.53)   114  0.500  0.386  +0.114  ###############.....|...................
  [0.53,0.60)   114  0.566  0.544  +0.022  ######################.|................
  [0.60,0.67)   142  0.632  0.599  +0.033  ########################.|..............
  [0.67,0.73)   127  0.701  0.591  +0.110  ########################....|...........
  [0.73,0.80)   162  0.770  0.648  +0.121  ##########################.....|........
  [0.80,0.87)   222  0.836  0.739  +0.097  ##############################...|......
  [0.87,0.93)   336  0.904  0.801  +0.103  ################################....|...
  [0.93,1.00)  3587  0.990  0.972  +0.018  #######################################.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```

## Robustness: the paired benchmarks

Each `paired_*` row derives pairs from the first 1,000 evaluation cases (seeded shuffle) with templates
from the evaluation pool, which no training case used. Flip rate and
drift (total variation) compare each answer with the anchor it was derived
from; incoherence is |P(yes) + P(yes on the complement) - 1|. Read every
held-fixed metric beside accuracy: a model that ignores its input never
flips. The generic `jaggedness/*` rows, when present, ask a four-option
support schema this model was not trained on.

| Benchmark | Metric | Value |
| --- | --- | ---: |
| `jaggedness/paired_injection` | accuracy_anchor | 0.8840 |
| `jaggedness/paired_injection` | accuracy_drop | 0.0150 |
| `jaggedness/paired_injection` | accuracy_variant | 0.8690 |
| `jaggedness/paired_injection` | flip_rate | 0.0480 |
| `jaggedness/paired_injection` | mean_drift | 0.0524 |
| `jaggedness/paired_padding` | accuracy_anchor@pad16 | 0.8840 |
| `jaggedness/paired_padding` | accuracy_anchor@pad4 | 0.8840 |
| `jaggedness/paired_padding` | accuracy_drop@pad16 | 0.0490 |
| `jaggedness/paired_padding` | accuracy_drop@pad4 | 0.0170 |
| `jaggedness/paired_padding` | accuracy_variant@pad16 | 0.8350 |
| `jaggedness/paired_padding` | accuracy_variant@pad4 | 0.8670 |
| `jaggedness/paired_padding` | flip_rate@pad16 | 0.1150 |
| `jaggedness/paired_padding` | flip_rate@pad4 | 0.0670 |
| `jaggedness/paired_padding` | mean_drift@pad16 | 0.1535 |
| `jaggedness/paired_padding` | mean_drift@pad4 | 0.0994 |
| `jaggedness/paired_padding` | rot | 0.0490 |
| `jaggedness/paired_paraphrase` | accuracy_anchor | 0.8840 |
| `jaggedness/paired_paraphrase` | accuracy_drop | -0.0020 |
| `jaggedness/paired_paraphrase` | accuracy_variant | 0.8860 |
| `jaggedness/paired_paraphrase` | flip_rate | 0.0030 |
| `jaggedness/paired_paraphrase` | mean_drift | 0.0041 |
| `jaggedness/paired_negation` | accuracy | 0.4980 |
| `jaggedness/paired_negation` | coherent_rate | 0.7360 |
| `jaggedness/paired_negation` | max_incoherence | 0.3685 |
| `jaggedness/paired_negation` | mean_incoherence | 0.0382 |
| `jaggedness/paired_negation` | mean_p_yes_affirm | 0.4960 |
| `jaggedness/paired_negation` | mean_p_yes_affirm_when_false | 0.4961 |
| `jaggedness/paired_negation` | mean_p_yes_affirm_when_true | 0.4958 |
| `jaggedness/paired_negation` | mean_p_yes_deny | 0.4954 |
| `jaggedness/paired_negation` | mean_p_yes_deny_when_false | 0.4954 |
| `jaggedness/paired_negation` | mean_p_yes_deny_when_true | 0.4954 |
| `jaggedness/paired_negation` | separation | -0.0002 |
| `jaggedness/paired_negation` | stdev_p_yes_affirm | 0.0260 |
| `jaggedness/paired_negation` | stdev_p_yes_deny | 0.0269 |
