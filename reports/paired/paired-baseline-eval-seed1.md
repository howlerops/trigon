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
| `intent` | 0.0124 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.9873 | 4.3603 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 4 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+228ae1aa

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.8980 | 0.0448 | 0.0445 | 0.1618 | 62.6 | 66.1 | 298 |
| banking77/calibrated | 5000 | 0.8980 | 0.0448 | 0.0445 | 0.1618 | 62.9 | 66.9 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0102 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0073 on this run, p95 0.0102
- PASS accuracy_over_baseline: 0.8814 (limit 0.0500) -- model 0.8980 vs marginal predictor 0.0166; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.8814 (limit 0.0500) -- worst is 'intent' at 0.8980 vs its own marginal 0.0166; the pooled gate hides this
- PASS workhorse_ece: 0.0448 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0445 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0448 (limit 0.0500) -- worst is 'choice' at ECE 0.0448 (overconfidence +0.044); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.8980 | 0.0166 | +0.8814 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.898 | 0.942 | +0.044 | 0.0448 | 0.0445 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.898 | 0.942 | +0.044 | 0.0448 | 0.0445 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0073 (95th percentile 0.0102), simulated over 200 resamples. The measured ECE is 0.0448.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.07,0.13)     3  0.123  0.000  +0.123  .....|..................................
  [0.13,0.20)     3  0.182  0.333  -0.151  #######|#####...........................
  [0.20,0.27)     5  0.234  0.200  +0.034  ########.|..............................
  [0.27,0.33)    17  0.296  0.235  +0.060  #########...|...........................
  [0.33,0.40)    26  0.375  0.308  +0.068  ############...|........................
  [0.40,0.47)    34  0.434  0.441  -0.008  #################|......................
  [0.47,0.53)    78  0.502  0.474  +0.028  ###################.|...................
  [0.53,0.60)    85  0.567  0.400  +0.167  ################.......|................
  [0.60,0.67)    79  0.635  0.608  +0.027  ########################.|..............
  [0.67,0.73)    95  0.701  0.547  +0.153  ######################......|...........
  [0.73,0.80)    94  0.769  0.543  +0.227  ######################.........|........
  [0.80,0.87)   169  0.838  0.663  +0.175  ###########################.......|.....
  [0.87,0.93)   245  0.906  0.792  +0.114  ################################....|...
  [0.93,1.00)  4067  0.994  0.967  +0.027  #######################################.
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
| `jaggedness/paired_injection` | accuracy_anchor | 0.9080 |
| `jaggedness/paired_injection` | accuracy_drop | 0.4290 |
| `jaggedness/paired_injection` | accuracy_variant | 0.4790 |
| `jaggedness/paired_injection` | flip_rate | 0.4940 |
| `jaggedness/paired_injection` | mean_drift | 0.5211 |
| `jaggedness/paired_padding` | accuracy_anchor@pad16 | 0.9080 |
| `jaggedness/paired_padding` | accuracy_anchor@pad4 | 0.9080 |
| `jaggedness/paired_padding` | accuracy_drop@pad16 | 0.8490 |
| `jaggedness/paired_padding` | accuracy_drop@pad4 | 0.6810 |
| `jaggedness/paired_padding` | accuracy_variant@pad16 | 0.0590 |
| `jaggedness/paired_padding` | accuracy_variant@pad4 | 0.2270 |
| `jaggedness/paired_padding` | flip_rate@pad16 | 0.9410 |
| `jaggedness/paired_padding` | flip_rate@pad4 | 0.7660 |
| `jaggedness/paired_padding` | mean_drift@pad16 | 0.9456 |
| `jaggedness/paired_padding` | mean_drift@pad4 | 0.7896 |
| `jaggedness/paired_padding` | rot | 0.8490 |
| `jaggedness/paired_paraphrase` | accuracy_anchor | 0.9080 |
| `jaggedness/paired_paraphrase` | accuracy_drop | 0.0020 |
| `jaggedness/paired_paraphrase` | accuracy_variant | 0.9060 |
| `jaggedness/paired_paraphrase` | flip_rate | 0.0120 |
| `jaggedness/paired_paraphrase` | mean_drift | 0.0106 |
| `jaggedness/paired_negation` | accuracy | 0.4980 |
| `jaggedness/paired_negation` | coherent_rate | 0.0510 |
| `jaggedness/paired_negation` | max_incoherence | 0.9847 |
| `jaggedness/paired_negation` | mean_incoherence | 0.4724 |
| `jaggedness/literal_reading` | accuracy | 0.7500 |
| `jaggedness/counting` | accuracy | 0.1500 |
| `jaggedness/date_comparison` | accuracy | 0.5250 |
| `jaggedness/indirection_depth` | accuracy | 0.2250 |
| `jaggedness/indirection_depth` | accuracy@depth=1 | 0.3000 |
| `jaggedness/indirection_depth` | accuracy@depth=2 | 0.2000 |
| `jaggedness/indirection_depth` | accuracy@depth=3 | 0.2000 |
| `jaggedness/indirection_depth` | accuracy@depth=4 | 0.2000 |
| `jaggedness/context_rot` | accuracy@pad0 | 0.2500 |
| `jaggedness/context_rot` | accuracy@pad16 | 0.3750 |
| `jaggedness/context_rot` | accuracy@pad4 | 0.2000 |
| `jaggedness/context_rot` | rot | -0.1250 |
| `jaggedness/injection_steering` | accuracy_clean | 0.2750 |
| `jaggedness/injection_steering` | accuracy_drop | -0.0250 |
| `jaggedness/injection_steering` | accuracy_injected | 0.3000 |
| `jaggedness/injection_steering` | flip_rate | 0.4000 |
| `jaggedness/injection_steering` | guardrail_detection | 0.0250 |
| `jaggedness/injection_steering` | mean_drift | 0.3526 |
| `jaggedness/contradictory_criteria` | mean_confidence | 0.6451 |
| `jaggedness/contradictory_criteria` | overconfident_rate | 0.7250 |
| `jaggedness/negation_coherence` | coherent_rate | 0.0000 |
| `jaggedness/negation_coherence` | max_incoherence | 0.4876 |
| `jaggedness/negation_coherence` | mean_incoherence | 0.2975 |
| `jaggedness/noul_choice_agreement` | agreement_rate | 0.2500 |
| `jaggedness/noul_choice_agreement` | max_disagreement | 0.7891 |
| `jaggedness/noul_choice_agreement` | mean_disagreement | 0.3766 |
