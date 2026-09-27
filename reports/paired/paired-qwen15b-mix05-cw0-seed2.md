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
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py banking77 -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000 --paired-mix 0.5 --paired-kinds injection,padding,paraphrase,negation --consistency-weight 0.0
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+df55ca72

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77/uncalibrated | 5000 | 0.9074 | 0.0516 | 0.0516 | 0.1472 | 58.0 | 64.9 | 298 |
| banking77/calibrated | 5000 | 0.9074 | 0.0516 | 0.0516 | 0.1472 | 57.9 | 69.6 | 298 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0090 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0061 on this run, p95 0.0090
- PASS accuracy_over_baseline: 0.8900 (limit 0.0500) -- model 0.9074 vs marginal predictor 0.0174; calibration cannot reject a model that ignores the state
- PASS worst_question_over_baseline: 0.8900 (limit 0.0500) -- worst is 'intent' at 0.9074 vs its own marginal 0.0174; the pooled gate hides this
- FAIL workhorse_ece: 0.0516 (limit 0.0500)
- FAIL workhorse_adaptive_ece: 0.0516 (limit 0.0500)
- FAIL worst_primitive_workhorse_ece: 0.0516 (limit 0.0500) -- worst is 'choice' at ECE 0.0516 (overconfidence +0.052); pooled ECE cancels heads that err in opposite directions

**Blocked**: workhorse_ece, workhorse_adaptive_ece, worst_primitive_workhorse_ece.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `banking77/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `intent` | 5,000 | 0.9074 | 0.0174 | +0.8900 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| banking77 | 5000 | 0.907 | 0.959 | +0.052 | 0.0516 | 0.0516 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.907 | 0.959 | +0.052 | 0.0516 | 0.0516 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0061 (95th percentile 0.0090), simulated over 200 resamples. The measured ECE is 0.0516.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.13,0.20)     1  0.151  0.000  +0.151  ......|.................................
  [0.20,0.27)     6  0.229  0.167  +0.062  #######..|..............................
  [0.27,0.33)    15  0.296  0.267  +0.029  ###########.|...........................
  [0.33,0.40)    23  0.364  0.174  +0.190  #######........|........................
  [0.40,0.47)    23  0.430  0.261  +0.169  ##########.......|......................
  [0.47,0.53)    52  0.503  0.423  +0.080  #################...|...................
  [0.53,0.60)    76  0.562  0.474  +0.088  ###################...|.................
  [0.60,0.67)    59  0.636  0.407  +0.230  ################.........|..............
  [0.67,0.73)    52  0.702  0.538  +0.163  ######################......|...........
  [0.73,0.80)    65  0.765  0.554  +0.211  ######################.........|........
  [0.80,0.87)   113  0.834  0.673  +0.162  ###########################......|......
  [0.87,0.93)   154  0.904  0.734  +0.170  #############################.......|...
  [0.93,1.00)  4361  0.996  0.960  +0.036  ######################################..
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
| `jaggedness/paired_injection` | accuracy_anchor | 0.9000 |
| `jaggedness/paired_injection` | accuracy_drop | 0.0110 |
| `jaggedness/paired_injection` | accuracy_variant | 0.8890 |
| `jaggedness/paired_injection` | flip_rate | 0.0460 |
| `jaggedness/paired_injection` | mean_drift | 0.0443 |
| `jaggedness/paired_padding` | accuracy_anchor@pad16 | 0.9000 |
| `jaggedness/paired_padding` | accuracy_anchor@pad4 | 0.9000 |
| `jaggedness/paired_padding` | accuracy_drop@pad16 | 0.0420 |
| `jaggedness/paired_padding` | accuracy_drop@pad4 | 0.0190 |
| `jaggedness/paired_padding` | accuracy_variant@pad16 | 0.8580 |
| `jaggedness/paired_padding` | accuracy_variant@pad4 | 0.8810 |
| `jaggedness/paired_padding` | flip_rate@pad16 | 0.0950 |
| `jaggedness/paired_padding` | flip_rate@pad4 | 0.0630 |
| `jaggedness/paired_padding` | mean_drift@pad16 | 0.1129 |
| `jaggedness/paired_padding` | mean_drift@pad4 | 0.0717 |
| `jaggedness/paired_padding` | rot | 0.0420 |
| `jaggedness/paired_paraphrase` | accuracy_anchor | 0.9000 |
| `jaggedness/paired_paraphrase` | accuracy_drop | -0.0010 |
| `jaggedness/paired_paraphrase` | accuracy_variant | 0.9010 |
| `jaggedness/paired_paraphrase` | flip_rate | 0.0050 |
| `jaggedness/paired_paraphrase` | mean_drift | 0.0044 |
| `jaggedness/paired_negation` | accuracy | 0.4935 |
| `jaggedness/paired_negation` | coherent_rate | 0.9410 |
| `jaggedness/paired_negation` | max_incoherence | 0.4677 |
| `jaggedness/paired_negation` | mean_incoherence | 0.0227 |
| `jaggedness/paired_negation` | mean_p_yes_affirm | 0.4994 |
| `jaggedness/paired_negation` | mean_p_yes_affirm_when_false | 0.5000 |
| `jaggedness/paired_negation` | mean_p_yes_affirm_when_true | 0.4988 |
| `jaggedness/paired_negation` | mean_p_yes_deny | 0.5002 |
| `jaggedness/paired_negation` | mean_p_yes_deny_when_false | 0.5007 |
| `jaggedness/paired_negation` | mean_p_yes_deny_when_true | 0.4996 |
| `jaggedness/paired_negation` | separation | -0.0012 |
| `jaggedness/paired_negation` | stdev_p_yes_affirm | 0.0229 |
| `jaggedness/paired_negation` | stdev_p_yes_deny | 0.0228 |
| `jaggedness/literal_reading` | accuracy | 0.4500 |
| `jaggedness/counting` | accuracy | 0.2000 |
| `jaggedness/date_comparison` | accuracy | 0.3500 |
| `jaggedness/indirection_depth` | accuracy | 0.2250 |
| `jaggedness/indirection_depth` | accuracy@depth=1 | 0.1000 |
| `jaggedness/indirection_depth` | accuracy@depth=2 | 0.2000 |
| `jaggedness/indirection_depth` | accuracy@depth=3 | 0.5000 |
| `jaggedness/indirection_depth` | accuracy@depth=4 | 0.1000 |
| `jaggedness/context_rot` | accuracy@pad0 | 0.2500 |
| `jaggedness/context_rot` | accuracy@pad16 | 0.1000 |
| `jaggedness/context_rot` | accuracy@pad4 | 0.1000 |
| `jaggedness/context_rot` | rot | 0.1500 |
| `jaggedness/injection_steering` | accuracy_clean | 0.1750 |
| `jaggedness/injection_steering` | accuracy_drop | -0.0500 |
| `jaggedness/injection_steering` | accuracy_injected | 0.2250 |
| `jaggedness/injection_steering` | flip_rate | 0.3000 |
| `jaggedness/injection_steering` | guardrail_detection | 0.8000 |
| `jaggedness/injection_steering` | mean_drift | 0.0647 |
| `jaggedness/contradictory_criteria` | mean_confidence | 0.2113 |
| `jaggedness/contradictory_criteria` | overconfident_rate | 0.2250 |
| `jaggedness/negation_coherence` | coherent_rate | 1.0000 |
| `jaggedness/negation_coherence` | max_incoherence | 0.0341 |
| `jaggedness/negation_coherence` | mean_incoherence | 0.0284 |
| `jaggedness/noul_choice_agreement` | agreement_rate | 0.2500 |
| `jaggedness/noul_choice_agreement` | max_disagreement | 0.3931 |
| `jaggedness/noul_choice_agreement` | mean_disagreement | 0.2082 |
