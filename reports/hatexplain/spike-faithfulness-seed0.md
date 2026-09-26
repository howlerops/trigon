# hatexplain

**HateXplain (Mathew et al., AAAI 2021), Punyajoy Saha and co-authors. MIT (repository LICENSE, Copyright (c) 2020 Punyajoy Saha); the authors' dataset card states CC BY 4.0. https://github.com/punyajoy/HateXplain at 01d742279dac**

Real labelled data, not the synthetic generator. There is no
Bayes-optimal loss to quote here -- the labels are human and the
corpus does not come with a noise rate -- so the floor reported
below is the marginal predictor, which is the floor that matters
for the `accuracy_over_baseline` gate.

| | |
| --- | ---: |
| Training cases | 13,229 |
| Calibration cases (held out of train) | 1,000 |
| Evaluation cases | 5,000 |
| — from the corpus's own test split | 4,789 |
| — held out of train to reach the floor | 211 |
| Questions per request | 1 |
| Labels per question | 3 |

| Question | Marginal predictor |
| --- | ---: |
| `label` | 0.4078 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6584 | 1.0866 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 4 |
| Seed | 0 |
| Device | cpu (x86_64) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 4 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device auto
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

Model(s): trigon-reference-0.1.0+8278b654

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.5798 | 0.0096 | 0.0223 | 0.5317 | 308.1 | 912.1 | 191 |
| hatexplain/calibrated | 5000 | 0.5798 | 0.0096 | 0.0223 | 0.5317 | 122.8 | 304.1 | 191 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0232 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0153 on this run, p95 0.0232
- PASS accuracy_over_baseline: 0.1720 (limit 0.0500) -- model 0.5798 vs marginal predictor 0.4078; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1720 (limit 0.0500) -- worst is 'label' at 0.5798 vs its own marginal 0.4078; the pooled gate hides this
- PASS workhorse_ece: 0.0096 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0223 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0096 (limit 0.0500) -- worst is 'choice' at ECE 0.0096 (overconfidence -0.006); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.5798 | 0.4078 | +0.1720 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.580 | 0.574 | -0.006 | 0.0096 | 0.0223 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.580 | 0.574 | -0.006 | 0.0096 | 0.0223 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0153 (95th percentile 0.0232), simulated over 200 resamples. The measured ECE is 0.0096.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)   645  0.375  0.398  -0.023  ###############|........................
  [0.40,0.47)   923  0.434  0.437  -0.003  #################|......................
  [0.47,0.53)   792  0.499  0.509  -0.010  ####################|...................
  [0.53,0.60)   634  0.566  0.573  -0.006  #######################|................
  [0.60,0.67)   556  0.632  0.629  +0.002  #########################|..............
  [0.67,0.73)   505  0.697  0.699  -0.002  ############################|...........
  [0.73,0.80)   444  0.766  0.752  +0.014  ##############################.|........
  [0.80,0.87)   279  0.831  0.824  +0.007  #################################|......
  [0.87,0.93)   197  0.895  0.929  -0.034  ####################################|...
  [0.93,1.00)    25  0.949  0.920  +0.029  #####################################.|.
  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.
```

## Evidence: plausibility against human rationales

What the model highlights, scored against what the annotators
highlighted, on words of the state. Token F1 is per-case F1 over
highlighted words, averaged; IOU F1 counts a predicted span as found
when it overlaps a human span by at least half their union. The three
rows after the model are floors, and the rationale lexicon is the one
that matters: every word highlighted in at least half its training
occurrences. A highlighter that does not beat it has learned a
vocabulary, not a reading. The model rows are one set of weights
under each evidence method; `model` is the one it serves.

| Highlighter | Method | Rationales | Token F1 | Precision | Recall | IOU F1 | Highlighted | Human |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model | `span_head` | 2,960 | 0.4875 | 0.5393 | 0.6455 | 0.3301 | 0.339 | 0.339 |
| model, gradient x input | `gradient_x_input` | 2,960 | 0.3247 | 0.5534 | 0.3184 | 0.2637 | 0.129 | 0.339 |
| model, integrated gradients | `integrated_gradients` | 2,960 | 0.3292 | 0.5252 | 0.3462 | 0.2619 | 0.147 | 0.339 |
| lexical floor | `lexical_overlap` | 2,960 | 0.0258 | 0.1297 | 0.0155 | 0.0016 | 0.027 | 0.339 |
| rationale lexicon | `fitted word list` | 2,960 | 0.5728 | 0.7650 | 0.6043 | 0.4544 | 0.172 | 0.339 |
| every word | `all words` | 2,960 | 0.4361 | 0.3387 | 1.0000 | 0.2185 | 1.000 | 0.339 |

Integrated gradients' completeness on the first 200 of these cases (32 points, `u ** 3` spacing): relative error of the summed attributions against the log-probability difference they must add up to, median 0.0035, p90 0.0187, max 0.3427, over the 195 whose difference is at least 0.01 nats (median difference 0.552). A large error means too few points, and the `integrated gradients` row is then a quadrature artefact, not the method.

## Evidence: faithfulness

Did the model use what it highlights? For the label it selects on the
full post, over the first 500 of these cases: **comprehensiveness**
is how far that label's probability falls when the top-k% of words by
the method's own scores are deleted from the post and the post is asked
again (higher: the words mattered); **sufficiency** is how far it falls
when only those words are kept (lower: they suffice). ERASER's AOPC, the
mean over k in 1%, 5%, 10%, 20%, 50%, on the model's uncalibrated distribution.
`random` and the `rationale lexicon` are controls scored the same way:
a method that does not beat `random` found nothing the answer needed,
and one that does not beat the lexicon found no more than vocabulary.
The `− random` and `− lexicon` columns are the paired per-case
difference from each control, with a 95% interval.

| Highlighter | Method | Cases | Comprehensiveness ↑ | − random | − lexicon | Sufficiency ↓ | − random | − lexicon | p(selected), full |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model, span head | `span_head` | 500 | 0.1473 | +0.1138 ± 0.0190 | +0.0132 ± 0.0099 | 0.0459 | -0.1200 ± 0.0242 | +0.0128 ± 0.0140 | 0.5656 |
| model, gradient x input | `gradient_x_input` | 500 | 0.2334 | +0.1999 ± 0.0154 | +0.0993 ± 0.0111 | -0.0588 | -0.2248 ± 0.0203 | -0.0920 ± 0.0156 | 0.5656 |
| model, integrated gradients | `integrated_gradients` | 500 | 0.2403 | +0.2069 ± 0.0150 | +0.1063 ± 0.0111 | -0.0738 | -0.2398 ± 0.0195 | -0.1069 ± 0.0153 | 0.5656 |
| rationale lexicon | `fitted word list` | 500 | 0.1341 | +0.1006 ± 0.0176 | — | 0.0332 | -0.1328 ± 0.0245 | — | 0.5656 |
| random | `random` | 500 | 0.0335 | — | -0.1006 ± 0.0176 | 0.1660 | — | +0.1328 ± 0.0245 | 0.5656 |

| Highlighter, comprehensiveness / sufficiency | 1% | 5% | 10% | 20% | 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, span head | 0.102 / 0.084 | 0.121 / 0.061 | 0.136 / 0.039 | 0.168 / 0.026 | 0.210 / 0.020 |
| model, gradient x input | 0.161 / 0.018 | 0.189 / -0.042 | 0.228 / -0.090 | 0.282 / -0.113 | 0.307 / -0.068 |
| model, integrated gradients | 0.155 / -0.000 | 0.185 / -0.058 | 0.230 / -0.104 | 0.287 / -0.116 | 0.345 / -0.091 |
| rationale lexicon | 0.101 / 0.057 | 0.118 / 0.040 | 0.130 / 0.028 | 0.151 / 0.024 | 0.170 / 0.017 |
| random | 0.013 / 0.205 | 0.015 / 0.201 | 0.023 / 0.181 | 0.034 / 0.158 | 0.082 / 0.084 |

Cost: 18,995 re-asks in 164.6 s (batches of 16), 252.0 s with attribution.
