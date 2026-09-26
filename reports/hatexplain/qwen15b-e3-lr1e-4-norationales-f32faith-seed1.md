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
| `label` | 0.4088 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6581 | 1.0861 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 1 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 1 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+64e50267

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.5688 | 0.0226 | 0.0296 | 0.5334 | 50.7 | 55.2 | 92 |
| hatexplain/calibrated | 5000 | 0.5688 | 0.0226 | 0.0296 | 0.5334 | 50.4 | 55.4 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0220 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0149 on this run, p95 0.0220
- PASS accuracy_over_baseline: 0.1600 (limit 0.0500) -- model 0.5688 vs marginal predictor 0.4088; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.1600 (limit 0.0500) -- worst is 'label' at 0.5688 vs its own marginal 0.4088; the pooled gate hides this
- PASS workhorse_ece: 0.0226 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0296 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0226 (limit 0.0500) -- worst is 'choice' at ECE 0.0226 (overconfidence -0.013); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.5688 | 0.4088 | +0.1600 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.569 | 0.556 | -0.013 | 0.0226 | 0.0296 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.569 | 0.556 | -0.013 | 0.0226 | 0.0296 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0149 (95th percentile 0.0220), simulated over 200 resamples. The measured ECE is 0.0226.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)   733  0.374  0.393  -0.019  ###############|........................
  [0.40,0.47)  1246  0.433  0.425  +0.008  #################|......................
  [0.47,0.53)   975  0.496  0.524  -0.028  ####################|...................
  [0.53,0.60)   501  0.563  0.603  -0.039  #######################|................
  [0.60,0.67)   301  0.632  0.694  -0.062  #########################|##............
  [0.67,0.73)   236  0.700  0.665  +0.035  ###########################.|...........
  [0.73,0.80)   319  0.767  0.796  -0.029  ###############################|........
  [0.80,0.87)   358  0.835  0.821  +0.014  #################################|......
  [0.87,0.93)   284  0.899  0.898  +0.001  ####################################|...
  [0.93,1.00)    47  0.946  0.936  +0.010  #####################################.|.
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
| model | `unavailable` | 2,956 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.337 |
| model, gradient x input | `gradient_x_input` | 2,956 | 0.1797 | 0.3862 | 0.1655 | 0.1420 | 0.102 | 0.337 |
| model, integrated gradients | `integrated_gradients` | 2,956 | 0.2004 | 0.3754 | 0.1886 | 0.1611 | 0.111 | 0.337 |
| lexical floor | `lexical_overlap` | 2,956 | 0.0254 | 0.1292 | 0.0152 | 0.0016 | 0.027 | 0.337 |
| rationale lexicon | `fitted word list` | 2,956 | 0.5736 | 0.7537 | 0.6147 | 0.4521 | 0.178 | 0.337 |
| every word | `all words` | 2,956 | 0.4341 | 0.3369 | 1.0000 | 0.2170 | 1.000 | 0.337 |

Integrated gradients' completeness on the first 200 of these cases (32 points, `u ** 3` spacing, float32 path): relative error of the summed attributions against the log-probability difference they must add up to, median 7.5033, p90 79.1624, max 801.5799, over the 199 whose difference is at least 0.01 nats (median difference 0.595). A large error means too few points or too little precision, and the `integrated gradients` row is then an artefact of the arithmetic, not the method.

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
| model, gradient x input | `gradient_x_input` | 500 | 0.1034 | +0.0625 ± 0.0174 | -0.0675 ± 0.0172 | 0.1106 | -0.0421 ± 0.0163 | +0.0654 ± 0.0149 | 0.5694 |
| model, integrated gradients | `integrated_gradients` | 500 | 0.1236 | +0.0827 ± 0.0173 | -0.0473 ± 0.0150 | 0.0959 | -0.0568 ± 0.0163 | +0.0506 ± 0.0135 | 0.5694 |
| rationale lexicon | `fitted word list` | 500 | 0.1709 | +0.1300 ± 0.0199 | — | 0.0453 | -0.1075 ± 0.0175 | — | 0.5694 |
| random | `random` | 500 | 0.0409 | — | -0.1300 ± 0.0199 | 0.1527 | — | +0.1075 ± 0.0175 | 0.5694 |

| Highlighter, comprehensiveness / sufficiency | 1% | 5% | 10% | 20% | 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, gradient x input | 0.071 / 0.155 | 0.089 / 0.135 | 0.103 / 0.119 | 0.119 / 0.087 | 0.136 / 0.058 |
| model, integrated gradients | 0.096 / 0.143 | 0.104 / 0.127 | 0.119 / 0.100 | 0.138 / 0.073 | 0.161 / 0.037 |
| rationale lexicon | 0.131 / 0.088 | 0.157 / 0.059 | 0.178 / 0.040 | 0.193 / 0.018 | 0.196 / 0.021 |
| random | 0.018 / 0.180 | 0.017 / 0.176 | 0.026 / 0.169 | 0.046 / 0.146 | 0.098 / 0.092 |

Cost: 16,155 re-asks in 117.4 s (batches of 16), 575.4 s with attribution.
