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
| `label` | 0.4104 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6579 | 1.0858 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 3 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+19f21a1c

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.6930 | 0.0237 | 0.0209 | 0.4086 | 51.4 | 55.0 | 92 |
| hatexplain/calibrated | 5000 | 0.6930 | 0.0237 | 0.0209 | 0.4086 | 49.1 | 50.6 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0210 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0147 on this run, p95 0.0210
- PASS accuracy_over_baseline: 0.2826 (limit 0.0500) -- model 0.6930 vs marginal predictor 0.4104; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.2826 (limit 0.0500) -- worst is 'label' at 0.6930 vs its own marginal 0.4104; the pooled gate hides this
- PASS workhorse_ece: 0.0237 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0209 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0237 (limit 0.0500) -- worst is 'choice' at ECE 0.0237 (overconfidence -0.016); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.6930 | 0.4104 | +0.2826 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.693 | 0.677 | -0.016 | 0.0237 | 0.0209 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.693 | 0.677 | -0.016 | 0.0237 | 0.0209 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0147 (95th percentile 0.0210), simulated over 200 resamples. The measured ECE is 0.0237.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)    93  0.378  0.398  -0.020  ###############|........................
  [0.40,0.47)   403  0.438  0.392  +0.046  ################..|.....................
  [0.47,0.53)   700  0.501  0.513  -0.011  ####################|...................
  [0.53,0.60)   633  0.566  0.580  -0.014  #######################|................
  [0.60,0.67)   598  0.635  0.656  -0.020  #########################|..............
  [0.67,0.73)   598  0.700  0.699  +0.001  ############################|...........
  [0.73,0.80)   648  0.767  0.816  -0.049  ###############################|#.......
  [0.80,0.87)   574  0.832  0.878  -0.046  #################################|#.....
  [0.87,0.93)   464  0.898  0.918  -0.020  ####################################|...
  [0.93,1.00)   289  0.957  0.952  +0.005  ######################################|.
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
| model | `span_head` | 2,948 | 0.7148 | 0.7651 | 0.8151 | 0.6143 | 0.345 | 0.340 |
| model, gradient x input | `gradient_x_input` | 2,948 | 0.3058 | 0.5047 | 0.3163 | 0.2363 | 0.149 | 0.340 |
| model, integrated gradients | `integrated_gradients` | 2,948 | 0.3596 | 0.5581 | 0.3834 | 0.3001 | 0.155 | 0.340 |
| lexical floor | `lexical_overlap` | 2,948 | 0.0254 | 0.1300 | 0.0152 | 0.0014 | 0.027 | 0.340 |
| rationale lexicon | `fitted word list` | 2,948 | 0.5729 | 0.7614 | 0.6084 | 0.4526 | 0.175 | 0.340 |
| every word | `all words` | 2,948 | 0.4369 | 0.3396 | 1.0000 | 0.2193 | 1.000 | 0.340 |

Integrated gradients' completeness on the first 200 of these cases (32 points, `u ** 3` spacing, float32 path): relative error of the summed attributions against the log-probability difference they must add up to, median 1.5540, p90 12.8437, max 197.6727, over the 198 whose difference is at least 0.01 nats (median difference 1.276). A large error means too few points or too little precision, and the `integrated gradients` row is then an artefact of the arithmetic, not the method.

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
| model, span head | `span_head` | 500 | 0.3446 | +0.2754 ± 0.0274 | +0.0175 ± 0.0070 | 0.0755 | -0.2657 ± 0.0311 | -0.0127 ± 0.0076 | 0.6524 |
| model, gradient x input | `gradient_x_input` | 500 | 0.2005 | +0.1312 ± 0.0220 | -0.1267 ± 0.0221 | 0.1956 | -0.1455 ± 0.0237 | +0.1074 ± 0.0231 | 0.6524 |
| model, integrated gradients | `integrated_gradients` | 500 | 0.2836 | +0.2143 ± 0.0207 | -0.0436 ± 0.0198 | 0.1093 | -0.2318 ± 0.0226 | +0.0211 ± 0.0207 | 0.6524 |
| rationale lexicon | `fitted word list` | 500 | 0.3272 | +0.2579 ± 0.0270 | — | 0.0882 | -0.2529 ± 0.0298 | — | 0.6524 |
| random | `random` | 500 | 0.0693 | — | -0.2579 ± 0.0270 | 0.3411 | — | +0.2529 ± 0.0298 | 0.6524 |

| Highlighter, comprehensiveness / sufficiency | 1% | 5% | 10% | 20% | 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, span head | 0.240 / 0.137 | 0.297 / 0.110 | 0.352 / 0.072 | 0.399 / 0.045 | 0.436 / 0.014 |
| model, gradient x input | 0.117 / 0.309 | 0.141 / 0.268 | 0.184 / 0.201 | 0.232 / 0.140 | 0.328 / 0.060 |
| model, integrated gradients | 0.181 / 0.232 | 0.214 / 0.177 | 0.269 / 0.112 | 0.339 / 0.033 | 0.415 / -0.007 |
| rationale lexicon | 0.227 / 0.163 | 0.280 / 0.118 | 0.334 / 0.081 | 0.383 / 0.052 | 0.411 / 0.027 |
| random | 0.018 / 0.423 | 0.022 / 0.408 | 0.034 / 0.385 | 0.076 / 0.314 | 0.196 / 0.177 |

Cost: 18,950 re-asks in 142.1 s (batches of 16), 626.1 s with attribution.
