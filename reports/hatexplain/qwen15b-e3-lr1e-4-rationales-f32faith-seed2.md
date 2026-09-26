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
| `label` | 0.4090 |

| Marginal predictor, pooled | Brier | NLL |
| --- | ---: | ---: |
| ignores its input | 0.6583 | 1.0865 |

A proper scoring rule, where argmax accuracy cannot separate a model
from the population: compare the model's Brier in the Suites table.

| | |
| --- | ---: |
| Epochs | 3 |
| Seed | 2 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+20c2f068

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.7008 | 0.0109 | 0.0170 | 0.4058 | 49.3 | 53.7 | 92 |
| hatexplain/calibrated | 5000 | 0.6938 | 0.0277 | 0.0376 | 0.4119 | 51.7 | 58.5 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0177 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0112 on this run, p95 0.0177
- PASS accuracy_over_baseline: 0.2848 (limit 0.0500) -- model 0.6938 vs marginal predictor 0.4090; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.2848 (limit 0.0500) -- worst is 'label' at 0.6938 vs its own marginal 0.4090; the pooled gate hides this
- PASS workhorse_ece: 0.0277 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0376 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0277 (limit 0.0500) -- worst is 'choice' at ECE 0.0277 (overconfidence +0.018); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.6938 | 0.4090 | +0.2848 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.694 | 0.712 | +0.018 | 0.0277 | 0.0376 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.694 | 0.712 | +0.018 | 0.0277 | 0.0376 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0147 (95th percentile 0.0205), simulated over 200 resamples. The measured ECE is 0.0109.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)    53  0.378  0.321  +0.057  #############..|........................
  [0.40,0.47)   282  0.441  0.429  +0.012  #################.|.....................
  [0.47,0.53)   625  0.499  0.502  -0.004  ####################|...................
  [0.53,0.60)   558  0.566  0.557  +0.009  ######################.|................
  [0.60,0.67)   545  0.633  0.611  +0.022  ########################.|..............
  [0.67,0.73)   591  0.700  0.685  +0.014  ###########################.|...........
  [0.73,0.80)   589  0.767  0.757  +0.010  ##############################.|........
  [0.80,0.87)   684  0.834  0.844  -0.009  #################################|......
  [0.87,0.93)   676  0.900  0.889  +0.011  ####################################|...
  [0.93,1.00)   397  0.958  0.955  +0.003  ######################################|.
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
| model | `span_head` | 2,954 | 0.7197 | 0.8271 | 0.7655 | 0.6213 | 0.300 | 0.339 |
| model, gradient x input | `gradient_x_input` | 2,954 | 0.3072 | 0.5304 | 0.3119 | 0.2286 | 0.149 | 0.339 |
| model, integrated gradients | `integrated_gradients` | 2,954 | 0.3873 | 0.6092 | 0.3899 | 0.3279 | 0.138 | 0.339 |
| lexical floor | `lexical_overlap` | 2,954 | 0.0259 | 0.1298 | 0.0155 | 0.0015 | 0.027 | 0.339 |
| rationale lexicon | `fitted word list` | 2,954 | 0.5714 | 0.7595 | 0.6081 | 0.4491 | 0.175 | 0.339 |
| every word | `all words` | 2,954 | 0.4367 | 0.3392 | 1.0000 | 0.2180 | 1.000 | 0.339 |

Integrated gradients' completeness on the first 200 of these cases (32 points, `u ** 3` spacing, float32 path): relative error of the summed attributions against the log-probability difference they must add up to, median 1.3410, p90 10.8864, max 298.5196, over the 200 whose difference is at least 0.01 nats (median difference 2.519). A large error means too few points or too little precision, and the `integrated gradients` row is then an artefact of the arithmetic, not the method.

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
| model, span head | `span_head` | 500 | 0.3760 | +0.2898 ± 0.0288 | +0.0233 ± 0.0085 | 0.0643 | -0.2653 ± 0.0289 | -0.0165 ± 0.0085 | 0.6833 |
| model, gradient x input | `gradient_x_input` | 500 | 0.2041 | +0.1179 ± 0.0230 | -0.1487 ± 0.0247 | 0.2159 | -0.1138 ± 0.0246 | +0.1350 ± 0.0253 | 0.6833 |
| model, integrated gradients | `integrated_gradients` | 500 | 0.3436 | +0.2574 ± 0.0240 | -0.0092 ± 0.0185 | 0.0744 | -0.2553 ± 0.0254 | -0.0065 ± 0.0173 | 0.6833 |
| rationale lexicon | `fitted word list` | 500 | 0.3528 | +0.2666 ± 0.0281 | — | 0.0809 | -0.2488 ± 0.0286 | — | 0.6833 |
| random | `random` | 500 | 0.0862 | — | -0.2666 ± 0.0281 | 0.3297 | — | +0.2488 ± 0.0286 | 0.6833 |

| Highlighter, comprehensiveness / sufficiency | 1% | 5% | 10% | 20% | 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, span head | 0.284 / 0.105 | 0.334 / 0.088 | 0.390 / 0.065 | 0.427 / 0.045 | 0.446 / 0.018 |
| model, gradient x input | 0.127 / 0.322 | 0.147 / 0.280 | 0.179 / 0.228 | 0.245 / 0.160 | 0.322 / 0.090 |
| model, integrated gradients | 0.244 / 0.182 | 0.285 / 0.126 | 0.336 / 0.066 | 0.398 / 0.015 | 0.455 / -0.017 |
| rationale lexicon | 0.262 / 0.136 | 0.314 / 0.098 | 0.362 / 0.084 | 0.399 / 0.057 | 0.427 / 0.029 |
| random | 0.034 / 0.400 | 0.048 / 0.377 | 0.060 / 0.363 | 0.092 / 0.312 | 0.197 / 0.196 |

Cost: 18,721 re-asks in 146.2 s (batches of 16), 637.3 s with attribution.
