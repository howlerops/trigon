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

Model(s): trigon-qwen2.5-1.5b-0.1.0+f447b883

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.6966 | 0.0268 | 0.0276 | 0.4090 | 52.8 | 56.4 | 92 |
| hatexplain/calibrated | 5000 | 0.6772 | 0.0300 | 0.0251 | 0.4145 | 53.8 | 56.4 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0210 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0140 on this run, p95 0.0210
- PASS accuracy_over_baseline: 0.2668 (limit 0.0500) -- model 0.6772 vs marginal predictor 0.4104; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.2668 (limit 0.0500) -- worst is 'label' at 0.6772 vs its own marginal 0.4104; the pooled gate hides this
- PASS workhorse_ece: 0.0300 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0251 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0300 (limit 0.0500) -- worst is 'choice' at ECE 0.0300 (overconfidence -0.009); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.6772 | 0.4104 | +0.2668 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.677 | 0.668 | -0.009 | 0.0300 | 0.0251 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.677 | 0.668 | -0.009 | 0.0300 | 0.0251 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0152 (95th percentile 0.0230), simulated over 200 resamples. The measured ECE is 0.0268.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)    98  0.379  0.357  +0.022  ##############.|........................
  [0.40,0.47)   452  0.438  0.449  -0.011  ##################|.....................
  [0.47,0.53)   703  0.500  0.498  +0.002  ####################|...................
  [0.53,0.60)   632  0.567  0.595  -0.028  #######################|................
  [0.60,0.67)   571  0.635  0.676  -0.041  #########################|#.............
  [0.67,0.73)   659  0.700  0.731  -0.032  ############################|...........
  [0.73,0.80)   589  0.765  0.810  -0.045  ###############################|........
  [0.80,0.87)   599  0.833  0.871  -0.039  #################################|#.....
  [0.87,0.93)   459  0.896  0.924  -0.028  ####################################|...
  [0.93,1.00)   238  0.959  0.958  +0.001  ######################################|.
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
| model | `unavailable` | 2,948 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.340 |
| model, gradient x input | `gradient_x_input` | 2,948 | 0.3009 | 0.5012 | 0.3158 | 0.2267 | 0.150 | 0.340 |
| model, integrated gradients | `integrated_gradients` | 2,948 | 0.3861 | 0.6151 | 0.3960 | 0.3260 | 0.146 | 0.340 |
| lexical floor | `lexical_overlap` | 2,948 | 0.0254 | 0.1300 | 0.0152 | 0.0014 | 0.027 | 0.340 |
| rationale lexicon | `fitted word list` | 2,948 | 0.5729 | 0.7614 | 0.6084 | 0.4526 | 0.175 | 0.340 |
| every word | `all words` | 2,948 | 0.4369 | 0.3396 | 1.0000 | 0.2193 | 1.000 | 0.340 |

Integrated gradients' completeness on the first 200 of these cases (32 points, `u ** 3` spacing, float32 path): relative error of the summed attributions against the log-probability difference they must add up to, median 0.6812, p90 4.4590, max 30.8978, over the 200 whose difference is at least 0.01 nats (median difference 4.000). A large error means too few points or too little precision, and the `integrated gradients` row is then an artefact of the arithmetic, not the method.

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
| model, gradient x input | `gradient_x_input` | 500 | 0.2021 | +0.1280 ± 0.0196 | -0.1120 ± 0.0206 | 0.1898 | -0.1540 ± 0.0220 | +0.1005 ± 0.0212 | 0.6509 |
| model, integrated gradients | `integrated_gradients` | 500 | 0.2961 | +0.2220 ± 0.0211 | -0.0181 ± 0.0169 | 0.0850 | -0.2588 ± 0.0231 | -0.0043 ± 0.0163 | 0.6509 |
| rationale lexicon | `fitted word list` | 500 | 0.3141 | +0.2400 ± 0.0251 | — | 0.0893 | -0.2545 ± 0.0274 | — | 0.6509 |
| random | `random` | 500 | 0.0741 | — | -0.2400 ± 0.0251 | 0.3437 | — | +0.2545 ± 0.0274 | 0.6509 |

| Highlighter, comprehensiveness / sufficiency | 1% | 5% | 10% | 20% | 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| model, gradient x input | 0.116 / 0.315 | 0.145 / 0.259 | 0.187 / 0.190 | 0.242 / 0.125 | 0.321 / 0.060 |
| model, integrated gradients | 0.199 / 0.193 | 0.233 / 0.144 | 0.288 / 0.078 | 0.342 / 0.020 | 0.418 / -0.010 |
| rationale lexicon | 0.215 / 0.164 | 0.265 / 0.116 | 0.318 / 0.081 | 0.371 / 0.053 | 0.401 / 0.032 |
| random | 0.023 / 0.425 | 0.029 / 0.413 | 0.040 / 0.383 | 0.083 / 0.316 | 0.195 / 0.182 |

Cost: 16,233 re-asks in 120.3 s (batches of 16), 584.6 s with attribution.
