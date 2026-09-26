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
| Epochs | 3 |
| Seed | 0 |
| Device | cuda (NVIDIA A10) |
| Model | reference spike, d_model 128, 2 layers |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0003 --accumulate 8 --d-model 128 --layers 2 --seed 0 --device cuda --max-batch-cells 50000000
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+d1365c5a

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.6478 | 0.0224 | 0.0203 | 0.4582 | 49.3 | 54.2 | 92 |
| hatexplain/calibrated | 5000 | 0.6478 | 0.0224 | 0.0203 | 0.4582 | 49.0 | 52.7 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0221 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0153 on this run, p95 0.0221
- PASS accuracy_over_baseline: 0.2400 (limit 0.0500) -- model 0.6478 vs marginal predictor 0.4078; calibration cannot reject a model that ignores the state
- PASS (advisory) worst_question_over_baseline: 0.2400 (limit 0.0500) -- worst is 'label' at 0.6478 vs its own marginal 0.4078; the pooled gate hides this
- PASS workhorse_ece: 0.0224 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0203 (limit 0.0500)
- PASS (advisory) worst_primitive_workhorse_ece: 0.0224 (limit 0.0500) -- worst is 'choice' at ECE 0.0224 (overconfidence -0.006); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.6478 | 0.4078 | +0.2400 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.648 | 0.642 | -0.006 | 0.0224 | 0.0203 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.648 | 0.642 | -0.006 | 0.0224 | 0.0203 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0153 (95th percentile 0.0221), simulated over 200 resamples. The measured ECE is 0.0224.

The measured error is **above** that floor, so the miscalibration is real rather than sampling noise.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)   184  0.378  0.348  +0.030  ##############.|........................
  [0.40,0.47)   520  0.438  0.415  +0.022  #################.|.....................
  [0.47,0.53)   785  0.498  0.480  +0.018  ###################.|...................
  [0.53,0.60)   617  0.566  0.577  -0.011  #######################|................
  [0.60,0.67)   590  0.634  0.619  +0.015  #########################|..............
  [0.67,0.73)   667  0.700  0.738  -0.037  ############################|#..........
  [0.73,0.80)   768  0.767  0.781  -0.014  ###############################|........
  [0.80,0.87)   585  0.829  0.853  -0.024  #################################|......
  [0.87,0.93)   265  0.894  0.947  -0.053  ####################################|#..
  [0.93,1.00)    19  0.943  1.000  -0.057  ######################################|#
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
| model | `unavailable` | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.356 |
| model, gradient x input | `gradient_x_input` | 200 | 0.2619 | 0.5034 | 0.2517 | 0.2035 | 0.131 | 0.356 |
| model, integrated gradients | `integrated_gradients` | 200 | 0.3066 | 0.5236 | 0.3183 | 0.2462 | 0.132 | 0.356 |
| lexical floor | `lexical_overlap` | 200 | 0.0343 | 0.1533 | 0.0207 | 0.0014 | 0.028 | 0.356 |
| rationale lexicon | `fitted word list` | 200 | 0.5653 | 0.7896 | 0.5823 | 0.4302 | 0.167 | 0.356 |
| every word | `all words` | 200 | 0.4485 | 0.3560 | 1.0000 | 0.2417 | 1.000 | 0.356 |

Integrated gradients' completeness on the first 200 of these cases (256 points, `u ** 3` spacing, float32 path): relative error of the summed attributions against the log-probability difference they must add up to, median 2.4835, p90 11.5154, max 66.9118, over the 199 whose difference is at least 0.01 nats (median difference 0.713). A large error means too few points or too little precision, and the `integrated gradients` row is then an artefact of the arithmetic, not the method.
