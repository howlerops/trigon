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
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 3 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000 --rationale-weight 0.0
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
| hatexplain/uncalibrated | 5000 | 0.6966 | 0.0268 | 0.0276 | 0.4090 | 50.4 | 54.2 | 92 |
| hatexplain/calibrated | 5000 | 0.6772 | 0.0300 | 0.0251 | 0.4145 | 50.7 | 55.4 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0210 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0140 on this run, p95 0.0210
- PASS accuracy_over_baseline: 0.2668 (limit 0.0500) -- model 0.6772 vs marginal predictor 0.4104; calibration cannot reject a model that ignores the state
- PASS worst_question_over_baseline: 0.2668 (limit 0.0500) -- worst is 'label' at 0.6772 vs its own marginal 0.4104; the pooled gate hides this
- PASS workhorse_ece: 0.0300 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0251 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0300 (limit 0.0500) -- worst is 'choice' at ECE 0.0300 (overconfidence -0.009); pooled ECE cancels heads that err in opposite directions

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
vocabulary, not a reading.

| Highlighter | Method | Rationales | Token F1 | Precision | Recall | IOU F1 | Highlighted | Human |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model | `gradient_x_input` | 2,948 | 0.2993 | 0.4988 | 0.3149 | 0.2251 | 0.150 | 0.340 |
| lexical floor | `lexical_overlap` | 2,948 | 0.0254 | 0.1300 | 0.0152 | 0.0014 | 0.027 | 0.340 |
| rationale lexicon | `fitted word list` | 2,948 | 0.5729 | 0.7614 | 0.6084 | 0.4526 | 0.175 | 0.340 |
| every word | `all words` | 2,948 | 0.4369 | 0.3396 | 1.0000 | 0.2193 | 1.000 | 0.340 |
