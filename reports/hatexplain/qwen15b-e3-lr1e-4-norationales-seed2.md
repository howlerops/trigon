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
| Model | qwen2.5-1.5b, LoRA rank 16 |

```
python scripts/train_corpus.py hatexplain -n 0 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed 2 --device cuda --backbone qwen2.5-1.5b --lora-rank 16 --max-batch-cells 50000000 --rationale-weight 0.0
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

Model(s): trigon-qwen2.5-1.5b-0.1.0+f3c461fe

## Suites

| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain/uncalibrated | 5000 | 0.6836 | 0.0185 | 0.0193 | 0.4201 | 54.7 | 85.9 | 92 |
| hatexplain/calibrated | 5000 | 0.6836 | 0.0185 | 0.0193 | 0.4201 | 54.6 | 91.8 | 92 |

## Release gates

- PASS sample_size: 5000.0000 (limit 5000.0000) -- below this, ECE is dominated by estimator noise
- PASS gate_is_testable: 0.0213 (limit 0.0250) -- a perfectly calibrated model scores ECE 0.0142 on this run, p95 0.0213
- PASS accuracy_over_baseline: 0.2746 (limit 0.0500) -- model 0.6836 vs marginal predictor 0.4090; calibration cannot reject a model that ignores the state
- PASS worst_question_over_baseline: 0.2746 (limit 0.0500) -- worst is 'label' at 0.6836 vs its own marginal 0.4090; the pooled gate hides this
- PASS workhorse_ece: 0.0185 (limit 0.0500)
- PASS workhorse_adaptive_ece: 0.0193 (limit 0.0500)
- PASS worst_primitive_workhorse_ece: 0.0185 (limit 0.0500) -- worst is 'choice' at ECE 0.0185 (overconfidence +0.000); pooled ECE cancels heads that err in opposite directions

All blocking gates passed.

## Accuracy per question

The pooled lift above averages over questions. A model that has learned one question and answers the rest by rote clears a pooled gate, so the breakdown is printed whether or not it is gated on.

Measured on `hatexplain/calibrated`, the same run the gates read. Calibration can move a decision -- an isotonic map is monotone per class and not jointly -- so this can differ from the uncalibrated accuracy in the suites table above.

| Question | n | Accuracy | Its marginal predictor | Lift |
| --- | ---: | ---: | ---: | ---: |
| `label` | 5,000 | 0.6836 | 0.4090 | +0.2746 |

## Per-domain calibration

| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hatexplain | 5000 | 0.684 | 0.684 | +0.000 | 0.0185 | 0.0193 |

## Per-primitive calibration

| Primitive | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| choice | 5000 | 0.684 | 0.684 | +0.000 | 0.0185 | 0.0193 |

## Is this number evidence?

On these 5000 predictions a perfectly calibrated model scores a mean ECE of 0.0142 (95th percentile 0.0213), simulated over 200 resamples. The measured ECE is 0.0185.

The measured error is **within** that floor, so this model is statistically indistinguishable from perfectly calibrated at this sample size. That is the strongest claim the data supports — it is not the same as proving the error is zero.

## Reliability diagram

```
  bin            n    conf     acc     gap
  ----------------------------------------------------------------------
  [0.33,0.40)    64  0.383  0.406  -0.023  ###############|........................
  [0.40,0.47)   412  0.439  0.439  -0.001  ##################|.....................
  [0.47,0.53)   647  0.499  0.454  +0.045  ##################..|...................
  [0.53,0.60)   575  0.565  0.579  -0.014  #######################|................
  [0.60,0.67)   583  0.634  0.605  +0.028  ########################.|..............
  [0.67,0.73)   583  0.700  0.698  +0.002  ############################|...........
  [0.73,0.80)   690  0.768  0.778  -0.010  ###############################|........
  [0.80,0.87)   715  0.834  0.857  -0.023  #################################|......
  [0.87,0.93)   600  0.898  0.917  -0.019  ####################################|...
  [0.93,1.00)   131  0.951  0.947  +0.005  ######################################|.
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
| model | `gradient_x_input` | 2,954 | 0.2755 | 0.4792 | 0.2801 | 0.1995 | 0.153 | 0.339 |
| lexical floor | `lexical_overlap` | 2,954 | 0.0259 | 0.1298 | 0.0155 | 0.0015 | 0.027 | 0.339 |
| rationale lexicon | `fitted word list` | 2,954 | 0.5714 | 0.7595 | 0.6081 | 0.4491 | 0.175 | 0.339 |
| every word | `all words` | 2,954 | 0.4367 | 0.3392 | 1.0000 | 0.2180 | 1.000 | 0.339 |
