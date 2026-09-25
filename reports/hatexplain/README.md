# HateXplain — evidence against human rationales, on the CPU spike

HateXplain (Mathew et al., AAAI 2021), Punyajoy Saha and co-authors. MIT
(repository LICENSE, Copyright (c) 2020 Punyajoy Saha); the authors' dataset
card states CC BY 4.0. https://github.com/punyajoy/HateXplain at `01d742279dac`.

**This is the pipeline proved on CPU, not a result about the architecture.**
The model is the reference spike — 128 wide, two layers, a 1,164-token BPE
that splits 64% of HateXplain's words — and two seeds per arm is a spread
measured, not a certification. What it establishes is that the evidence head
trains, serves and is scored end to end, and where the floors sit. What the
mechanism can do is the backbone run's question.

One Choice question — `normal`, `offensive`, `hatespeech` — over each post.
Label: the majority of three annotators, three-way splits dropped (19,229 posts
remain). Training target: all three annotators' labels as a distribution.
Rationale: the tokens at least half of the rationale-giving annotators
marked, only on posts whose majority is hateful or offensive. Split by a hash
of the post id, a quarter held out; the evaluation set is topped up to the
5,000-case floor from unseen training rows.

## Accuracy and calibration

| Run | Rationales in training | Accuracy | Marginal | ECE | Adaptive ECE | Gates |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sup-seed0` | yes, 7,100 | 0.5798 | 0.4078 | 0.0096 | 0.0223 | all blocking pass |
| `sup-seed1` | yes, 7,113 | 0.5798 | 0.4088 | 0.0210 | 0.0241 | all blocking pass |
| `unsup-seed0` | no (`--rationale-weight 0`) | 0.5664 | 0.4078 | 0.0313 | 0.0313 | all blocking pass |
| `unsup-seed1` | no (`--rationale-weight 0`) | 0.5702 | 0.4088 | 0.0209 | 0.0253 | all blocking pass |

Both supervised seeds sit above both unsupervised ones, by 0.010–0.013. Two
seeds a side is not enough to call that an effect of the rationale loss, and
it is not claimed as one; what it does show is that adding the loss did not
cost the answer anything measurable.

## Plausibility

Scored on the evaluation split's posts with a non-empty human rationale, on
words of the state (`trigon.evals.rationale`). Token F1 is per-post F1 over
highlighted words, averaged; IOU F1 counts a predicted run of words as found
when it overlaps a human run by at least half their union.

| Highlighter | Method | Token F1 | IOU F1 | Highlighted | Human |
| --- | --- | ---: | ---: | ---: | ---: |
| span head, trained on rationales | `span_head` | 0.4875 / 0.4954 | 0.3301 / 0.3308 | 0.339 / 0.374 | 0.339 / 0.337 |
| same weights, attributed | `gradient_x_input` | 0.3247 / 0.2988 | 0.2637 / 0.2458 | 0.129 / 0.118 | |
| never shown a rationale | `gradient_x_input` | 0.3167 / 0.3079 | 0.2587 / 0.2590 | 0.129 / 0.111 | |
| *lexical floor* | `lexical_overlap` | 0.0258 / 0.0254 | 0.0016 / 0.0016 | 0.027 | |
| ***rationale lexicon*** | fitted word list | **0.5728 / 0.5736** | **0.4544 / 0.4521** | 0.172 / 0.178 | |
| *every word* | all words | 0.4361 / 0.4341 | 0.2185 / 0.2170 | 1.000 | |

Seed 0 / seed 1 throughout; 2,960 and 2,956 rationales. The floors differ
between seeds only because each seed's evaluation set is topped up from a
different shuffle.

**The word list wins.** Every word highlighted in at least half its training
occurrences — 1,505 and 1,560 of them, on this corpus very nearly a slur
list — beats
the trained span head by 0.08 token F1 and 0.12 IOU F1 on both seeds. That is
the floor the brief warned of, and the spike does not clear it.

**Supervision is worth something, and the attribution is not.** The trained
head is +0.17–0.20 token F1 over gradient × input on the same weights. Gradient
× input is below highlighting every word on token F1, on every run — though it
is more precise than every word (0.53–0.55 against 0.34) and above it on IOU
F1. Training the head did not make the model's own attribution more plausible:
0.30–0.32 with the rationale loss, 0.31–0.32 without.

**The head's threshold is not a coincidence of the numbers.** At 0.5 it
highlights 34–37% of words against the annotators' 34%, which is what a head
fitted with a proper scoring rule and thresholded at 0.5 should do.

## Cost

300 held-out posts, one question each, the trained checkpoint, schema prefix
cache on, one thread on an idle `Intel(R) Xeon(R) Processor @ 2.10GHz`:

| Request | p50 ms | p99 ms |
| --- | ---: | ---: |
| no evidence | 2.68 | 4.11 |
| `span_head` | 3.36 | 6.31 |
| `gradient_x_input` | 5.16 | 9.06 |

## `trigon ask` with evidence

```bash
trigon ask request.json --backend torch --weights sup-seed0.pt \
    --temperatures reports/hatexplain/sup-seed0-temperatures.json
```

on `"honestly you are a stupid idiot and nobody at this school likes you"`,
asking the HateXplain question and an unrelated Noul, with
`"options": {"include_evidence": true}`:

```json
"label": {
  "selected": "normal",
  "probabilities": {"normal": 0.4694, "offensive": 0.4024, "hatespeech": 0.1282},
  "evidence": [
    {"start": 26, "end": 31, "text": "idiot", "score": 0.5002},
    {"start": 51, "end": 63, "text": "school likes", "score": 0.6662}
  ],
  "evidence_method": "span_head"
},
"school": {
  "probability": 0.3969,
  "evidence": [{"start": 51, "end": 63, "text": "school likes", "score": 0.6449}],
  "evidence_method": "span_head"
}
```

The spike answers `normal` for a post most readers would call offensive, and
its head reaches past `idiot` to `school likes`: this is what the mechanism
looks like at 128 wide, and why it is not a result.

## What needs a GPU

- **The backbone arm.** `scripts/modal_train.py launch --corpus hatexplain
  --extra "--backbone qwen2.5-1.5b"`, three or more seeds with rationales and
  as many without. It decides both falsifiers in `docs/decisions.md`
  (*Evidence is attribution until it is supervised*): the head has to beat the
  rationale lexicon on both metrics, and gradient × input has to beat every
  word on token F1.
- **Faithfulness**, which nothing here measures: delete the highlighted spans
  and measure how far the answer moves.

## Reproducing

```bash
python scripts/train_corpus.py hatexplain -n 0 --epochs 4 --calibration-n 1000 \
    --seed 0 --out reports/hatexplain/sup-seed0.md
python scripts/train_corpus.py hatexplain -n 0 --epochs 4 --calibration-n 1000 \
    --seed 0 --rationale-weight 0 --out reports/hatexplain/unsup-seed0.md
```

About twelve minutes a run on one core. The supervised runs were trained
twice, before and after the span head moved to the no-grad serving path, and
matched to the fourth decimal; seed 0's weights fingerprint `8278b654` both
times. The unsupervised runs predate that move and the rule that widens a subword
token to its word, and neither touches them: the first changes only the span
head's path, and the second never widens across whitespace, so it cannot
change which words a span touches — which is all plausibility reads.
