# HateXplain — evidence against human rationales

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

## On Qwen2.5-1.5B: the span head reads, and gradient × input does not

The backbone runs the spike could only set up: Qwen2.5-1.5B with LoRA rank 16,
lr 1e-4, 3 epochs, on the same splits (13,229 training posts, 5,000
evaluated). There are two arms of four seeds on `NVIDIA A10`: **with**
rationale supervision (`qwen15b-e3-lr1e-4-rationales-*`) and **without** it
(`--rationale-weight 0`, `qwen15b-e3-lr1e-4-norationales-*`). Commit `3a845d2`,
clean tree, about 41 minutes a seed.

**Plausibility against the human rationales, rationale arm, four seeds:**

| Highlighter | Token F1 | IOU F1 |
| --- | ---: | ---: |
| **`span_head`, trained on rationales** | **0.7148–0.7197** | **0.6143–0.6213** |
| `gradient_x_input`, same weights | 0.2871–0.3081 | 0.2114–0.2336 |
| `gradient_x_input`, never shown a rationale (other arm) | 0.1790–0.2993 | 0.1416–0.2251 |
| *rationale lexicon* (floor) | 0.5714–0.5736 | 0.4491–0.4544 |
| *every word* (floor) | 0.4341–0.4369 | 0.2170–0.2193 |

**The first falsifier in `docs/decisions.md` did not fire.** The span head
beats the rationale lexicon on both metrics, on every seed, by +0.14 token F1
and +0.16 IOU F1. On the spike it lost to the same lexicon by 0.08. Above the
word list, what the head adds is context: which occurrences of a word the
annotators marked, and the non-lexical spans a list cannot hold.

**The second falsifier fired.** Gradient × input does not beat highlighting
every word on token F1, on either arm or any seed. So as an unsupervised
attribution it is worse than trivial. Every checkpoint trained without
rationales served it by default, the deployed Banking77 model included, until
the change recorded below. The
decision names integrated gradients as the next candidate, scored the same
way; it is being built and will be scored on these same checkpoints without
retraining. Until then, `evidence_method: "gradient_x_input"` should be read
as unvalidated, and the response already says which method produced the
spans.

**Integrated gradients, scored on the same eight checkpoints**
(`qwen15b-e3-lr1e-4-{rationales,norationales}-igscore-*`: `--weights`, no
retraining, commit `5a19070`; the reloaded models reproduce their original
calibration decisions seed for seed):

| Highlighter | Rationale arm, token F1 | No-rationale arm, token F1 | Rationale arm, IOU F1 |
| --- | ---: | ---: | ---: |
| `integrated_gradients` | 0.3524–0.3839 | 0.1934–0.3849 | 0.2923–0.3214 |
| `gradient_x_input` | 0.2873–0.3075 | 0.1793–0.2998 | 0.2138–0.2358 |
| *every word* | 0.4341–0.4369 | 0.4341–0.4369 | 0.2170–0.2193 |

It beats gradient × input on seven of eight checkpoints, and still misses
*every word* on token F1 on all eight. **So neither unsupervised attribution
is served by default any more.** A checkpoint never trained on rationales
answers `include_evidence` with `evidence_method: "unavailable"`, and an
operator can opt into either method with `TRIGON_UNSUPERVISED_EVIDENCE`.

**Integrated gradients is not complete on the backbone.** The summed
attributions miss the log-probability difference they must equal: the median
error is 137–179% on the rationale arm and 78–895% on the other, against
under 1% in float32 on CPU. The backbone runs under bf16 autocast. This rules
out the implementation on this hardware, not the method. A float32 path is
the open item.

**Accuracy and calibration, both arms** (majority label, three classes,
marginal 0.4104; every blocking gate passes on all eight runs):

| Seed | Accuracy, rationales | Accuracy, none | Brier, rationales | Brier, none | ECE, rationales | ECE, none |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6970 | 0.6478 | 0.4102 | 0.4582 | 0.0271 | 0.0224 |
| 1 | 0.6854 | 0.5688 | 0.4232 | 0.5334 | 0.0272 | 0.0226 |
| 2 | 0.6938 | 0.6836 | 0.4119 | 0.4201 | 0.0277 | 0.0185 |
| 3 | 0.6930 | 0.6772 | 0.4086 | 0.4145 | 0.0237 | 0.0300 |
| median | **0.6934** | 0.6625 | **0.4111** | 0.4392 | 0.0272 | 0.0225 |

**Rationale supervision also made the classifier better and steadier.** The
median is +0.031 accuracy, Brier is better on all four seeds, and the
accuracy spread narrows from 0.115 (0.5688–0.6836) to 0.012. Four seeds a
side is enough to call the spread, and the medians differ by less than the
unsupervised arm's range, so the accuracy gain itself is suggestive rather
than established. ECE is slightly higher with rationales and inside the gate
on every seed. The floor's p95 is 0.017–0.022, so several of these ECEs are
within noise of perfect.

The CPU spike results below are unchanged. They are what the backbone was
measured against.

## On the backbone, the span head is faithful as well as plausible

Eval-only on the same eight checkpoints (`--weights`, no retraining), commit
`e061a18`, clean tree. Two sweeps per arm:
- `*-f32faith-*`: integrated gradients in float32 at 32 points on every
  rationale case, plus faithfulness on 500 cases.
- `*-igf32-s256-*`: integrated gradients at 256 points on 200 cases, to
  separate quadrature from arithmetic.

The first sweep ran on `NVIDIA A10` and reproduces the original
runs' calibration decisions seed for seed.

**Faithfulness, rationale arm, four seeds** (ERASER AOPC over 1/5/10/20/50% of
words, on the uncalibrated distribution, 500 cases a seed):

| Highlighter | Comprehensiveness ↑ | Sufficiency ↓ | Against the lexicon control |
| --- | ---: | ---: | --- |
| **`span_head`** | **0.331–0.376** (median 0.345) | **0.064–0.088** (0.079) | better on **both** metrics on **all four** seeds, 95% intervals clear of zero: comprehensiveness +0.015 to +0.023, sufficiency −0.009 to −0.017 |
| `integrated_gradients` | 0.284–0.344 (0.302) | 0.074–0.109 (0.093) | level with it: within noise, or worse, on every seed |
| `gradient_x_input` | 0.201–0.216 (0.206) | 0.180–0.216 (0.189) | worse on both metrics on every seed |
| *rationale lexicon* | 0.316–0.353 (0.328) | 0.081–0.097 (0.090) | (control) |
| *random* | 0.069–0.086 (0.073) | 0.311–0.341 (0.328) | (control) |

On the no-rationale arm there is no span head. Gradient × input
(comprehensiveness 0.103–0.202) and integrated gradients (0.124–0.296) both
fall below the lexicon on comprehensiveness on every seed.

**The spike's reversal does not survive the backbone.** On the 128-wide spike
the span head was the most plausible highlighter and the least faithful. On
Qwen2.5-1.5B it is the most plausible **and** the most faithful:
- deleting its top words moves the answer more than deleting the rationale
  lexicon's words;
- keeping only its words preserves the answer better.

So it is reading more than a vocabulary, and the answer depends on what it
highlights. `docs/decisions.md`'s falsifier *Plausibility is the wrong
target* did not fire.

**Integrated gradients in float32 still does not clear the bar.**

| Run | Completeness error, median | IG token F1 | *Every word* |
| --- | ---: | ---: | ---: |
| Rationale arm, 32 points, all cases | 1.07–1.55 | 0.360–0.393 | 0.434–0.437 |
| Rationale arm, 256 points, 200 cases | 0.65–1.27 | 0.418–0.445 | 0.431–0.456 |
| No-rationale arm, 32 points | 0.68–7.50 | 0.200–0.386 | 0.434–0.437 |
| No-rationale arm, 256 points | 0.57–8.26 | 0.238–0.438 | 0.431–0.456 |

Float32 and more points both help:
- the bf16 medians at 32 points were 1.37–1.79 and 0.78–8.95;
- 256 points roughly halves the error again.

Even so, completeness is nowhere near holding. Token F1 stays below *every
word* on fifteen of sixteen runs; one no-rationale seed reaches 0.438
against 0.431. What remains is the roughness of the straight path through a
pre-norm backbone, not arithmetic. So the unsupervised default stays `none`.

**The same checkpoint on a different GPU is a different run.** The 256-point
rationale sweep landed on `A10G`, where every other sweep had `A10`. Same
weights and code, and accuracy moved by up to 0.0008 (0.6974 against 0.6970,
0.6846 against 0.6854). The calibrator also chose differently on two seeds:
on seed 2, isotonic 0.0816 → 0.0523 where the A10 fit 0.0896 → 0.0434.
Nothing here depends on that, but it is a GPU-side measurement of the ledger's
open item that "certified on four seeds" means four seeds on one machine.

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
- **Faithfulness on the backbone.** It is built and proved on the spike
  (below), and the eight saved checkpoints are scored eval-only with
  `--weights`.

## Faithfulness, on the spike

Does the model use what it highlights? This uses ERASER's comprehensiveness
and sufficiency (`docs/evals.md`, section 7). The top 1–50% of words by each
method's own scores are deleted from the post, or kept alone, and the post
is asked again. The weights are `sup-seed0`, retrained at this commit and
reloaded eval-only with `--weights` (`spike-faithfulness-seed0.md`). They
reproduce that seed's accuracy, ECE and plausibility to the fourth decimal.
The latency columns in that report were measured on a shared, oversubscribed
machine and mean nothing. The table covers 500 of the
evaluation rationales, with differences paired per case ± a 95% interval:

| Highlighter | Comprehensiveness ↑ | − lexicon | Sufficiency ↓ | − lexicon |
| --- | ---: | ---: | ---: | ---: |
| `span_head` | 0.147 | +0.013 ± 0.010 | 0.046 | +0.013 ± 0.014 |
| `gradient_x_input` | 0.233 | +0.099 ± 0.011 | −0.059 | −0.092 ± 0.016 |
| `integrated_gradients` | 0.240 | +0.106 ± 0.011 | −0.074 | −0.107 ± 0.015 |
| *rationale lexicon* | 0.134 | | 0.033 | |
| *random* | 0.034 | | 0.166 | |

Every method beats the random control by at least +0.11 comprehensiveness
and −0.12 sufficiency. **On the spike, faithfulness reverses plausibility.**
The trained head's spans are about as faithful as the word list. They are
0.013 better on comprehensiveness, an interval that barely clears zero, and
no better on sufficiency. The gradient methods, which lose to highlighting
every word on plausibility, are the most faithful by about 0.1 on both
metrics. The head learned what annotators mark, and the model's answer
leans on other words as well. One seed of the spike does not decide
the decision's falsifier; the backbone checkpoints do.

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
