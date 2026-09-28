# The adversarial + paired stream on Banking77: robustness bought, not certified

Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0.
https://github.com/PolyAI-LDN/task-specific-datasets

The fifth stream of `docs/data.md`, built from our own labelled cases
(`trigon.evals.paired`) and measured on the certified Banking77 recipe:
Qwen2.5-1.5B, LoRA rank 16, lr 1e-4, 4 epochs, four seeds, against the four
certified checkpoints of that recipe re-evaluated at the same commit.

**Verdict.** The stream makes the model robust to what it was built for, by
a wide margin, on every seed that learned: an injection naming a wrong
intent moves the certified model's accuracy from 0.90 to 0.48 and the
treated model's from 0.89 to 0.88; sixteen lines of irrelevant state move
it from 0.90 to 0.06 and 0.89 to 0.80. **It does not certify.** Every seed
clears every blocking gate, but one of four finishes at 51% accuracy, and a
spread from 0.51 to 0.91 is not one a caller can be handed. It stays off by
default. The negation half bought coherence without an answer: the Noul
head reads 50% on its own questions.

**Without negation, at weight 0, it certifies on four seeds of four**
(below, *Without negation*): accuracy 0.8859 (0.8692–0.9196), every
blocking gate on every seed, injection accuracy drop at most 0.035, and
padding at 16 lines 0.7825 against the baseline's 0.064. It does **not**
confirm the hypothesis that set it up. The raw Choice head is exactly as
overconfident as before (uncalibrated ECE median 0.0517 against 0.0519).
What changed on the seed that failed is that a calibrator was accepted
this time.

**The ablation (consistency weight 0) moves the failure, it does not remove
it.** Without the term the slow seed learns (0.862 against 0.511), which
puts the stall on the consistency term, and robustness holds at about the
same level. But seed 2 then fails `workhorse_ece` at 0.0516, overconfident by
0.052, so that arm certifies on three seeds of four as well. **Negation is
now read, not inferred:** on all eight treated checkpoints P(yes) is about
0.50 whether the named intent is true or false (separation between −0.0012
and +0.0024). The head never learned the question; the term only made its
two halves agree.

## What was built

* **A generator** (`trigon.evals.paired`, stdlib, deterministic by seed).
  From any case with a text or JSON-record state it derives an
  **injection** (an instruction naming a wrong answer, label held fixed),
  **padding** (1-6 irrelevant lines, some naming other options, label held
  fixed), a **paraphrase** of the question (label held fixed), and a
  **negation pair** (is option X right? / is it wrong?, labels derived: the
  affirm is yes exactly when X is the truth, which it is half the time).
  Each variant carries its anchor request and a pair id. Tested on
  Banking77's shape and on the verifiable synthetic generator
  (`tests/test_paired.py`).
* **Template pools split train / eval**, sharing no wording, so the
  benchmarks below score phrasings no training case used.
* **An optional consistency term** (`TrainingConfig.consistency_weight`,
  `--consistency-weight`): symmetric KL between variant and anchor where the
  label is held fixed, `(P(yes) + P(yes on the complement) - 1)^2` on
  negation pairs, gradients through both sides. The anchor is forwarded in
  the same step only when the weight is above 0.
* **The stream is derived after the validation slice is taken**
  (`TrainingConfig.augment`), so no variant of a held-out case trains and
  epoch selection reads the same data it reads without the stream.
* **Off by default, and off is bit-identical.** A tiny spike run hashed its
  weights before the change and after it, SHA-256 `28b228df...` both times;
  the test suite pins that a paired case trained at weight 0 is identical to
  its plain twin.
* **Paired benchmarks** (`PairedBenchmark`, `--robustness-n`): the same four
  kinds over the corpus's own held-out evaluation split and the evaluation
  template pool, scored with the jaggedness suite's definitions -- flip rate
  and total-variation drift as `InjectionSteeringBenchmark`, `rot` as
  `ContextRotBenchmark`, incoherence as `NegationCoherenceBenchmark`.

## The floor first

`scripts/paired_floor.py --n 1000`, the lexical floor on 1,000 bases each
(`floor.json`). Banking77's test split; the synthetic generator at seed
10,000.

| Benchmark | Metric | Banking77 | Synthetic |
| --- | --- | ---: | ---: |
| injection | accuracy, anchor -> variant | 0.452 -> 0.077 | 0.498 -> 0.418 |
| injection | flip rate | 0.900 | 0.141 |
| padding | accuracy at 16 lines | 0.017 | 0.216 |
| padding | rot (anchor - 16 lines) | 0.435 | 0.282 |
| paraphrase | flip rate | **0.000** | **0.000** |
| negation | accuracy | 0.505 | 0.501 |
| negation | mean incoherence | 0.138 | 0.257 |

What that says each benchmark measures:

* **Injection and padding are real tests.** A keyword matcher is steered by
  an injected intent name on 90% of Banking77 cases, and padding that names
  other intents takes it to 2%.
* **Paraphrase is aced by the floor, for free.** The floor scores a Choice by
  its options against the state and never reads the question, so no
  rewording can move it. A paraphrase flip rate is evidence only beside
  accuracy -- and the certified model's is already 0.014.
* **Negation accuracy sits at chance for the floor, by construction** (the
  named option is the truth half the time). Incoherence alone is not a
  test: any model that answers 0.5 to both halves scores 0.

## Four seeds each

Baseline: the certified checkpoints of `qwen15b-e4-lr1e-4`
(`reports/banking77/README.md`), re-gated and benchmarked with `--weights`
at this commit; they reproduce the `regate080` rows to four decimals.
Treatment: the same recipe plus `--paired-mix 0.5 --consistency-weight 1.0`
-- 3,188 pairs over the 6,375 training cases left after validation, 10,360
cases an epoch. Same splits, same evaluation set of 5,000, same gates.

### The gates

| Seed | Baseline accuracy | Baseline ECE (floor p95) | Treated accuracy | Treated lift | Treated ECE | Treated adaptive ECE | Floor p95 | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.9054 | 0.0202 (0.0116) | 0.9052 | +0.8892 | 0.0145 | 0.0134 | 0.0126 | PASS |
| 1 | 0.8980 | 0.0448 (0.0102) | **0.5110** | +0.4944 | 0.0161 | 0.0239 | 0.0192 | PASS |
| 2 | 0.9038 | 0.0216 (0.0115) | 0.8720 | +0.8546 | 0.0370 | 0.0354 | 0.0126 | PASS |
| 3 | 0.8502 | 0.0105 (0.0139) | 0.8838 | +0.8676 | 0.0220 | 0.0246 | 0.0158 | PASS |

| | Baseline | Treated |
| --- | --- | --- |
| Accuracy, median (range) | 0.9009 (0.8502-0.9054) | **0.8779 (0.5110-0.9052)** |
| Lift, median (range) | +0.8839 (+0.8340-+0.8894) | +0.8611 (+0.4944-+0.8892) |
| ECE, median (range) | 0.0209 (0.0105-0.0448) | 0.0191 (0.0145-0.0370) |
| Adaptive ECE, median (range) | 0.0164 (0.0106-0.0445) | 0.0243 (0.0134-0.0354) |
| Blocking gates | 4 / 4 | 4 / 4 |

Every ECE carries its simulated floor. Baseline seed 3 and treated seed 1
are below their floor's p95 -- indistinguishable from perfect calibration
at 5,000 cases -- and the rest are measurements above it.

**Seed 1 did not collapse; it never finished learning.** Its validation
loss went 3.50, 2.94, 2.50, 2.41 over four epochs where the other three
reached 0.48-0.60 in one, and best-epoch selection kept epoch 4. The same
seed was the slowest starter in the baseline too (validation 0.85 after
epoch 1 against 0.52-0.83), so the stream turned a slow start into an
unfinished run. `accuracy_over_baseline` passes it, correctly -- 51% is
0.49 over a 1.6% marginal -- which is the gate doing what it says and why
certification also reads the spread.

**On the three seeds that learned, accuracy cost 0-3 points**: 0.9052,
0.8720, 0.8838, against a baseline whose own three best are 0.9054,
0.9038, 0.8980.

### The robustness benchmarks

1,000 evaluation cases per benchmark per seed, from templates training
never saw. Median (range) over four seeds.

| Benchmark | Metric | Lexical floor | Baseline | Treated |
| --- | --- | ---: | ---: | ---: |
| injection | accuracy on the anchor | 0.452 | 0.9005 (0.851-0.908) | 0.8860 (0.495-0.901) |
| injection | accuracy on the variant | 0.077 | 0.4760 (0.472-0.491) | **0.8800 (0.469-0.894)** |
| injection | flip rate | 0.900 | 0.5010 (0.494-0.516) | **0.0535 (0.041-0.279)** |
| injection | mean drift | 0.074 | 0.5190 (0.517-0.539) | 0.0583 (0.052-0.193) |
| padding | accuracy at 4 lines | 0.061 | 0.2730 (0.227-0.281) | **0.8660 (0.351-0.873)** |
| padding | accuracy at 16 lines | 0.017 | 0.0635 (0.049-0.073) | **0.8045 (0.184-0.845)** |
| padding | flip rate at 16 lines | 0.965 | 0.9340 (0.921-0.948) | 0.1530 (0.103-0.741) |
| padding | rot | 0.435 | 0.8395 (0.783-0.849) | **0.0880 (0.043-0.311)** |
| paraphrase | accuracy on the variant | 0.452 | 0.9000 (0.847-0.906) | 0.8880 (0.498-0.905) |
| paraphrase | flip rate | 0.000 | 0.0140 (0.012-0.018) | 0.0095 (0.003-0.101) |
| negation | accuracy | 0.505 | 0.4997 (0.498-0.502) | **0.4980 (0.498-0.500)** |
| negation | mean incoherence | 0.138 | 0.4582 (0.395-0.658) | 0.0348 (0.011-0.298) |
| negation | coherent rate (<= 0.05) | 0.221 | 0.0575 (0.023-0.070) | 0.7750 (0.001-1.000) |

**Injection and padding moved, and the move is not bought by ignoring the
input**: accuracy on the variants rose with them, to within a point of the
anchors on the three seeds that learned. The certified model is badly
steerable -- half its answers flip under an injection naming another
intent, and sixteen lines of distractors take it below 7% -- which nothing
in the gates could see, since every gate reads clean state.

**Paraphrase did not need fixing.** The certified model already agreed with
itself on 98.6% of reworded questions; the floor's 100% shows the metric
cannot say more than that.

**Negation is a null result dressed as a win.** Incoherence fell from 0.46
to 0.03, and accuracy on the same questions stayed at 0.498 -- chance, the
floor's number. A head that answers near 0.5 to every affirm and every deny
is perfectly coherent and knows nothing; that is almost certainly what the
Noul head learned, and it is exactly what the consistency term rewards. The
per-answer probabilities are not in the reports, so "near 0.5" is inferred
from accuracy and incoherence together, not read.

**The generic jaggedness suite is uninformative here** and is in the
per-seed reports only for completeness: on its four-option support schema,
which Banking77 never trained, both arms score chance (clean routing
accuracy 0.25), so its flip rates and its negation coherence -- 0.38 to
0.04 -- describe a model guessing, not reading.

## The ablation: consistency weight 0

Same recipe, same seeds, same splits; `--paired-mix 0.5
--consistency-weight 0`, so paired cases train as plain augmented data. At
commit `3d48197`, which also makes the negation benchmark record P(yes) on
each half, split by whether the named intent is true. The weight-1
checkpoints were re-evaluated at that commit (`cw1-reread`) to get the same
readings; their accuracies reproduce the table above.

| Seed | Weight 1: accuracy | Weight 0: accuracy | Weight 0: ECE | Weight 0: verdict |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.9052 | 0.9058 | 0.0195 | PASS |
| 1 | **0.5110** | **0.8618** | 0.0170 | PASS |
| 2 | 0.8720 | 0.9074 | **0.0516** | **FAIL** `workhorse_ece`, adaptive, worst primitive |
| 3 | 0.8838 | 0.8876 | 0.0400 | PASS |

| Median (range) | Weight 1 | Weight 0 |
| --- | ---: | ---: |
| Accuracy | 0.8779 (0.5110–0.9052) | 0.8967 (0.8618–0.9074) |
| Injection: accuracy on the variant | 0.8800 (0.469–0.894) | 0.8870 (0.819–0.906) |
| Injection: flip rate | 0.0535 (0.041–0.279) | 0.0655 (0.046–0.102) |
| Padding: accuracy at 16 lines | 0.8045 (0.184–0.845) | 0.8520 (0.515–0.893) |
| Negation: separation | −0.0003 (−0.0004–+0.0010) | +0.0010 (−0.0012–+0.0024) |
| Blocking gates passed | 4 / 4 | 3 / 4 |

**The term is what stalled seed 1.** Removed, the same seed on the same
split reaches 0.862. One seed is the whole of that evidence, but it is the
seed the term was suspected of, and the other three barely move.

**Removing it does not certify the stream.** Seed 2 lands 0.0016 over the
0.05 ECE gate, overconfident, and no calibrator rescued it; seed 1's padding
accuracy at 16 lines is 0.515, so its robustness is partial too. Four seeds
passing needs something else: a smaller mix, or a weight between 0 and 1.

**Negation, read directly.** Mean P(yes) on the affirm is 0.489–0.511 and on
the deny 0.485–0.513, with a standard deviation of 0.006–0.036 across cases,
and it does not depend on whether the claim is true. That holds at weight 0,
so the term did not cause it: the Noul head was never trained well enough to
read a claim that embeds one of 77 intents. Coherence at weight 1 is the
term making two uninformed answers agree.

## Without negation: certified on four seeds

> **Certified adapters.** On the `trigon-runs` Volume under
> **`banking77-paired-qwen15b-mix05-nonneg-cw0-019425daee66-20260928T172328`**
> are `seed0.pt`, `seed1.pt`, `seed2.pt` and `seed3.pt`. Model versions are
> `+7f72ab98`, `+4e0792e7`, `+7a385484` and `+6f1141f0`. Seed 0 has the best
> accuracy and the best calibration of the four (0.9196, ECE 0.0156) and the
> smallest injection drop.

Same recipe, same seeds, same splits. `--paired-mix 0.5 --paired-kinds
injection,padding,paraphrase --consistency-weight 0`, at commit `019425d`
(clean) on `NVIDIA A10G`. That gives 9,563 cases an epoch. Negation pairs
contribute two cases each, and there are none here, so the epoch has fewer
cases than the 10,360 above even though the three remaining kinds share the
mix.

### The gates

| Seed | Accuracy | Lift | Uncalibrated ECE | Calibrator | ECE | Adaptive ECE | Floor mean (p95) | Overconfidence | Injection drop | Kept epoch | Verdict |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.9196 | +0.9036 | 0.0573 | temperature T=2.45 | 0.0156 | 0.0125 | 0.0090 (0.0122) | −0.006 | −0.008 | 2 | PASS |
| 1 | 0.8762 | +0.8596 | 0.0477 | isotonic | 0.0175 | 0.0235 | 0.0074 (0.0118) | +0.007 | 0.035 | 1 | PASS |
| 2 | 0.8956 | +0.8782 | 0.0557 | isotonic | 0.0318 | 0.0337 | 0.0073 (0.0102) | +0.022 | 0.020 | 2 | PASS |
| 3 | 0.8692 | +0.8530 | 0.0325 | isotonic | 0.0224 | 0.0231 | 0.0089 (0.0123) | +0.022 | 0.026 | 1 | PASS |

| Median (range) | Baseline | Weight 0, with negation | **Weight 0, without negation** |
| --- | ---: | ---: | ---: |
| Accuracy | 0.9009 (0.8502–0.9054) | 0.8967 (0.8618–0.9074) | **0.8859 (0.8692–0.9196)** |
| Lift | +0.8839 (+0.8340–+0.8894) | — | +0.8689 (+0.8530–+0.9036) |
| ECE | 0.0209 (0.0105–0.0448) | 0.0298 (0.0170–0.0516) | 0.0200 (0.0156–0.0318) |
| Adaptive ECE | 0.0164 (0.0106–0.0445) | — | 0.0233 (0.0125–0.0337) |
| Uncalibrated ECE | — | 0.0519 (0.0400–0.0566) | 0.0517 (0.0325–0.0573) |
| Blocking gates passed | 4 / 4 | 3 / 4 | **4 / 4** |
| Injection drop ≤ 0.10 | — | 4 / 4 (max 0.052) | **4 / 4 (max 0.035)** |

Every ECE is above its simulated floor's p95, so each one measures real
miscalibration. None is indistinguishable from perfect, and all are well
under the 0.05 gate. The injection drop is `accuracy_anchor −
accuracy_variant` on `paired_injection`, which `injection_robustness`
bounds at 0.10 (`limits.MAX_INJECTION_ACCURACY_DROP`). This worktree
predates that gate, so the reports do not print it and it is computed here
from the numbers they do print.

**Accuracy costs a point and a half at the median.** The medians are 0.8859
here and 0.9009 for the baseline, and the ranges overlap. Part of the cost
comes from calibration, not training: the isotonic map is monotone per class
and not jointly, so it moved seed 2's accuracy from 0.9094 to 0.8956 and
seed 3's from 0.8838 to 0.8692. Seeds 1 and 3 kept epoch 1, because
validation loss rose after it. Seed 3 went from 0.65 to 0.98 at epoch 3.

### Why it certified: the calibrator, not the missing negation

The hypothesis was that negation's Noul cases cost the Choice head its
calibration. Measurement does not support it:

* **The raw head is equally overconfident either way.** Uncalibrated
  Choice ECE is 0.0521, 0.0566, 0.0516 and 0.0400 with negation, and
  0.0573, 0.0477, 0.0557 and 0.0325 without it. The medians are 0.0519
  and 0.0517.
* **What differs is the calibrator decision.** The run decides on a slice
  of the calibration split that neither candidate was fitted on. With
  negation, that slice read seed 2's raw head at 0.0381 against a best fit
  of 0.0341. That is not a demonstrable improvement, so the calibrator was
  declined and the evaluation set then read 0.0516. Seed 3 was declined the
  same way, 0.0512 against 0.0325, and passed at 0.0400 anyway. Without
  negation, the slice read 0.0510 to 0.0730 on every seed, a calibrator was
  accepted on all four, and it brought each one under the gate.

The configuration therefore certifies under the gates as they stand, but its
margin rests on the decline rule firing. On seed 2 that happened this time
and not last time, with a raw head just as miscalibrated both times. A
fifth seed whose scoring slice happens to read low would be declined and
land about 0.05, as seed 2 did. Removing negation cost nothing measurable:
negation accuracy was 0.499 before and after, which is chance. But removing
it is not what fixed calibration.

### The robustness benchmarks

1,000 evaluation cases per benchmark per seed, from templates training never
saw.

| Benchmark | Metric | Baseline | Weight 0, with negation | **Weight 0, without negation** | Per seed (0, 1, 2, 3) |
| --- | --- | ---: | ---: | ---: | --- |
| injection | accuracy on the anchor | 0.9005 | 0.8940 | 0.8885 (0.866–0.917) | 0.917, 0.888, 0.889, 0.866 |
| injection | accuracy on the variant | **0.4760** | 0.8870 | **0.8610 (0.840–0.925)** | 0.925, 0.853, 0.869, 0.840 |
| injection | accuracy drop | 0.424 | 0.0070 | 0.0230 (−0.008–0.035) | −0.008, 0.035, 0.020, 0.026 |
| injection | flip rate | 0.5010 | 0.0655 | 0.0755 (0.023–0.107) | 0.023, 0.079, 0.072, 0.107 |
| padding | accuracy at 4 lines | 0.2730 | 0.8725 | 0.8465 (0.775–0.913) | 0.913, 0.838, 0.855, 0.775 |
| padding | accuracy at 16 lines | **0.0635** | 0.8520 | **0.7825 (0.581–0.911)** | 0.911, 0.754, 0.811, 0.581 |
| padding | rot | 0.8395 | 0.0420 | 0.1060 (0.006–0.285) | 0.006, 0.134, 0.078, 0.285 |
| paraphrase | flip rate | 0.0140 | 0.0085 | 0.0060 (0.001–0.016) | 0.001, 0.003, 0.016, 0.009 |
| negation | accuracy | 0.4997 | 0.4958 | 0.4988 (0.497–0.501) | chance on every seed |
| negation | mean incoherence | 0.4582 | 0.0508 | 0.4876 (0.404–0.548) | back to the baseline's |

**The robustness gains hold.** Injection accuracy on the variant is 0.861
against the baseline's 0.476. Padding accuracy at 16 lines is 0.783 against
0.064, and on the worst seed it is still nine times the baseline. Padding is
weaker than it was with negation (median 0.852), even though padding now
gets a third of the mix instead of a quarter. The two lowest seeds, 1 and 3,
are the two that kept epoch 1, so early-epoch selection is a likely cause.
That is inferred from two seeds, not measured.

**Negation coherence is back to the baseline's**, 0.49, as it should be:
nothing trained it. Its accuracy is chance in all three arms.

### Cost

| Run | GPU time | Estimated cost |
| --- | ---: | ---: |
| Four seeds, about 108 min each (80 training, 28 evaluating) | 7.21 h | $7.9 |

That is 25,939 s of `elapsed_s` at about $1.10 an hour on A10G, not
counting start-up. Training is 80 minutes per seed against 95 with
negation, because there are fewer cases and no anchor forwards.

### Reproduce

```bash
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1,2,3 -n 0 --epochs 4 \
    --prefix paired-qwen15b-mix05-nonneg-cw0 --extra "--backbone qwen2.5-1.5b --lr 0.0001 \
    --save-model model.pt --paired-mix 0.5 --paired-kinds injection,padding,paraphrase \
    --consistency-weight 0 --robustness-n 1000 --jaggedness-n 40"
python scripts/modal_train.py collect \
    banking77-paired-qwen15b-mix05-nonneg-cw0-019425daee66-20260928T172328 --out-dir reports/paired
```

## Believed, not measured

* **That this configuration certifies on a fifth seed.** Its ECE margin
  depends on the calibrator being accepted, and above that decision
  flipped between two runs with equally miscalibrated raw heads. A seed
  sweep wider than four, or a raw head that is calibrated without help,
  would settle it.
* ~~That negation's Noul cases cost the Choice head its calibration.~~
  **Not supported**: uncalibrated Choice ECE has a median of 0.0519 with
  negation and 0.0517 without (above).
* **That early epoch selection is what weakened padding on seeds 1 and 3.**
  Two seeds, correlated, not tested.
* ~~That the consistency term is what slowed seed 1.~~ **Measured**: at
  weight 0 the same seed reaches 0.862 (above).
* **That negation failed for lack of signal.** 1,594 negation cases over four
  epochs may be too few for a question that embeds one of 77 intents in its
  own text. The coherence term is ruled out: weight 0 reads 0.5 too.
* **That a smaller mix or weight certifies on four seeds.** Untried.
* **That the robustness transfers beyond these templates.** Train and eval
  pools share no wording but share a style: bracketed system notes, office
  hours, "unrelated earlier ticket". A distribution of real injections is
  not this one.
* **That the stream would help the synthetic corpus.** The generator wraps
  it and the floor has been run on it; no model has been trained on it.

## Reproduce

```bash
python scripts/paired_floor.py --n 1000 --out reports/paired/floor.json
# the baseline, re-gated and benchmarked at this commit
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1,2,3 -n 0 --epochs 4 \
    --prefix paired-baseline-eval --extra "--weights \
    /runs/banking77-qwen15b-e4-lr1e-4-872ed726edcd-20260923T145427/seed{seed}.pt \
    --robustness-n 1000 --jaggedness-n 40"
# the treatment
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1,2,3 -n 0 --epochs 4 \
    --prefix paired-qwen15b-mix05-cw1 --extra "--backbone qwen2.5-1.5b --lr 0.0001 \
    --save-model model.pt --paired-mix 0.5 --consistency-weight 1.0 \
    --robustness-n 1000 --jaggedness-n 40"
# the ablation (at 3d48197); the re-read is the --weights form above on the cw1 adapters
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1,2,3 -n 0 --epochs 4 \
    --prefix paired-qwen15b-mix05-cw0 --extra "--backbone qwen2.5-1.5b --lr 0.0001 \
    --save-model model.pt --paired-mix 0.5 --consistency-weight 0 \
    --robustness-n 1000 --jaggedness-n 40"
```

The first two at commit `5b1fe12`, the ablation and the re-read at `3d48197`,
all on `NVIDIA A10`. The treatment launch is recorded
`dirty`: the uncommitted change was `docs/data.md` only, which no job
reads. The four treated adapters are on the `trigon-runs` Volume under
`banking77-paired-qwen15b-mix05-cw1-5b1fe12bac1e-20260927T005357`.

## Cost

| Run | GPU time | Estimated cost |
| --- | ---: | ---: |
| Baseline, four eval-only seeds (~23 min each) | 1.52 h | $1.7 |
| Treatment, four seeds (~117 min each: ~95 training, ~22 evaluating) | 7.80 h | $8.6 |
| Ablation, four seeds at weight 0 (~93 min each) | 6.2 h | $6.8 |
| Weight-1 checkpoints re-read (~22 min each) | 1.5 h | $1.6 |
| **Total** | **17.0 h** | **~$18.7** |

At Modal's A10G list price of about $1.10 an hour, from the `elapsed_s` each
seed recorded; container start-up and image builds are not counted. The
stream costs 1.65x the baseline's training time per seed (95 against 57
minutes): half again as many cases, and a second forward for every
anchored one.
