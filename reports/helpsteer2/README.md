# HelpSteer2: more data moved it off the marginal; more training and a pretrained backbone barely did

## Read this first: accuracy is the wrong instrument here, and the models learned

Every run below fails `accuracy_over_baseline`, and that gate cannot be
cleared on this corpus by much of anything. `reports/helpsteer2/ceiling.md`
measures it from the annotators themselves: an oracle that knows the *other*
annotators' ratings of the same response predicts one annotator at **+0.021**
over the marginal, and one half-panel predicts the other half's average at
**−0.024**. The gate asks for +0.05 -- more agreement than the people have
with each other.

A proper scoring rule does separate the models from the population. Brier on
the same 25,000 predictions against the marginal distribution of each seed's
own training split:

| Run | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: | ---: |
| Spike, 6 epochs | +5.7% | +5.2% | +5.5% | +5.2% |
| Spike, 12 epochs | +5.7% | +5.4% | +5.6% | +5.5% |
| Qwen2.5-1.5B, lr 3e-4 | +1.9% | +7.9% | +9.4% | +9.2% |

Brier reduction over the marginal predictor; higher is better. Every run
reports each response's ratings better than the population does, the
backbone by nearly twice as much, and its one weak seed is the one that ran
at the learning rate that collapsed a Banking77 seed. Argmax accuracy hid all
of it, because the argmax of a five-level rating is mostly annotator noise.

What follows is the accuracy record as it was written, kept because it is
what the gates read.

## Qwen2.5-1.5B, 12,000 cases, 3 epochs, four seeds — `qwen15b-n12000-e3-seed*`

The model that took Banking77 from 72% to 90%, at the same 12,000 cases the
spike used. LoRA rank 16, lr 3e-4 (the rate that collapsed one Banking77
seed; 1e-4 was found after this was launched), commit `d52ab71`, all on
`NVIDIA A10` except seed 2 on an `A10G`, about 2 h 20 min a seed. Seed 2 was
preempted and restarted from step 0, which is why it finished last.

| Seed | Accuracy | Lift | ECE | Adaptive ECE | `complexity` | `verbosity` | `helpfulness` | `correctness` | `coherence` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5583 | +0.0059 | 0.0225 | 0.0284 | +0.0238 | +0.0058 | +0.0000 | +0.0000 | +0.0000 |
| 1 | 0.5744 | +0.0209 | 0.0120 | 0.0135 | +0.0920 | +0.0340 | −0.0092 | −0.0126 | +0.0002 |
| 2 | 0.5858 | **+0.0327** | 0.0102 | 0.0099 | +0.0876 | +0.0490 | +0.0154 | +0.0114 | +0.0000 |
| 3 | 0.5791 | **+0.0308** | 0.0119 | 0.0138 | +0.0958 | +0.0416 | +0.0118 | +0.0044 | +0.0006 |

**Median lift +0.0259 over four seeds, range +0.0059 to +0.0327, against the
spike's +0.0188** — and the gate is +0.05. Where Banking77 moved by eighteen
points, this moved by seven tenths of one, and the spread across seeds is
wider than the gain.

What the backbone does buy is the surface questions: `complexity` +0.092 to
+0.096 and `verbosity` +0.034 to +0.042 on seeds 1 and 3, above anything the
spike reached. And seeds 2 and 3 are the first runs of either model to
move the quality questions at all -- `helpfulness` +0.0154 and +0.0118,
`correctness` +0.0114 and +0.0044 -- while seeds 0 and 1 sit on or below the
marginal. Two seeds of four, by about a point: a direction worth following,
not yet a finding. `coherence` does not move on any seed of either model.

Seed 0 looks like a partial version of the Banking77 collapse: every
question but `complexity` on its marginal, validation loss barely moving
(1.156 → 1.117). It ran at the learning rate that collapsed a Banking77
seed, and a rerun at 1e-4 is the obvious next measurement -- not a reason to
drop it from the median.

**Calibration held on every seed** -- ECE 0.0102–0.0225, each above its floor
-- and the calibrator declined on all four, because a head this close to the
marginal has little to correct.

**What it establishes:** a 1.5B pretrained backbone with LoRA does not
learn HelpSteer2's quality judgements from 12,000 aggregated labels either.
The labels may be the ceiling: these are integer averages of annotators who
disagree, and the per-annotator `disagreements/` split -- the distribution the
product is meant to learn -- is still unloaded (`docs/next.md` A.2).

---


## 12,000 cases, 12 epochs, four seeds — `modal-n12000-e12-seed*`

The 6-epoch run below kept its last epoch on every seed with validation loss
still falling, and was written up as under-trained. **Doubling the epochs
says it was not.**

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Kept epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.5702 | +0.0178 | 0.0232 | 0.0226 | 6 |
| 1 | 0.5738 | +0.0202 | 0.0187 | 0.0188 | 11 |
| 2 | 0.5728 | +0.0197 | 0.0222 | 0.0229 | 8 |
| 3 | 0.5653 | +0.0170 | 0.0218 | 0.0220 | 8 |

**Median lift +0.0188, range +0.0170 to +0.0202** — against +0.0189 at six
epochs. Validation loss bottomed between epoch 6 and 11 on every seed and
rose after, so best-epoch selection kept an earlier epoch and the extra six
bought nothing. The per-question picture is the six-epoch one exactly:
`complexity` +0.067 to +0.074, `verbosity` +0.019 to +0.030, and
`coherence`, `correctness`, `helpfulness` on their marginals (−0.0038 to
+0.0010) on all four seeds.

Calibration got slightly **worse** — ECE 0.0187–0.0232 against 0.0082–0.0139
— which is what overfitting a head looks like, and still well inside the
gate; the calibrator declined itself on every seed again.

**What it establishes:** the 128-wide, two-layer spike has reached its
ceiling on HelpSteer2 at 12,000 cases. The three questions that judge quality
do not want more training; they want a different model, which is A.3.

Commit `22f4533`, clean tree; three seeds on devices reporting `NVIDIA A10`
and one on `NVIDIA A10G`, 61 to 73 minutes each
(`modal-n12000-e12-modal-run.json`).

---


HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0.
https://huggingface.co/datasets/nvidia/HelpSteer2

Two runs are recorded here. The first, at 1,400 cases, collapsed to the
marginal. The second, at 12,000 cases on a GPU, is the experiment the first
said it had not run. **It fails the same gate on every seed, and it is not the
same failure.**

## 12,000 cases, 6 epochs, four seeds — `modal-n12000-e6-seed*`

Five Score questions over one prompt-and-response pair. 12,000 training
cases (10% of them held out to pick the epoch), 1,000 held out to fit the
calibrator, evaluated on 5,000 cases — 1,038 from the corpus's test split and
3,962 held out of train, which neither training nor calibration saw. 25,000
predictions per seed, five times the sample-size floor.

Trained on Modal at commit `de60db3` on a clean tree, one container per seed,
each reporting itself as `NVIDIA A10` (an A10G was requested);
`modal-n12000-e6-modal-run.json` has the commit, the device and the wall clock — 32 to 34
minutes a seed, at 38 cases/s.

| Seed | Accuracy | Marginal | Lift | ECE | Adaptive ECE | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.5716 | 0.5524 | +0.0192 | 0.0082 | 0.0106 | **FAIL** |
| 1 | 0.5721 | 0.5536 | +0.0185 | 0.0103 | 0.0144 | **FAIL** |
| 2 | 0.5734 | 0.5531 | +0.0203 | 0.0083 | 0.0099 | **FAIL** |
| 3 | 0.5638 | 0.5482 | +0.0155 | 0.0139 | 0.0168 | **FAIL** |

**Median lift +0.0189, range +0.0155 to +0.0203**, against a gate of +0.05.
At 1,400 cases it was +0.0023. The spread is narrow — this configuration
does not decide its outcome by seed — so the median is a fair reading of it.

Per question, all four seeds:

| Question | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: | ---: |
| `complexity` | +0.0802 | +0.0688 | +0.0780 | +0.0608 |
| `verbosity` | +0.0242 | +0.0254 | +0.0250 | +0.0170 |
| `coherence` | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `correctness` | −0.0044 | −0.0002 | −0.0006 | +0.0000 |
| `helpfulness` | −0.0042 | −0.0014 | −0.0008 | −0.0002 |

### How to read it

**It is no longer a collapse.** At 1,400 cases four of five questions sat
exactly on their marginal and `complexity` moved by between +0.0024 and
+0.0202, which could not be told apart from noise. Here `complexity` is
learned on every seed at +0.06 to +0.08 and `verbosity` at +0.017 to +0.025,
both with a spread far smaller than the effect. Those are real.

**Three questions are still not learned, and they are the three that ask
about quality.** `coherence` is exactly on its marginal on all four seeds,
and `correctness` and `helpfulness` are on it or just below. `complexity` and
`verbosity` are properties of the response's *surface* — its length, its
vocabulary — which a 128-wide, two-layer encoder over hashed-BPE tokens can
plausibly read. Whether a response is *correct* or *helpful* is a judgement
about the prompt and the response together. That split is suggestive and it
is not established: nothing here separates "this model cannot represent
quality" from "this model has not trained long enough to".

**It has not trained long enough to tell.** Every seed kept its **last**
epoch, and validation loss was still falling at epoch 6 on all four (seed 0:
1.1058 → 1.0632, still −0.0016 over the last epoch). A run that selects its
final epoch has not found its best one. The next experiment is more epochs at
this size, not more data — the cheapest intervention that the data here says
is unexhausted.

**Calibration held, and it is the model's own.** ECE 0.0082–0.0139, adaptive
0.0099–0.0168, against a 0.05 limit and a perfect-calibration floor of about
0.0069 (95th percentile 0.0100): real numbers, measured well clear of the
instrument. The calibrator declined itself on all four seeds — uncalibrated
and calibrated rows are identical — so no fitted map is flattering them.
`CLAUDE.md`'s fourth rule is again the whole story: the ECE column alone would
certify this model, and `accuracy_over_baseline` is the only thing that does
not.

### What it took to run

The first launch of this configuration died four minutes in. HelpSteer2's
longest case compiles to 7,171 tokens, length bucketing correctly puts it
beside the next-longest, and a chunk of eight asked a 22 GiB A10 for 6.13 GiB
in one allocation. `--max-batch-cells` now splits such a step into
sub-batches that accumulate into the same update; `tests/test_training.py`
asserts each optimizer step receives the same gradient either way.

Before that, the launcher would have trained on the container's CPU while
recording the GPU: nothing in the backend moved a tensor to a device. And the
training path was still building every attention mask in Python, 352 ms a
request against 10.2 ms. Both are in `docs/ledger.md`.

---

## 1,400 cases, 2 epochs, three seeds — `hs2-seed*`

Five Score questions over one prompt-and-response pair. 1,400 training cases,
1,000 held out to fit the calibrator, 2 epochs, evaluated on 5,000 cases —
25,000 predictions, five times the sample-size floor.

**This was a failure, and the most useful result in the repository when it was run.**

| Seed | Accuracy | Marginal | Lift | ECE | Adaptive ECE | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.5559 | 0.5536 | **+0.0023** | 0.0144 | 0.0258 | **FAIL** |
| 2 | 0.5571 | 0.5531 | **+0.0040** | 0.0207 | 0.0248 | **FAIL** |
| 3 | 0.5463 | 0.5482 | **−0.0020** | 0.0108 | 0.0118 | **FAIL** |

Median lift **+0.0023**, range −0.0020 to +0.0040, against a gate of +0.05.
The spread is narrow and every draw is indistinguishable from the marginal
predictor — this is not a seed that went badly.

Per question, seed 2:

| Question | Accuracy | Its marginal | Lift |
| --- | ---: | ---: | ---: |
| `coherence` | 0.7164 | 0.7164 | **+0.0000** |
| `correctness` | 0.4808 | 0.4808 | **+0.0000** |
| `helpfulness` | 0.4186 | 0.4186 | **+0.0000** |
| `verbosity` | 0.6210 | 0.6210 | **+0.0000** |
| `complexity` | 0.5488 | 0.5286 | +0.0202 |

Four of five, exactly on the marginal, to four decimal places. Seed 1 is
identical in shape — the same four at `+0.0000`, `complexity` at +0.0116 —
and seed 3 has three of five exactly on it with `helpfulness` **below** it at
−0.0122.

`complexity` is the only question that moves at all, on all three seeds, and
by between +0.0024 and +0.0202. That is small enough that it may be the
easiest question rather than evidence of learning, and nothing here
distinguishes those.

### Why this is the useful result

`CLAUDE.md`'s fourth ground rule, written long before this corpus existed:

> **Calibration never certifies alone.** A model that reports each question's
> marginal distribution is calibrated by construction and useless, and it
> passes every ECE gate. `accuracy_over_baseline` is what rejects it.

That is exactly what happened, on real data, for the first time:

- **Every calibration gate passed.** ECE 0.0207 and 0.0108, adaptive 0.0248
  and 0.0118, against a 0.05 limit.
- **The numbers are real, not noise.** `sample_size` passed at 25,000 and
  `gate_is_testable` at 0.0091 against a floor of 0.0055 — the measurement is
  well clear of the instrument, which the two-seed Banking77 pilot was not.
- **The calibrator declined itself on seeds 1 and 2**, correctly: a model
  predicting the marginal is calibrated by construction, so there is nothing
  to fix, and their uncalibrated and calibrated suite rows are identical to
  four decimals. On seed 3 it was applied and took ECE from 0.0317 to 0.0108
  — improving the calibration of a model that had learned nothing.
- **`accuracy_over_baseline` failed on all three seeds**, and
  `worst_question_over_baseline` named the question each time.

A reader shown only the ECE column would conclude this model is excellent.

### What this does and does not establish

**Established:** at 1,400 training cases and 2 epochs, this architecture
learns nothing about HelpSteer2's five ordinal ratings, and the gate set
catches it rather than certifying it.

**Not established:** that Score cannot learn this corpus. 1,400 cases is
small — Banking77 needed 7,083 to reach 74% — and the size was chosen to fit
inside a cloud session's idle window rather than because it was enough. Three
earlier attempts at 5,000–6,000 cases died: one to the mask-cache
out-of-memory bug, one to a 45-hour projection before the mask build was
vectorized, one to a VM reclamation ninety minutes in. `docs/gpu-access.md`
explains why a run longer than the idle window cannot finish in this
environment at all.

So the honest reading is: **the pipeline works and the gates work; the
experiment has not been run at a size that could answer the question.**

The 12,000-case run above is that experiment. It moved two questions off
the marginal and not the other three.

### Three seeds, not four

Seed 0 was out-of-memory killed during evaluation. `CLAUDE.md` asks for a
median and range over four, so this is a measurement rather than a
certification.

It matters less than usual here. The rule exists because a single draw can
flatter or damn a configuration that is genuinely borderline, and nothing
about these three is borderline: the lifts are +0.0023, +0.0040 and −0.0020
against a gate of +0.05, and twelve of the fifteen question-level results are
*exactly* their own marginal. A fourth seed would have to disagree with the
first three by an order of magnitude to change the reading.
