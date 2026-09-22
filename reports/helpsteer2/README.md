# HelpSteer2: the model collapsed to the marginal, and every calibration gate passed

HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0.
https://huggingface.co/datasets/nvidia/HelpSteer2

Five Score questions over one prompt-and-response pair. 1,400 training cases,
1,000 held out to fit the calibrator, 2 epochs, evaluated on 5,000 cases —
25,000 predictions, five times the sample-size floor.

**This is a failure, and it is the most useful result in the repository.**

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

## Why this is the useful result

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

## What this does and does not establish

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

## Three seeds, not four

Seed 0 was out-of-memory killed during evaluation. `CLAUDE.md` asks for a
median and range over four, so this is a measurement rather than a
certification.

It matters less than usual here. The rule exists because a single draw can
flatter or damn a configuration that is genuinely borderline, and nothing
about these three is borderline: the lifts are +0.0023, +0.0040 and −0.0020
against a gate of +0.05, and twelve of the fifteen question-level results are
*exactly* their own marginal. A fourth seed would have to disagree with the
first three by an order of magnitude to change the reading.
