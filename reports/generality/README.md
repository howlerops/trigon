# Generality: does a model answer the caller's question, or its training set's?

The drop-in claim is that a caller who declares *their own* options and
questions gets a usable, calibrated answer. This is the first measurement of
that claim. Every number here comes from `scripts/generality.py` on the cases
`trigon.evals.generality.build` draws (seed 20260930, 1,000 per task unless
noted), so every row answered the same questions.

| Task | What it asks |
| --- | --- |
| `banking77/declared` | the 77 options Banking77 models are trained on, in order — the control |
| `banking77/shift` | 50 of those options (the true one always present), shuffled |
| `banking77/renamed` | the same, every option name redrawn in another case style |
| `clinc150` | 150 intents no mix trains on, 50 per case |
| `boolq` | yes/no about a passage, a different question on every case (eval-only licence) |
| `banking77/order` | `banking77/shift` re-ordered; *agreement* is how often the answer survives |

## Results

Accuracy; chance is 0.02 on every Choice task (0.013 at 77 options) and 0.50
on BoolQ, where always answering "yes" scores 0.62.

| System | declared | shift | renamed | clinc150 | boolq | order agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A compatible hosted service (2026-09-30) | over its cap | 0.544 | 0.395 | **0.849** | **0.758** | 0.641 |
| v2, the served model (1.5B, Banking77, declared set) | **0.883** | 0.012 | 0.018 | 0.026 | 0.398 | 0.024 |
| 0.5B, Banking77, `--reshape-max 77` | 0.011 | **0.782** | **0.677** | 0.420 | 0.369 | **0.867** |
| 0.5B, Banking77, reshape + crossover balance 0.5 | 0.030 | 0.664 | 0.565 | 0.451 | 0.369 | 0.802 |
| 0.5B, mix of six corpora, 2 epochs, crossover 256 | 0.358 | 0.427 | 0.366 | 0.226 | 0.605 | 0.613 |
| the same mix, resumed to 4 epochs | 0.602 | 0.680 | 0.586 | 0.251 | 0.586 | 0.824 |
| **the same mix on Qwen3-0.6B**, 4 epochs from the start | 0.663 | 0.729 | 0.653 | **0.585** | 0.565 | 0.832 |
| MiniCPM5-1B, the mix plus 1,464 local-teacher cases (workflows and documents), 4 epochs | **0.751** | **0.806** | **0.695** | 0.487 | 0.524 | **0.919** |

Every 0.5B row is one seed (seed 0) trained locally on Apple MPS, and is a
measurement, not a certification. ECE beside its noise floor is in each
`results.md`.

## What it shows

**1. The served model answers its training set, not the caller.** 0.883 on
the 77 options it was trained on and at or below chance everywhere else. Its
answers to the same 50 options in a different order agree 2.4% of the time.

**2. The cause was a switch, not a habit.** `DOT_PRODUCT_CROSSOVER` sends a
Choice over 64 options to the dot-product head and one at or under it to the
per-option readout. Trained only at 77, v2 never trained the head that
answers 50. The 0.5B reshape model is the mirror image: trained mostly under
64, it answers 50 options at 0.782 and 77 at 0.011. The crossover is now a
property of the checkpoint (`ReadoutConfig.option_crossover`), saved with the
weights; checkpoints that record none keep 64.

**3. Reshaping teaches a model to read its options.** Trained on option sets
redrawn per case per epoch, a model a third of v2's size answers shuffled
and renamed sets far better than the hosted service (+0.24, +0.28), keeps
its answer under re-ordering 87% of the time against the service's 64%, and
transfers to CLINC150 at 0.42 from Banking77 alone. Uncalibrated, its ECE on
the 50-option tasks is within or near the noise floor.

**4. Balancing training across the crossover does not rescue the
dot-product head.** At half the data share it moved 77-option accuracy from
0.011 to 0.030 and cost the per-option head 12 points. Hence crossover 256
for new checkpoints rather than training both.

**5. The mix is undertrained, and is not yet better.** Six corpora, two
epochs: validation loss was still falling steeply (1.66 → 1.28). It reads
77 options again (0.358) and BoolQ moves off chance (0.605) without passing
the always-yes rate (0.62); everything else is below the Banking77-only
reshape model, CLINC150 included. Calibration held: no calibrator was
needed on any primitive, and the Banking77 tasks sit within their floors.
**Four epochs improve what the mix contains and not what it does not.**
Resumed to four epochs (the schedule recomputed for four at the resume, so
not identical to a four-epoch run from the start): Banking77 rises to 0.680
shifted, 0.586 renamed, 0.824 order agreement and 0.602 at 77 options;
CLINC150 stays at 0.251 and BoolQ at 0.586, below always-yes. Calibration
moved the wrong way, ECE 0.086–0.116 against floors near 0.045, and no
calibrator was accepted on held-out calibration data (choice 0.0607 raw
against 0.0518 fitted, inside its noise). **Held-out task transfer is not a
matter of more epochs on these six corpora at 0.5B.**

**The backbone was the bottleneck on held-out tasks.** The same six-corpus mix on Qwen3-0.6B (2025, post-trained, QK-norm; `scripts/backbone_parity.py` exact) instead of Qwen2.5-0.5B more than doubles CLINC150, 0.251 to 0.585, and lifts every Banking77 task. BoolQ does not move (0.565, still under always-yes). Calibration on the held-out tasks is not there yet: ECE 0.168 on CLINC150 and 0.161 on BoolQ against floors near 0.043; the cross-task calibrator applied temperatures to Noul and Score and declined Choice. Trained in 8 h on Apple MPS — 2× Qwen2.5-0.5B's time, as its width predicts once nothing else is resident in unified memory (an 18 GB ollama model alongside made it 5×).

**MiniCPM5-1B with the local teacher is the strongest on what the mix contains and the weakest on transfer.** Best on every Banking77 task, 0.919 order agreement, and ECE within or near its floor on all four Banking77 tasks with no calibrator applied; but CLINC150 0.487 against Qwen3-0.6B's 0.585, and BoolQ 0.524. Two things changed at once -- backbone and data -- so this does not say which; the document-reading teacher cases, built for BoolQ's shape, did not lift BoolQ.

**6. The hosted service is not order-invariant either**, and depends on
option wording: re-ordering identical options changes 36% of its answers,
and renaming them costs it 15 points. It caps a Choice at 50 options and
refuses a Noul without criteria — neither is in `COMPAT_BUDGET`.

## CPU serving

`banking77/shift`, `clinc150` and `boolq` at n=500 on the 0.5B reshape
model, four threads on an Apple M-series CPU, every request a schema-cache
miss (each case's options differ):

| | shift | clinc150 | boolq | p50 shift | p50 clinc150 | p50 boolq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 0.772 | 0.416 | 0.364 | 0.35 s | 0.29 s | 0.28 s |
| int8 (`TRIGON_INT8=1`) | 0.748 | 0.410 | 0.364 | 0.82 s | 0.60 s | 0.43 s |

**int8 is slower here, not faster**: Apple's float32 matmuls run on
Accelerate, and PyTorch's ARM int8 kernels (qnnpack) have nothing like it.
x86 (fbgemm) is unmeasured. Float32 0.5B at ~0.3 s per cache-missing request
on a laptop CPU is the number that matters for "no L4".

## Reproduce

```bash
python scripts/train_corpus.py banking77 --backbone qwen2.5-0.5b -n 0 --epochs 3 \
  --lr 1e-4 --calibration-n 1000 --device mps --seed 0 \
  --reshape-max 77 --reshape-rename 0.3 --reshape-criteria-only 0.1 \
  --reshape-crossover-fraction 0 --save-model reshape-s0.pt --out reshape-s0.md
python scripts/train_mix.py --backbone qwen2.5-0.5b --device mps --epochs 2 --seed 0 \
  --out mix-05b-s0
python scripts/generality.py --bundle mix-05b-s0 --out reports/generality/mix-05b-s0
```

The first command predates `--reshape-crossover-fraction` and ran with its
current default of 0; the flag is written out here so the command still
reproduces the row.

Corpora: Banking77 (Casanueva et al., 2020), PolyAI, CC BY 4.0. CLINC150
(Larson et al., 2019), Clinc Inc., CC BY 3.0. BoolQ (Clark et al., 2019),
Google, CC BY-SA 3.0 — evaluation only. The mix's own corpora are listed with
their counts in `mix-05b-s0/mix.json`.
