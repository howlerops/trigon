# Does the teacher stream transfer? Banking77, four seeds, two budgets

**Measured, 2026-09-28.** The open item was: *whether training on it moves
Banking77 or the synthetic suite is unrun.* This measures Banking77. The
synthetic suite is still unrun.

**The answer is no, and at a small budget it is worse than no.** The four
teacher-trained Qwen2.5-1.5B students (`README.md` beside this file) were
used as the starting point for the certified Banking77 recipe. Every other
setting was unchanged.

- **At 1,000 Banking77 cases, the teacher init loses badly.** Median accuracy
  is **0.0895** (0.0228–0.6990) after the teacher init. From the base model
  it is **0.6549** (0.6054–0.6942), on the same splits, commit and flags.
  Three of the four teacher-initialised seeds are still near chance after four
  epochs. The base model left chance on all four seeds.
- **At the full budget, the teacher init does not certify.** Three seeds
  reach 0.86–0.91. Seed 3 is at **0.0476** after four epochs, with
  validation loss still 3.61, where ln 77 = 4.34. Median accuracy is 0.8787,
  against the certified 0.9009. The certified run passed every gate on all
  four seeds. This run fails `accuracy_over_baseline` on one seed.
- **When the teacher init does learn, it gets to about where the base
  gets.** On seeds 0 and 2 it reaches 0.9070 and 0.8928, with ECE 0.0110 and
  0.0140. The certified seeds are at 0.8502–0.9054. Nothing on this page
  suggests the stream adds anything to a model that has left the plateau.

Coverage transfer was expected to show first at the small budget. Instead,
the small budget is where the harm shows. The stream remains **coverage
believed, transfer measured negative on Banking77**.

## Design

The cheapest honest test is to continue training what already exists. The
comparison is certified recipe from the base model against certified recipe
from each teacher student, with the seeds paired. Teacher seed *s* initialises
Banking77 seed *s*. The Banking77 seed fixes the splits and the data order,
so each pair sees the same data in the same order and differs only in its
starting weights.

| | |
| --- | --- |
| Recipe | `reports/banking77/README.md`, `qwen15b-e4-lr1e-4`: lr 1e-4, 4 epochs, LoRA 16, chunks of 8, calibration 1,000, `--max-batch-cells 50000000`, best epoch kept |
| Splits | `splits()` keyed on `corpus:banking77:{seed}`. Calibration and evaluation (5,000: 3,080 test and 1,920 held out of train) are the same for every arm at a given seed, and the same as the certified runs |
| Small budget | `-n 1000`, the first 1,000 of the same shuffled training pool. 100 of them go to the trainer's validation split |
| Init | the kept epoch (2 of 3) of each `teacher-workflows-qwen15b-e3` seed |
| Commit | `3b6046d`, clean tree, for every run on this page |
| Hardware | A10 or A10G on Modal, as recorded per seed in `transfer/*-modal-run.json` |

**Two controls, because the certified runs are from another commit.** The
four certified seeds were trained at `872ed72`, 100 commits earlier. Length
bucketing and other trainer changes have landed since then. Their evaluation
still reproduces at a recent commit: the `paired-baseline-eval` re-read at
`5b1fe12` matches the `regate080` rows to four decimals. Their *training* is
not known to reproduce. So base-model runs at `3b6046d` are included too:
four seeds at the small budget, and seeds 0 and 1 at the full budget, as a
drift check. Two seeds were what the budget allowed.

### Getting the init: resume files, not checkpoints

The teacher runs were launched without `--save-model`, so only their resume
files are on the `trigon-runs` Volume. A resume file carries the kept epoch's
trainables (`best`), but it does not record the backbone, the LoRA rank or the
tokenizer. `scripts/resume_to_checkpoint.py` is told those values from the
run's own flags (`qwen15b-e3-modal-run.json`: `--backbone qwen2.5-1.5b`,
rank 16 by default). It refuses a file whose trainable tensor names or
shapes do not fit that model. It also refuses a file whose kept epoch and
validation loss do not match the run's `-training.json`. All four matched:
epoch 2 on each, with validation loss 1.076774, 1.067325, 1.028407 and
1.103894. The converted checkpoints are on the Volume at
`teacher-init/qwen15b-e3-seed{0,1,2,3}.pt`, named
`trigon-qwen2.5-1.5b-0.1.0+{770672fa, bb8d998c, 3c87aa69, cf72125a}`.

`train_corpus.py --init-weights` loads one of these checkpoints and then
trains. It refuses a checkpoint whose recorded backbone, LoRA rank, spike
shape or tokenizer `{kind, vocab_size}` differs from what the run's flags
would build (`tests/test_init_weights.py`). A continued build is named after
its own weights and after its init. Every model below that starts from the
teacher answers as `trigon-qwen2.5-1.5b-0.1.0+<digest>.init.<teacher digest>`,
so it cannot be mistaken for a from-scratch build.

## Small budget: 1,000 Banking77 cases

The columns are: accuracy, lift over the marginal predictor, ECE and adaptive
ECE after calibration, the noise floor's p95 at n = 5,000, uncalibrated ECE,
NLL, the calibrator the held-out check chose, the epoch kept, and the blocking
gates.

**From the base model** (`transfer/transfer-base-n1000-seed*`)

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Floor p95 | Uncal. ECE | NLL | Calibrator | Kept | Gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0.6054 | +0.5894 | 0.0462 | 0.0463 | 0.0219 | 0.1637 | 1.5978 | isotonic | 3 | PASS |
| 1 | 0.6942 | +0.6776 | 0.0498 | 0.0504 | 0.0197 | 0.1298 | 1.3163 | isotonic | 3 | FAIL: workhorse_adaptive_ece |
| 2 | 0.6796 | +0.6622 | 0.0291 | 0.0317 | 0.0219 | 0.0821 | 1.1721 | T = 1.40 | 3 | PASS |
| 3 | 0.6302 | +0.6140 | 0.0247 | 0.0171 | 0.0222 | 0.0876 | 1.3150 | T = 1.26 | 3 | PASS |
| **median** | **0.6549** | **+0.6381** | **0.0377** | 0.0390 | | 0.1087 | 1.3157 | | | 3 of 4 |

**From the teacher student** (`transfer/transfer-teacher-n1000-seed*`)

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Floor p95 | Uncal. ECE | NLL | Calibrator | Kept | Gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0.6990 | +0.6830 | 0.0522 | 0.0518 | 0.0191 | 0.0522 | 1.2080 | declined | 4 | FAIL: workhorse_ece, workhorse_adaptive_ece, worst_primitive_workhorse_ece |
| 1 | 0.0488 | +0.0322 | 0.0109 | 0.0244 | 0.0074 | 0.0109 | 4.0717 | declined | 4 | FAIL: accuracy_over_baseline, worst_question_over_baseline |
| 2 | 0.1302 | +0.1128 | 0.0210 | 0.0216 | 0.0109 | 0.0472 | 3.6856 | isotonic | 4 | PASS |
| 3 | 0.0228 | +0.0066 | 0.0181 | 0.0189 | 0.0049 | 0.0219 | 4.1073 | T = 1.11 | 4 | FAIL: accuracy_over_baseline, worst_question_over_baseline |
| **median** | **0.0895** | **+0.0725** | 0.0195 | 0.0230 | | 0.0345 | 3.8786 | | | 1 of 4 |

Every ECE on this page is at n = 5,000 and above its own floor's p95, so each
one is a measurement. At this budget the ECE does not favour the teacher
init. Its seeds 1–3 are calibrated because they are close to the marginal
predictor, which is calibrated by construction. That is the failure mode
`accuracy_over_baseline` exists to catch.

**The gate set certified a 13%-accurate model.** Teacher seed 2 passes every
blocking gate at accuracy 0.1302. Its lift of +0.113 clears
`accuracy_over_baseline`'s 0.05, and its ECE of 0.0210 clears 0.05. No gate
was relaxed to get there. The baseline term is simply that permissive on a
77-way question. It is recorded here as found and is not acted on.

**Where the difference comes from: leaving the plateau.** Validation loss by
epoch (ln 77 = 4.34):

| Seed | Base, n = 1,000 | Teacher init, n = 1,000 |
| ---: | --- | --- |
| 0 | 3.74, 2.65, **2.47**, 2.69 | 4.35, 3.43, 1.73, **1.66** |
| 1 | 3.40, 1.74, **1.67**, 1.93 | 4.36, 4.32, 4.24, **4.20** |
| 2 | 3.32, 1.69, **1.31**, 1.31 | 4.34, 4.34, 3.97, **3.74** |
| 3 | 4.23, 1.85, **1.30**, 1.32 | 4.33, 4.26, 4.18, **4.11** |

Every base seed is below the plateau after epoch 1 or 2. Every
teacher-initialised seed is still on it after epoch 1. Three are still near it
after epoch 4, and they are still falling. The one teacher seed that did leave
(seed 0) ends with the best accuracy of the eight (0.6990) and an NLL
inside the base range (1.21, where the base seeds are at 1.17–1.60). It fails
calibration because its held-out check declined the calibrator. That is one
seed of four, and it is not evidence of a benefit.

## Full budget: 7,083 Banking77 cases

**From the teacher student** (`transfer/transfer-teacher-full-seed*`)

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Floor p95 | Uncal. ECE | NLL | Calibrator | Kept | Gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0.9070 | +0.8910 | 0.0110 | 0.0051 | 0.0108 | 0.0485 | 0.4026 | isotonic | 2 | PASS |
| 1 | 0.8646 | +0.8480 | 0.0376 | 0.0376 | 0.0130 | 0.0376 | 0.5116 | declined | 3 | PASS |
| 2 | 0.8928 | +0.8754 | 0.0140 | 0.0142 | 0.0114 | 0.0272 | 0.4224 | isotonic | 2 | PASS |
| 3 | 0.0476 | +0.0314 | 0.0226 | 0.0190 | 0.0079 | 0.0226 | 3.5681 | declined | 4 | FAIL: accuracy_over_baseline, worst_question_over_baseline |
| **median** | **0.8787** | **+0.8617** | 0.0183 | 0.0166 | | 0.0324 | 0.4670 | | | 3 of 4 |

**From the base model at this commit, seeds 0 and 1: the drift control** (`transfer/transfer-base-full-seed*`)

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Floor p95 | Uncal. ECE | NLL | Calibrator | Kept | Gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0.9078 | +0.8918 | 0.0146 | 0.0164 | 0.0141 | 0.0532 | 0.3593 | T = 2.04 | 2 | PASS |
| 1 | 0.9046 | +0.8880 | 0.0158 | 0.0174 | 0.0139 | 0.0506 | 0.3651 | T = 1.77 | 2 | PASS |

**The paired comparison.** Same commit, same seed, same data and data
order; only the initial weights differ:

| Seed | Base | Teacher init | Difference |
| ---: | ---: | ---: | ---: |
| 0 | 0.9078 | 0.9070 | −0.0008 |
| 1 | 0.9046 | 0.8646 | **−0.0400** |

On 5,000 cases, one standard error of an accuracy near 0.9 is about 0.4
points. Seed 0 shows no difference. Seed 1 is ten standard errors worse.

**Certified, for reference** (`reports/banking77/qwen15b-e4-lr1e-4-regate080-seed*`, trained at `872ed72`, re-gated under the current calibrator rule)

| Seed | Accuracy | Lift | ECE | Adaptive ECE | Floor p95 | Uncal. ECE | NLL | Calibrator | Kept | Gates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0.9054 | +0.8894 | 0.0202 | 0.0112 | 0.0116 | 0.0489 | 0.3877 | isotonic | 2 | PASS |
| 1 | 0.8980 | +0.8814 | 0.0448 | 0.0445 | 0.0102 | 0.0448 | 0.4590 | declined | 2 | PASS |
| 2 | 0.9038 | +0.8864 | 0.0216 | 0.0216 | 0.0115 | 0.0534 | 0.4052 | isotonic | 2 | PASS |
| 3 | 0.8502 | +0.8340 | 0.0105 | 0.0106 | 0.0139 | 0.0195 | 0.5327 | isotonic | 1 | PASS |
| **median** | **0.9009** | **+0.8839** | 0.0209 | 0.0164 | | 0.0468 | 0.4321 | | | 4 of 4 |

The ECE floor is 0.0079–0.0139 on every row above. Seed 3 of the certified
run is the one row whose ECE (0.0105) is *not* separable from its floor
(p95 0.0139). That is a model indistinguishable from perfectly calibrated,
not a measurement of miscalibration.

**Validation loss by epoch, full budget:**

| Seed | Certified (`872ed72`) | Teacher init |
| ---: | --- | --- |
| 0 | 0.52, **0.43**, 0.56, 0.61 | 0.56, **0.45**, 0.52, 0.57 |
| 1 | 0.85, **0.45**, 0.50, 0.52 | 3.65, 1.69, **0.52**, 0.63 |
| 2 | 0.83, **0.43**, 0.61, 0.65 | 1.44, **0.44**, 0.48, 0.53 |
| 3 | **0.62**, 0.66, 0.66, 0.75 | 4.34, 4.09, 3.67, **3.61** |

Seed 0's teacher init trains like the base model does. Seeds 1 and 2 lose
one epoch or more to the plateau and then catch up. Seed 3 spends all 3,188
steps leaving it and does not get there. At step 400 the training loss was
2.66, 4.28, 4.36 and 4.35 for teacher seeds 0–3, and 2.82 and 2.79 for base
seeds 0 and 1 at this commit.

## Verdict

**Initialising from the teacher-trained student does not help Banking77 at
either budget, and it makes training less reliable.**

- **Accuracy**: median 0.0895 against 0.6549 at 1,000 cases, and 0.8787
  against 0.9009 (certified) at 7,083. The full-budget median is pulled
  down by one seed at chance. The three that learned are at 0.8646–0.9070,
  inside the certified range of 0.8502–0.9054. Paired at this commit, the
  teacher init is −0.0008 on seed 0 and −0.0400 on seed 1. No seed at
  either budget gained from it by more than noise, apart from small-budget
  seed 0.
- **Calibration**: no separable difference. Where a teacher-initialised seed
  learned, its ECE (0.0110–0.0376) is inside the certified range
  (0.0105–0.0448). Where it did not, its ECE is low because it is the
  marginal predictor.
- **Gates**: the full-budget configuration **does not certify**. Its spread
  includes a model at 4.8%, exactly as the lr 3e-4 run's did.
- **Data efficiency**, the place where coverage was expected to show first,
  is where the stream does the most damage.

**What moves from believed to measured.** The stream was built for
*coverage*, and coverage was believed. On Banking77, used as an init, it is
now measured, and the result is negative. One exception is on the page and
should not be read as a trend: small-budget seed 0 gains 9.4 points over its
paired base seed, and fails calibration.

## What is believed and not measured

- **Why it hurts.** The student learned the teacher's position prior
  (`README.md`, *A student trained on it*). The belief is that its heads and
  its already-moved LoRA adapters start Banking77 near the uniform solution
  with a flatter gradient than fresh heads, so leaving ln 77 takes longer.
  The step-400 losses are consistent with that. Nothing here isolates it: for
  example, re-drawing the heads while keeping the teacher's adapters, or the
  reverse, would. The certified lr 3e-4 seed 2 is the other known way to sit
  at ln 77, and there the mechanism was different: it learned and then
  collapsed.
- **That more epochs would rescue seed 3.** Its validation loss was still
  falling (4.34 → 3.61). A fifth or sixth epoch might certify it. That would
  still be a slower and less reliable configuration than the base model.
- **That this generalises beyond Banking77.** One 77-way Choice corpus was
  tested. The synthetic suite (three primitives, where the teacher student
  gained a little on Score) is **still unrun**, and so is any mixed-stream
  training, as opposed to sequential init.
- **Code drift since `872ed72`.** It is measured on two seeds, not four. At `3b6046d` the
  base model's seeds 0 and 1 reach 0.9078 and 0.9046. At `872ed72` they
  reached 0.9054 and 0.8980 (0.9118 before the calibrator rule changed).
  Epoch-1 validation loss is 0.53 and 0.54 at this commit, against 0.52 and
  0.85 at `872ed72`. Both are inside the certified range, so comparing with
  the certified seeds is not distorted by drift the two seeds can see. Seeds 2
  and 3 of the base model were not rerun at this commit, and whether they
  would reproduce is not measured.

## Cost

| Arm | Runs | Container time | Dollars |
| --- | ---: | ---: | ---: |
| Teacher init, n = 1,000 | 4 | 4,782 s | $2.05 |
| Base, n = 1,000 | 4 | 5,057 s | $2.17 |
| Teacher init, full | 4 | 16,835 s | $7.22 |
| Base, full (drift control) | 2 | 8,326 s | $3.57 |
| **Total** | 14 | 35,000 s (9.7 h) | **$15.01** |

These are priced at the rate `README.md` uses: A10G with 4 CPU cores and
32 GiB, $0.000429/s. Container time is measured inside the function, so
container start and the backbone download are not counted. Converting the
four resume files ran on this session's CPU and cost nothing on Modal. The
full-budget runs took 69–71 minutes each at this commit, against 57–60 at
`872ed72`: about 805 s an epoch against 700. Nothing here says why.

## Commands

```bash
# the init: resume files -> checkpoints (CPU, ~5 min each; the backbone is cached locally)
modal volume get trigon-runs teacher-workflows-qwen15b-e3-24ebf8422cc8-20260927T022435/seed0.resume.pt seed0.resume.pt
python scripts/resume_to_checkpoint.py seed0.resume.pt --out qwen15b-e3-seed0.pt \
    --backbone qwen2.5-1.5b --lora-rank 16 --expect-epoch 2 --expect-validation 1.076773528955452
modal volume put trigon-runs qwen15b-e3-seed0.pt teacher-init/qwen15b-e3-seed0.pt
# (seeds 1-3 alike: run ids in qwen15b-e3-modal-run.json, kept epoch and loss in -training.json)

# small budget, two seeds at a time (the 2-container cap)
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1 -n 1000 --epochs 4 \
    --prefix transfer-teacher-n1000 --extra "--backbone qwen2.5-1.5b --lr 0.0001 \
    --init-weights /runs/teacher-init/qwen15b-e3-seed{seed}.pt"
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1 -n 1000 --epochs 4 \
    --prefix transfer-base-n1000 --extra "--backbone qwen2.5-1.5b --lr 0.0001"
# full budget
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1 -n 0 --epochs 4 \
    --prefix transfer-teacher-full --extra "--backbone qwen2.5-1.5b --lr 0.0001 \
    --save-model model.pt --init-weights /runs/teacher-init/qwen15b-e3-seed{seed}.pt"
python scripts/modal_train.py launch --corpus banking77 --seeds 0,1 -n 0 --epochs 4 \
    --prefix transfer-base-full --extra "--backbone qwen2.5-1.5b --lr 0.0001 --save-model model.pt"
python scripts/modal_train.py collect <run id> --out-dir reports/teacher/transfer
```

Run ids, all at `3b6046d`:
`banking77-transfer-teacher-n1000-3b6046d5cc90-20260928T173836` (seeds 0, 1),
`…-teacher-n1000-…-20260928T181806` (2, 3),
`banking77-transfer-base-n1000-3b6046d5cc90-20260928T175812` (0, 1),
`…-base-n1000-…-20260928T184251` (2, 3),
`banking77-transfer-teacher-full-3b6046d5cc90-20260928T190651` (0, 1),
`…-teacher-full-…-20260928T201826` (2, 3),
`banking77-transfer-base-full-3b6046d5cc90-20260928T212843` (0, 1).
The full-budget adapters are on the Volume (`--save-model`) and are not
committed.
