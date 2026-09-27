# HelpSteer2 on Qwen2.5-7B: every seed gains a point over 1.5B, and none reaches the gate

The aggregated `helpsteer2` target on Qwen2.5-7B (`Qwen/Qwen2.5-7B` at
`d149729398750b98c0af14eb82c78cfe92750796`, Apache-2.0), four seeds. The
recipe is the certified-configuration run for 1.5B,
`reports/helpsteer2/qwen15b-n12000-e3-lr1e-4-*`, with only the backbone
changed: 12,000 cases, 3 epochs, lr 1e-4, LoRA rank 16, accumulate 8,
`--max-batch-cells 50000000`, 1,000 calibration cases, 5,000 evaluation
cases (25,000 predictions). Nothing had to change for memory or time.

**Verdict: fails `accuracy_over_baseline` on all four seeds.** The median
lift is **+0.0389**, and the range is +0.0364 to +0.0400, against a gate of
+0.05. That is +0.012 over 1.5B's median of +0.0268. The gain is paired
and it is not noise: on the same seeds, which give the same splits, 7B is
ahead of 1.5B on every seed by +0.0095 to +0.0155. The smallest of those
gains is twice the width of 1.5B's whole spread (0.0047). The gate still
needs another +0.011 that nothing here supplies. Calibration passes on
every seed, and one seed's ECE is below its own noise floor.

## Per seed — `qwen7b-n12000-e3-lr1e-4-seed*`

Commit `0f6fb07`, clean tree. Seed 0 ran on `NVIDIA A100 80GB PCIe`, and
seeds 1–3 on `NVIDIA A100-SXM4-80GB`. Each seed ran in one life, with no
preemption.

| Seed | Accuracy | Lift | ECE | Floor mean / p95 | Separable? | Adaptive ECE | Calibrator | Kept epoch | Val loss by epoch |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| 0 | 0.5920 | +0.0396 | 0.0149 | 0.0074 / 0.0100 | above floor | 0.0149 | temperature T=1.139 | 2 | 1.044, **0.987**, 1.036 |
| 1 | 0.5900 | +0.0364 | 0.0068 | 0.0071 / 0.0101 | **within floor** | 0.0076 | declined | 2 | 1.035, **0.928**, 0.998 |
| 2 | 0.5930 | **+0.0400** | 0.0100 | 0.0071 / 0.0101 | at p95 | 0.0133 | temperature T=0.894 | 2 | 1.058, **0.958**, 0.984 |
| 3 | 0.5866 | +0.0383 | 0.0374 | 0.0073 / 0.0103 | above floor | 0.0373 | declined | 3 | 1.051, 0.954, **0.947** |
| **median** | **0.5910** | **+0.0389** | 0.0124 | | | 0.0141 | | | |
| range | 0.5866–0.5930 | +0.0364 to +0.0400 | 0.0068–0.0374 | | | 0.0076–0.0373 | | | |

These are the calibrated numbers, which are what the gates read. The floor
is `gate_is_testable`'s simulation: a perfectly calibrated model's ECE on
the same 25,000 predictions, over 200 resamples. `gate_is_testable` passes
on every seed, with p95 0.0100–0.0103 against a limit of 0.025.

Per-question lift, calibrated, against each question's own marginal:

| Seed | `complexity` | `verbosity` | `helpfulness` | `correctness` | `coherence` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.0924 | +0.0518 | +0.0280 | +0.0254 | +0.0006 |
| 1 | +0.1038 | +0.0544 | +0.0166 | +0.0042 | +0.0032 |
| 2 | +0.0872 | +0.0492 | +0.0360 | +0.0284 | −0.0010 |
| 3 | +0.0824 | +0.0462 | +0.0348 | +0.0270 | +0.0012 |

Gates on every seed: `sample_size`, `gate_is_testable`, `workhorse_ece`,
`workhorse_adaptive_ece` and `worst_primitive_workhorse_ece` **pass**.
`accuracy_over_baseline` **fails**. `worst_question_over_baseline` also
**fails**, and it blocks now: this commit gates per question on a backbone
run, which it did not do when the 1.5B run was written up. `coherence` sits
on its marginal on every seed, at −0.0010 to +0.0032 against a limit of
+0.05.

## Against Qwen2.5-1.5B, same recipe, same seeds

| | 1.5B (`abe8888`, A10/A10G) | 7B (`0f6fb07`, A100 80GB) |
| --- | ---: | ---: |
| Lift, median (range) | +0.0268 (+0.0241 to +0.0288) | **+0.0389 (+0.0364 to +0.0400)** |
| Lift gain per seed, 0/1/2/3 | | +0.0155 / +0.0114 / +0.0113 / +0.0095 |
| Accuracy, median | 0.5778 | 0.5910 |
| ECE, median (range) | 0.0129 (0.0104–0.0179) | 0.0124 (0.0068–0.0374) |
| Adaptive ECE, median (range) | 0.0151 (0.0098–0.0242) | 0.0141 (0.0076–0.0373) |
| Brier, seeds 0/1/2/3 | 0.5489 / 0.5450 / 0.5454 / 0.5493 | 0.5279 / 0.5270 / 0.5320 / 0.5317 |
| Brier reduction over the marginal | +7.9% to +8.8% | **+11.1% to +11.6%** |
| `helpfulness` lift | −0.0054 to +0.0128 | **+0.0166 to +0.0360** |
| `correctness` lift | −0.0046 to +0.0068 | **+0.0042 to +0.0284** |
| `complexity` / `verbosity` lift | +0.081–0.099 / +0.035–0.044 | +0.082–0.104 / +0.046–0.054 |
| `coherence` lift | −0.0006 to +0.0008 | −0.0010 to +0.0032 |
| Kept epoch | 2, 2, 2, 2 | 2, 2, 2, 3 |

The Brier reduction is measured against the pooled marginal predictor that
each 7B report prints: the training split's per-question distribution,
add-one smoothed, scored on the evaluation labels. The seed fixes the
splits, and the 1.5B percentages in `reports/helpsteer2/README.md` come out
the same against these marginals (0.5962, 0.5964, 0.5983, 0.6006), so the
two columns are one comparison.

**What 7B buys is the quality questions.** On 1.5B, `helpfulness` and
`correctness` sat within ±0.013 of their marginals, and two seeds of four
were below them. On 7B, all four seeds are above the marginal on both,
`helpfulness` by +0.017 to +0.036. In the per-annotator ceiling
(`reports/helpsteer2/ceiling.md`), `helpfulness` and `correctness` are
the two questions the other annotators predict best, so this is where
signal was available to find. `coherence` still does not move, on either
model. Its half-panel lift is −0.10, which says its aggregated label is
mostly panel noise.

The pooled lift of +0.039 is well above the half-panel bound in
`ceiling.md` (−0.024). That bound is described there as loose, because
half-panels are noisier than the full panels behind the aggregated labels,
and this run is one measurement of how loose it is.

## Calibration

The ECE of three seeds is inside the range 1.5B reached. Seed 3 is the
outlier and the one to read.

- **Seed 1 is indistinguishable from perfectly calibrated.** Its ECE of
  0.0068 is below the noise floor's mean (0.0071), and the calibrator
  correctly declined.
- **Seeds 0 and 2 took a temperature, and in opposite directions.** Seed 0
  was overconfident and softened at T=1.139, moving its evaluation ECE from
  0.0465 to 0.0149. Seed 2 was underconfident and sharpened at T=0.894,
  moving it from 0.0286 to 0.0100. No run in `reports/helpsteer2/` ever
  kept a temperature: each took isotonic or declined.
- **Seed 3 declined on evidence that did not transfer.** On the
  calibration slice it scored 0.0230 unscaled against 0.0348 for the best
  fit, so it declined. On the 25,000 evaluation predictions the unscaled
  head reads ECE 0.0374 with overconfidence +0.037. That passes the 0.05 gate
  but is five times its floor. It is also the only seed that kept epoch 3,
  where validation loss was still falling: the most-trained head, and the
  most overconfident. The decline rule read a slice of 1,000 cases × 5
  questions, split between fitting and scoring. What this shows is that
  the slice can rank a head's calibration differently from the evaluation
  set. That is one seed, and it is not yet a finding about the decline
  rule.

## Throughput, memory and cost

| | |
| --- | ---: |
| Training throughput | 4.3 cases/s (A100 PCIe), 4.6 cases/s (A100 SXM4) |
| Epoch | 2,346–2,521 s for 10,800 cases |
| Training seconds per seed | 7,087–7,561 |
| Wall clock per seed, load + train + calibrate + evaluate | 8,114–8,454 s |
| Peak GPU memory in training | 28.7–28.9 GiB of 80 |
| Serving p50 / p99 on the evaluation set | 63–79 ms / 156–161 ms |

At 4.3–4.6 cases/s, 7B on an A100 trains at 1.5B's A10 rate, which was
2,538 s for epoch 1. The whole seed takes the same 2.3 h wall clock at five
times the parameters.

**Cost.** Seeds' `elapsed_s` sums to 32,814 s = 9.12 GPU-hours, and the
smoke run adds 1,382 s = 0.38 GPU-hours. At Modal's A100 80 GB price of
$0.000694/s ($2.50/h), that comes to **$22.77 for the four seeds and $0.96
for the smoke run, $23.73 in GPU time**. This is an estimate from the
recorded seconds, not the invoice. It excludes container start and image
pull, which fall outside `elapsed_s`, and the CPU and memory line items,
which at these durations add a few dollars at most. The approved budget
was $40–80.

## Smoke run — `qwen7b-smoke-n1000-e1-seed0`

One seed, 1,000 cases, 1 epoch, the same flags otherwise, run first to check
that the four shards load, train and fit in memory, and to put the 15 GB
download on the `trigon-weights` Volume once rather than four times. It
loaded, trained at 4.3 cases/s with a 27.4 GiB peak, calibrated (isotonic)
and gated, in 1,382 s. Its numbers (lift +0.0103) are a plumbing check, not a
measurement.

## Commands

```bash
# smoke
python scripts/modal_train.py launch --corpus helpsteer2 --seeds 0 -n 1000 --epochs 1 \
  --gpu A100-80GB --prefix qwen7b-smoke-n1000-e1 \
  --extra "--backbone qwen2.5-7b --lr 0.0001 --save-model model.pt"
python scripts/modal_train.py collect \
  helpsteer2-qwen7b-smoke-n1000-e1-0f6fb07994d8-20260927T001830 --out-dir reports/helpsteer2-7b

# the four seeds
python scripts/modal_train.py launch --corpus helpsteer2 --seeds 0,1,2,3 -n 12000 --epochs 3 \
  --calibration-n 1000 --gpu A100-80GB --prefix qwen7b-n12000-e3-lr1e-4 \
  --extra "--backbone qwen2.5-7b --lr 0.0001 --save-model model.pt"
python scripts/modal_train.py collect \
  helpsteer2-qwen7b-n12000-e3-lr1e-4-0f6fb07994d8-20260927T004223 --out-dir reports/helpsteer2-7b
```

Each seed ran:

```
python scripts/train_corpus.py helpsteer2 -n 12000 --calibration-n 1000 --eval-n 0 --epochs 3 --lr 0.0001 --accumulate 8 --d-model 128 --layers 2 --seed <s> --device cuda --backbone qwen2.5-7b --lora-rank 16 --max-batch-cells 50000000
```

The checkpoints are on the `trigon-runs` Volume under the run id, at
`seed<s>.pt`, and are not committed (`collect --models` fetches them).
