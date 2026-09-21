# Seed sweeps

One configuration, several seeds, reported as a spread. Produced by
`scripts/seed_sweep.py`; see `docs/decisions.md`, "A single-seed training run is
not evidence", for why this exists.

## `reference-config` — 1,200 cases, 3 epochs, d_model 128, 2 layers

```bash
python scripts/seed_sweep.py --seeds 0 1 2 3 --name reference-config \
  -n 1200 --epochs 3 --lr 0.01 --d-model 128 --layers 2 --noise 0.2 \
  --option-scoring auto --eval-n 2000 --floor-trials 20 --validation-fraction 0
```

| Seed | Final loss | Lift over baseline | ECE | Certified | Blocked on |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1.1399 | +0.0000 | 0.0151 | **no** | `accuracy_over_baseline` |
| 1 | 0.9689 | **+0.1213** | 0.0122 | **no** | `workhorse_adaptive_ece` |
| 2 | 1.0692 | +0.0517 | 0.0095 | **no** | `workhorse_adaptive_ece` |
| 3 | 1.0438 | +0.0510 | 0.0223 | **no** | `workhorse_adaptive_ece` |

Final loss: median 1.0565, range 0.9689–1.1399, **spread 0.1710**. Chance is
1.1552.

**Read the lift column.** It is the gate's own headline metric, its limit is
0.05, and across four seeds of one configuration it takes the values 0.0000,
0.1213, 0.0517 and 0.0510 — from "does not beat a model that ignores the state"
to "beats it by more than twice the limit". Seed 0 fails that gate outright;
the other three clear it. Reporting any one of these as *the* result of this
configuration is reporting a draw.

They all fail to certify, on two different gates, which is its own useful
signal: this configuration is not a candidate, and a single run of it that
happened to pass would not have made it one.

The `.pt` files are deleted after each seed — a sweep is about the distribution,
not about keeping four models.

## `candidate-8k` — 8,000 cases, 8 epochs, d_model 128, 2 layers

```bash
python scripts/seed_sweep.py --seeds 0 1 2 3 --jobs 4 --name candidate-8k \
  -n 8000 --epochs 8 --lr 0.01 --d-model 128 --layers 2 --noise 0.2 \
  --option-scoring auto --eval-n 6000 --floor-trials 40
```

| Seed | Final loss | Kept epoch | Lift over baseline | ECE | Certified | Blocked on | Advisory |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | 1.0457 | 1 | +0.1614 | 0.0102 | **yes** | — | `worst_question_over_baseline` |
| 1 | 1.0016 | 1 | +0.1926 | 0.0677 | no | `workhorse_ece`, `workhorse_adaptive_ece` | `worst_question_over_baseline` |
| 2 | 0.8341 | 7 | +0.2279 | 0.0067 | **yes** | — | `worst_question_over_baseline` |
| 3 | 0.8265 | 6 | +0.2141 | 0.0190 | **yes** | — | `worst_question_over_baseline` |

Final loss: median 0.9178, range 0.8265–1.0457, **spread 0.2192**.
**Certified: 3 of 4 seeds.**

**More data fixed the accuracy coin flip and did not fix the calibration
one.** That is the whole result, and it is two findings, not one.

The accuracy gate is no longer a draw. Across four seeds the lift is +0.1614,
+0.1926, +0.2279 and +0.2141 — every seed clears the 0.05 limit by three to
four times, and the narrowest of them is more than double the committed
reference run's +0.0639. Compare the 1,200-case sweep above, where the same
gate took the values 0.0000, 0.1213, 0.0517 and 0.0510 and *failed outright on
seed 0*. At 1,200 cases whether this configuration uses its input at all
depended on the seed. At 8,000 it does not.

Calibration still does. Seed 1 scores ECE 0.0677 against a 0.05 limit while
the other three score 0.0067–0.0190 — a spread of 0.0610 on a gate whose limit
is 0.05, which is a gate deciding on the draw. So this configuration is **not
certifiable** under `--require all`, and a single run of it that happened to
draw seed 2 would report 0.0067 as the result of the configuration when the
range is ten times that.

Note also the kept-epoch column. Seeds 0 and 1 kept **epoch 1 of 8** — their
held-out loss never improved again — while seeds 2 and 3 kept epochs 7 and 6
and ended at a final loss 0.2 lower. Those same two early-stopping seeds have
the two lowest lifts, and the one seed that fails a blocking gate is one of
them. That is not enough to call stalling at epoch 1 the cause of the failure:
seed 0 stalled at epoch 1 and certified comfortably. What it does say is that
the same configuration takes two distinguishable trajectories on the same
data, which is the attractor the dot-product head fell into, visible in a
different instrument.

`worst_question_over_baseline` fails on every seed, as it does on every
configuration tried so far, which is why it is advisory. It is much closer to
passing than before: −0.0023 to −0.0230 here against −0.0400 on the committed
reference run. The model is learning the other two questions a little, rather
than sitting exactly on their marginals.

**This sweep also found a bug in the sweep.** It first reported *0 of 4 seeds
certified*. `scripts/seed_sweep.py` re-derived each verdict from the gate list
by filtering on an `advisory` key that `render_json` did not emit, so every
advisory failure counted as blocking. The flag is emitted now, the sweep takes
the verdict from the report's own `passed` field rather than recomputing one,
and it raises if the two disagree. A tool for measuring whether a result is
real is the last place a silent wrong answer belongs.

## `calibsplit-8k` — the same configuration, with the temperature fitted properly

Identical to `candidate-8k` in every respect but one: the temperature is
fitted on a third split the model never trained on
(`docs/decisions.md`, "The temperature was fitted on the split the model
trained on"). Same seeds, same data, same weights — the *uncalibrated* column
is bit-identical across the two sweeps, which is how we know only the
temperature moved.

```bash
python scripts/seed_sweep.py --seeds 0 1 2 3 --jobs 4 --name calibsplit-8k \
  -n 8000 --epochs 8 --lr 0.01 --d-model 128 --layers 2 --noise 0.2 \
  --option-scoring auto --eval-n 6000 --floor-trials 40
```

| Seed | Uncalibrated | Fit on **train** | Fit on **held-out** | Certified |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.0303 | 0.0102 | 0.0128 | **yes** |
| 1 | 0.0431 | 0.0677 ❌ | 0.0516 ❌ | no |
| 2 | 0.0096 | 0.0067 | 0.0083 | **yes** |
| 3 | 0.0121 | 0.0190 | 0.0157 | **yes** |

**Certified: 3 of 4 seeds, unchanged.** The fix is real and it is not enough,
and both halves of that are worth stating.

**What it bought.** The worst seed improved from 0.0677 to 0.0516, and its
adaptive-ECE failure cleared entirely (0.0682 → 0.0498, inside the gate). The
range across seeds narrowed from 0.0610 to 0.0433 — a 29% reduction in exactly
the quantity that makes a gate a coin flip.

**What it did not buy, and this is the part that matters.** Temperature
scaling *still* raises ECE on seeds 1 and 3, relative to not scaling at all.
Fitting on data the model had memorised was a real defect with a measurable
cost, and it was not the whole cause. A diagnosis that explains part of an
effect and gets promoted to the explanation is how the dot-product head
absorbed three wrong hypotheses before instrumentation found the real one.

The fitted temperatures point somewhere specific. Per primitive, across the
four seeds:

| Seed | Choice | Noul | **Score** |
| ---: | ---: | ---: | ---: |
| 0 | 0.868 | 1.418 | **3.301** |
| 1 | 0.949 | 0.706 | **0.201** |
| 2 | 1.012 | 0.990 | **0.460** |
| 3 | 0.995 | 1.045 | **0.346** |

Choice and Noul sit near 1.0 on every seed. Score ranges over a factor of
sixteen, and on three seeds of four it is below 0.5 — a temperature below 1
*sharpens*, so the fit is making a head more confident. `size`, the Score
question, is the one the model never learns: 0.2447 accuracy against a 0.2558
marginal on seed 1. Sharpening a head that carries no signal is how you
manufacture confident wrong answers, which is precisely what ECE measures.

`temperature.py` already warns when a fit pins at the ceiling (20.0, flattening
to uniform). It has a floor warning too, at 0.05 — and 0.20 does not trip it.
Whether the floor is in the wrong place, or the Score head needs a different
treatment, is the open question. The per-primitive calibration table added
alongside this sweep is the instrument for answering it; the pooled number
could only say that something was wrong.
