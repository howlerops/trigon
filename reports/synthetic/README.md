# The synthetic suite on Qwen2.5-1.5B — `size` is learned, on every seed

The generator's three questions, the suite every earlier certification ran on,
with one thing changed from the reference spike: the model. 8,000 training
cases, noise 0.2, 8 epochs (best-epoch selection kept epoch 3 on every seed),
lr 1e-4, LoRA rank 16 over Qwen2.5-1.5B at revision `8faed76`. 6,000 evaluation
cases. Commit `d9eb143`, clean tree, four seeds on `NVIDIA A10`, ~57 minutes a
seed (`qwen15b-synthetic-e8-modal-run.json`).

| Seed | Accuracy | Lift | `plan` | `at_risk` | `size` | ECE | Adaptive ECE | int8 Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8361 | +0.4438 | +0.5952 | +0.1423 | **+0.5940** | 0.0408 | 0.0444 | 0.0009 |
| 1 | 0.8299 | +0.4359 | +0.5888 | +0.1268 | **+0.5920** | 0.0307 | 0.0324 | 0.0002 |
| 2 | 0.8252 | +0.4337 | +0.5940 | +0.1360 | **+0.5712** | 0.0114 | 0.0127 | 0.0002 |
| 3 | 0.8249 | +0.4309 | +0.5712 | +0.1355 | **+0.5862** | 0.0106 | 0.0190 | 0.0008 |

**All three questions are far above their marginals on all four seeds.**
Per-question lift is the calibrated run's, against each question's own
marginal predictor.

**Three seeds of four certify under the gates as they now stand.** When these
ran, every seed cleared every blocking gate. On 2026-09-25 the per-question
and per-primitive gates became blocking for backbone runs, as Q16 of the
sign-off promised once a real backbone landed (`docs/evals.md`). Seed 0's
Score head sits at ECE **0.0551** against the 0.05 limit, and its pooled
0.0408 hid that. So seed 0 no longer certifies. The configuration still
certifies on its median: the worst-primitive ECE is 0.0254–0.0551, median
0.0303, and every other gate passes on all four seeds.

## What it settles

**`size` was the project's longest-running negative result.** A threshold
over 500 values; below its own marginal on every seed of seven interventions
and twenty-two runs of the spike -- digit tokenization, capacity, a readout
slot per level, the Score head, the residual -- and the investigation was
closed with one explanation left standing: every one of those runs shared the
128-wide, two-layer backbone (`reports/perlevel/README.md`). Here it is learned
at +0.57 to +0.59 on all four seeds, level with `plan`, which the spike
learned to Bayes-optimal. The model was the bottleneck.

**`at_risk` moves from three seeds of four at +0.09 to four of four at
+0.13–0.14.**

**It is at the noise floor.** The generator's Bayes-optimal loss is 0.5585;
the best validation loss per seed is 0.5556–0.6037, seed 1 at the floor.
Every seed overfit after epoch 3 -- validation rises to 0.71–0.85 by epoch 8
-- and best-epoch selection kept epoch 3, so eight epochs cost compute, not
quality.

**Calibration held, and the int8 twin moves ECE by at most 0.0009** against a
0.01 gate, now on a real backbone rather than the spike. Seed 0 is the
weakest calibrated (ECE 0.0408, and its Score head 0.0551 -- the one
per-primitive failure, blocking now); the other three are 0.0106–0.0307.

`docs/next.md` A.3's done-condition -- four seeds certify with all three
synthetic questions above their marginals, and per-corpus ECE holds on real
data (`reports/banking77/`, `reports/helpsteer2/`) -- is met.
