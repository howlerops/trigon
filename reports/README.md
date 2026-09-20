# Runs

Committed output from `trigon train` and the eval suites. These are results,
not fixtures: nothing in the test suite reads them, and they are here so a
reader can see what the gates actually said without running anything.

Every run is seeded — data generation, weight initialisation and shuffling — so
a report reproduces from the settings printed at the top of it rather than from
a checkpoint. No weights are committed.

| File | What it is |
| --- | --- |
| `reference-run.md` | the reference model: train, calibrate, gate |
| `reference-run-dotproduct.md` | the same run with dot-product option scoring — the phase-1 ablation |
| `served-model.md` | a smaller run (1,200 cases, 5 epochs) that **failed two gates** |

Sidecars beside each report carry the fitted temperatures and the loss curve in
machine-readable form. The dot-product arm has none: the two runs were launched
in parallel before sidecar names were derived from the report name, so they
collided and the last writer won. The loss curve and gates are in its markdown
report; the fix is in `trigon.cli`.

The headline is not the model's quality — it is a spike, and its accuracy says
so. It is that the loop closes and the gates bite, twice over:

- `reference-run.md` passed every calibration gate at **45.6%** accuracy, which
  is what added `accuracy_over_baseline` in the first place.
- `reference-run-dotproduct.md` scored **38.9%** — below the 39.2% marginal
  predictor, so literally worse than ignoring the input — and still passed
  every ECE gate, adaptive included. It fails on exactly one line.

- `served-model.md` is a deliberately undertrained run, kept because it is the
  clearest picture of what the gates are for. It closed 3% of the gap to Bayes,
  scored **0.3914** against the 0.3922 marginal predictor, and served this when
  asked which plan an account was on:

  ```
  plan   free   confidence=0.000
         {"free": 0.254, "standard": 0.246, "pro": 0.253, "enterprise": 0.246}
  ```

  A near-uniform distribution, and the wrong answer — at ECE 0.0171, inside the
  0.05 limit. It fails `accuracy_over_baseline` and adaptive ECE, and
  `trigon train` exits non-zero, so CI blocks it.

Under ECE-only gates, all three would have shipped. See `docs/evals.md` §4.

## What the certified model actually answers

`reference-run.pt` passes all five gates. Served on three accounts — the
probabilities are the model's own, the truth column is what the generator
recorded:

```
plan=pro, seats=340, tickets=5, payment_failed=True
  plan    -> pro    conf=0.105  [pro=0.323 enterprise=0.320]   truth: pro          OK
  at_risk -> P(yes)=0.348                                      truth: yes
  size    -> 1.50   conf=0.256                                 truth: gold

plan=free, seats=12, tickets=0, payment_failed=False
  plan    -> free   conf=0.600  [free=0.862 pro=0.047]         truth: free         OK
  at_risk -> P(yes)=0.382                                      truth: no
  size    -> 1.48   conf=0.262                                 truth: bronze

plan=enterprise, seats=480, tickets=7, payment_failed=True
  plan    -> pro    conf=0.103  [pro=0.322 enterprise=0.319]   truth: enterprise   X
  at_risk -> P(yes)=0.348                                      truth: yes
  size    -> 1.50   conf=0.256                                 truth: platinum
```

Two things to read here. **The confidence means something**: where the model is
confident it is right (free, 0.862, confidence 0.600), and where it is split
almost evenly between two options it reports confidence 0.105 — and that is the
one it gets wrong. High confidence tracking accuracy is the whole claim, and it
is visible in three rows.

**And it only learned one of the three questions.** `plan` is a copy task: the
answer appears verbatim in the state. `at_risk` is a conjunction and `size` is a
threshold on a number, and the model answers both with a constant regardless of
input — 0.348 and 1.50 every time. That is consistent with the project's own
non-goals (no arithmetic, no counting) and with a two-layer spike, and it is
exactly the shape `accuracy_over_baseline` exists to keep honest: the model
clears the gate on the strength of one question out of three.
