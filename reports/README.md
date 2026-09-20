# Runs

`docs/model-card.md` is the summary of what these runs add up to — what the
reference model can do, what it cannot, and what it is not suitable for.

> **Read every number here as one draw.** These are single-seed runs, and the
> 2,500-case configuration turns out to decide its own outcome by seed: seed 0
> never leaves chance while seeds 1, 2 and 3 beat this run's *final* loss by
> epoch 2, on one commit with identical flags. That is not a caveat about
> precision, it is the difference between certifying and not. `docs/decisions.md`
> has the numbers, and `scripts/seed_sweep.py` is how a configuration should be
> certified from now on. These runs are kept because the pipeline they exercise
> is real; their headline figures are not evidence that the configuration works.

Committed output from `trigon train` and the eval suites. These are results,
not fixtures: nothing in the test suite reads them, and they are here so a
reader can see what the gates actually said without running anything.

Every run is seeded — data generation, weight initialisation and shuffling — so
a report reproduces from the settings printed at the top of it rather than from
a checkpoint. No weights are committed.

| File | What it is |
| --- | --- |
| `reference-run.md` | the reference model: train, calibrate, gate |
| `reference-run-dotproduct.md` | the same run with dot-product option scoring — **failed**, and the reason is in `docs/decisions.md` |
| `ablations/dotproduct-*.md` | the three repair arms for that failure: `residual` fixes it, `normalize` does nothing, `both` is worse than `residual` alone |
| `served-model.md` | a smaller run (1,200 cases, 5 epochs) that **failed two gates** |

Sidecars beside each report carry the fitted temperatures and the loss curve in
machine-readable form — one set per report, named after it, so parallel runs
cannot overwrite each other's. (They did once: the dot-product arm lost its
sidecars to a fixed filename before the fix landed in `trigon.cli`. Both are
back.) `conformal/accounts.json` is a fitted conformal profile for the
certified model — target 90% coverage, 0.9227 achieved on a held-out split,
mean set 2.49 of 4 options.

Each report prints the exact command that produced it, and names the build it
measured: the version carries a hash of the weights, so the three runs are
`+50b2d6bd`, `+ff4c0f06` and `+87314780` rather than three different models
answering under one name. All three were regenerated from their own headers and
reproduced every gate verdict to the digit.

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

`reference-run.pt` passes all five gates. Served on three held-out accounts by
`scripts/served_answers.py`, so this block regenerates with the run — the
probabilities are the model's own, the truth column is what the generator
recorded:

```
plan=free, seats=93, tickets=5, payment_failed=False
  plan    -> free        conf=0.584  [free=0.854 pro=0.049]   truth: free        OK
  at_risk -> P(yes)=0.377                                     truth: no
  size    -> 1.45   conf=0.273                                truth: bronze

plan=standard, seats=142, tickets=12, payment_failed=False
  plan    -> pro         conf=0.102  [pro=0.322 enterprise=0.319]   truth: standard    X
  at_risk -> P(yes)=0.344                                           truth: no
  size    -> 1.49   conf=0.257                                      truth: silver

plan=enterprise, seats=24, tickets=2, payment_failed=False
  plan    -> pro         conf=0.103  [pro=0.322 enterprise=0.319]   truth: enterprise  X
  at_risk -> P(yes)=0.343                                           truth: yes
  size    -> 1.49   conf=0.257                                      truth: bronze
```

Two things to read here. **The confidence means something**: the one row it
gets right is the one it is confident about (free at 0.854, confidence 0.584),
and the two it gets wrong are the two where it is split almost evenly between
`pro` and `enterprise` and reports confidence 0.10. High confidence tracking
accuracy is the whole claim, and it is visible in three rows — including in the
third, where 0.10 is the model correctly declining to commit.

**And it only learned one of the three questions.** Measured over 300 held-out
accounts, against each question's own marginal predictor:

| Question | What it needs | Accuracy | Marginal predictor | Model output |
| --- | --- | ---: | ---: | --- |
| `plan` | copy a value from the state | 0.490 | 0.253 | uses the state |
| `at_risk` | a conjunction over two fields | 0.650 | 0.667 | 0.343–0.378, sd 0.016 |
| `size` | a threshold on a number | 0.237 | 0.257 | 1.448–1.491, sd 0.019 |

`plan` is a copy task and the model does it. On the other two it emits an
almost constant answer — `at_risk` never crosses 0.5, so it always says no, and
`size` is pinned near 1.47 whatever the seat count — and both land *below* the
marginal predictor. That is consistent with the project's own non-goals (no
arithmetic, no counting) and with a two-layer spike.

It is also exactly the shape `accuracy_over_baseline` exists to keep honest.
The gate is computed over all three questions pooled, so a model that answers
one of three and is worse than chance on the other two still clears it at
+0.0639. The gate is a floor against the fully degenerate case, not an accuracy
target, and this run is the reminder that clearing it is a low bar: the real
accuracy bar is the workflow suite.

## The dot-product repair arms

`ablations/` holds the three arms that answered why the dot-product head failed.
Same data, seeds and hyperparameters as `reference-run-dotproduct.md`; held-out
accuracy per question, against that question's own marginal predictor:

| Arm | `plan` (0.253) | `at_risk` (0.667) | `size` (0.257) | Pooled lift | Gates |
| --- | ---: | ---: | ---: | ---: | --- |
| Readout slot per option | 0.457 | 0.662 | 0.254 | +0.0673 | 5/5 |
| Dot product, as first shipped | 0.254 | 0.662 | 0.254 | −0.0002 | blocked |
| `dotproduct-normalize` | 0.254 | 0.662 | 0.254 | −0.0002 | blocked |
| `dotproduct-residual` | **0.847** | 0.662 | 0.251 | **+0.1962** | 5/5 |
| `dotproduct-both` | 0.457 | 0.662 | 0.254 | +0.0673 | 5/5 |

`match_residual` — adding each option's own input embedding to its key — is now
the default, and it makes the one-slot head beat the *n*-slot head on the one
question either of them learns. `match_normalize` does nothing alone and
cancels most of the gain when combined; the mechanism and the hypothesis that
survives are in `docs/decisions.md`.

Read the flat columns honestly: **`at_risk` and `size` are unlearned in every
arm**, sitting at their marginals. No arm passes the per-question gate, which
is why that gate is advisory. This fixes a head, not a model.
