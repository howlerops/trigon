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

Sidecars beside each report carry the fitted temperatures and the loss curve in
machine-readable form.

The headline is not the model's quality — it is a spike, and its accuracy says
so. It is that the loop closes and the gates bite, twice over:

- `reference-run.md` passed every calibration gate at **45.6%** accuracy, which
  is what added `accuracy_over_baseline` in the first place.
- `reference-run-dotproduct.md` scored **38.9%** — below the 39.2% marginal
  predictor, so literally worse than ignoring the input — and still passed
  every ECE gate, adaptive included. It fails on exactly one line.

Under ECE-only gates, both would have shipped. See `docs/evals.md` §4.
