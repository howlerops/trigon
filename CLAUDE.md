# Working in this repo

## What this is

Phase 0 of the build plan in `docs/roadmap.md`: the contract, the calibration
layer, the serving pipeline and the eval harness for a prefill-only, typed,
calibrated decision model. No trained weights yet.

## Ground rules

**Calibration is the product.** ECE numbers, reliability diagrams and the eval
harness are the differentiator, not the architecture. Never trade a calibration
property for an accuracy number without measuring both, and never relax a
release gate to make a run pass.

**Claims are tested, not asserted.** The two architectural claims — per-question
independence and schema-prefix cacheability — are asserted in
`tests/test_independence.py` against the reference model, to exact equality. If
you change the layout, the mask or the position scheme, those tests are the
specification.

**One source of truth per fact.** Budgets live in `src/trigon/limits.py`. The
pipeline's order of operations lives in `src/trigon/engine.py`. The contract
lives in `src/trigon/types.py` and is exported to `spec/openapi.json`. Do not
duplicate any of them.

**Honest defaults.** An uncalibrated deployment is allowed; a silently
uncalibrated one is not (`/healthz` reports it). A degenerate temperature fit
warns rather than returning a quiet number. Benchmarks that measure explicit
non-goals are marked `non_goal = True` and published anyway.

## Layout

The core package is dependency-light on purpose — `pydantic` only. The
calibration math, the metrics and the schema compiler are pure Python so they
can be imported by the gateway, by training code and by CI without a GPU stack.
`torch` lives behind the `train` extra and is imported lazily.

## Commands

```bash
pip install -e ".[dev,server]"     # add "train" for the reference model
pytest -q
ruff check src tests scripts
trigon eval all -n 200             # exits non-zero on a failed gate
python scripts/export_openapi.py   # after ANY change to the contract
```

`tests/test_openapi_drift.py` fails if the checked-in spec and the gateway
disagree. Both SDKs are generated from that spec, so regenerate it in the same
commit as the change.

## When adding a backend

Implement `trigon.backends.base.Backend`. Return **logits**, not probabilities —
temperature scaling happens above you, and a backend that can only return
probabilities cannot be calibrated post hoc. Return one entry per compiled
question, in declared label order. `validate_output` runs on every path and will
reject a wrong count, a wrong head or a non-finite value.

If your backend has an exact tokenizer, expose it as `.estimator` and build the
compiler with it (`make_compiler()`), so compiled token counts match the tensors.

## When adding a benchmark

Subclass `trigon.evals.jaggedness.Benchmark`, name the documented failure mode
it measures, and **run the lexical floor against it before believing it**. Three
benchmarks in the first draft could be aced by surface statistics — label-
correlated length, keyword leakage — and a benchmark the floor aces measures
nothing.

If the benchmark is paired (two variants of the same case, scored on the
difference), override `score`. If it measures something we deliberately do not
target, set `non_goal = True`.
