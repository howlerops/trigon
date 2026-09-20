# Working in this repo

## What this is

Phase 0 of the build plan in `docs/roadmap.md`: the contract, the calibration
layer, the serving pipeline and the eval harness for a prefill-only, typed,
calibrated decision model.

The loop closes: `trigon train` fits the reference model, calibrates it, runs
the gates and writes a servable checkpoint, which `trigon ask --weights` and
`trigon serve --weights` both load — and `tests/test_server.py` asserts the two
return the same probabilities, because for a while only one of them worked.
Three runs are committed under `reports/`. The reference model is a
spike — 128-wide, two layers, a hashing tokenizer — so its accuracy is not a
result; that the gates are exercised by a model rather than asserted about one
is.

## Ground rules

**Calibration is the product.** ECE numbers, reliability diagrams and the eval
harness are the differentiator, not the architecture. Never trade a calibration
property for an accuracy number without measuring both, and never relax a
release gate to make a run pass.

**A number is not evidence until the floor is under it.** Every published ECE
carries a simulated noise floor (`metrics.noise_floor`) and a verdict on
whether the two are separable. Never quote an ECE from a run below
`MIN_CALIBRATION_SAMPLES`, and never widen `MAX_FLOOR_FRACTION_OF_GATE` to make
`gate_is_testable` pass — if the gate is not a test, fix the run, not the gate.

**That rule applies to the training run too, and for a long time it did not.**
The reference configuration turned out to decide its own outcome by seed: one
draw never leaves chance, three others beat the committed run's final loss by
epoch 2, on one commit with identical flags. A single-seed run is one sample
from a distribution nobody measured. `scripts/seed_sweep.py` measures the
spread; certify a configuration on its median and range, never on its best
draw, and prefer a narrow spread to a good maximum.

**Calibration never certifies alone.** A model that reports each question's
marginal distribution is calibrated by construction and useless, and it passes
every ECE gate — the first trained reference model did exactly that at 46%
accuracy. `accuracy_over_baseline` is what rejects it. Any new gate set must
keep a term that fails a model which ignores its input.

**Anything derived only from the schema is derived once.** The attention mask
and the BM25 option index are cached on schema-shaped keys. When you add
something in that class, cache it the same way and key it on what it actually
depends on.

The KV prefix is the same class of thing and is *not* cached here — the layout
makes it cacheable, and `tests/test_independence.py` asserts that the schema
encodes identically regardless of state, but this repo holds no KV cache. The
cache is the phase-3 serving stack, and `Usage.cached_schema_tokens` therefore
reports 0 on every response and says so in the spec.

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
trigon train --out reports/run.md --save-model reports/run.pt   # train, calibrate, gate
trigon ask request.json --backend torch --weights reports/run.pt # one request
trigon serve --backend torch --weights reports/run.pt            # behind the API
python scripts/export_openapi.py   # after ANY change to the contract
```

**Four artifacts are generated, not maintained.** Change the source and
regenerate in the same commit — CI fails otherwise, which is the point:

| Artifact | From | Regenerate with |
| --- | --- | --- |
| `spec/openapi.json` | the gateway | `python scripts/export_openapi.py` |
| `sdk/python/trigon_client/_generated.py` | that spec | `python scripts/generate_sdk.py` |
| `src/trigon/data/bpe.json` | the eval corpus | `python scripts/train_tokenizer.py` |
| the cost tables in `docs/architecture.md` | the compiler and `limits.py` | `python scripts/attention_table.py --write` |

The vocabulary is the one that does not regenerate casually: it changes the
embedding table, so every existing checkpoint is trained against the old one.
Checkpoints record `{kind, vocab_size}` and are rebuilt with the tokenizer they
were trained on, which is why the certified run still loads.

## When adding a tokenizer

Encoding is pure Python and stays that way. The gateway budgets schemas on CPU
nodes with no weights, and the compiler, the calibration math and the drift
test all import without `torch` — a tokenizer needing a Rust extension to count
tokens pulls that dependency into all of them. Training a vocabulary is a
different matter and lives in `scripts/`.

Expose `encode`, `count`, `exact` and `kind`, and add `kind` to
`backends.tokenizer.build_tokenizer` so a checkpoint can name you. A model
served under a vocabulary it was not trained on reads every id as a different
word and nothing raises.

## When adding a backend

Implement `trigon.backends.base.Backend`. Return **logits**, not probabilities —
temperature scaling happens above you, and a backend that can only return
probabilities cannot be calibrated post hoc. Return one entry per compiled
question, in declared label order. `validate_output` runs on every path and will
reject a wrong count, a wrong head or a non-finite value.

If your backend has an exact tokenizer, expose it as `.estimator`. `Engine` picks
it up (`backends.base.estimator_of`) and compiles with it, so compiled token
counts match the tensors. That wiring is the engine's job rather than each call
site's because it was not: the gateway compiled with the character heuristic
while the torch backend built tensors from its own tokenizer, and every torch
request over HTTP died with a 500 while 190 tests stayed green.

Name the build after its weights. `model_version` is the only build identifier
that reaches the caller, so a randomly initialised model must not answer under
the same name as a trained one — `TorchReadoutBackend.stamp_version()` is called
when training finishes and when a checkpoint is written.

## When adding a benchmark

Subclass `trigon.evals.jaggedness.Benchmark`, name the documented failure mode
it measures, and **run the lexical floor against it before believing it**. Three
benchmarks in the first draft could be aced by surface statistics — label-
correlated length, keyword leakage — and a benchmark the floor aces measures
nothing.

If the benchmark is paired (two variants of the same case, scored on the
difference), override `score`. If it measures something we deliberately do not
target, set `non_goal = True`.
