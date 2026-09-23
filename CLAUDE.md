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

The KV prefix is the same class of thing and **is** cached now, behind a flag.
`SchemaPrefix` stores each layer's pre-attention normed states plus the schema
block's outputs, keyed on `schema_hash`; `Usage.cached_schema_tokens` reports a
hit rather than always 0.

**It is on by default on the gateway, now that it has been timed.** At the
shape the certified Banking77 model serves — 77 options, 707 of 725 tokens
schema — it is a 6× speedup, 91.8 ms to 15.3 ms p50; at 256 options, 23×.
`reports/cache/README.md` has the table and the hardware.

It was off for two reasons and both are spent. The first, that the cache
traded exact independence for compute, is wrong: on GitHub's runners the
*cached* path is exact and the uncached one drifts 2.4e-08, because what
decides it is whether a sequence length lands on a kernel that reduces in the
same order. Neither path leaks; neither is bit-reproducible across shapes. The
second, that nobody had timed it, was honest and is no longer true.
`TRIGON_CACHE_PREFIXES=0` turns it off and `/healthz` reports which way it is
set.

Never enable it during training: a prefix belongs to the weights that produced
it, weights move every step, and nothing raises because the shapes all match.

**Claims are tested, not asserted.** The two architectural claims — per-question
independence and schema-prefix cacheability — are asserted in
`tests/test_independence.py` against the reference model. If you change the
layout, the mask or the position scheme, those tests are the specification.

**Exact where the shapes match, a float32 bound where they do not.** Comparing
one question's answer with and without twenty others compares two different
sequence lengths, a different length can select a different GEMM kernel, and a
different reduction order rounds differently: those tests read exactly 0.0
here and 1.4e-08 on GitHub's runners. The mask is what guarantees
independence, and that is exact and provable from the layout; the arithmetic's
reproduction of it across shapes is not portable. Comparisons at a fixed
shape — the schema prefix against a different state, the dot-product head's
option keys — stay exact, because there the same kernel reduces the same way
anywhere.

**One source of truth per fact.** Budgets live in `src/trigon/limits.py`. The
pipeline's order of operations lives in `src/trigon/engine.py`. The contract
lives in `src/trigon/types.py` and is exported to `spec/openapi.json`. Do not
duplicate any of them.

Where the prose has to restate one anyway — a decisions document that says
"see `limits.py`" decides nothing — the restatement is pinned by a test.
`tests/test_openapi_drift.py` does it for the contract and
`tests/test_docs_drift.py` for the budgets and the gate limits, so a gate
relaxed in code and left alone in the docs fails the build instead of
certifying a threshold nothing enforces. If you add a number to the docs that
the code also holds, add it there too.

**Honest defaults.** An uncalibrated deployment is allowed; a silently
uncalibrated one is not (`/healthz` reports it). A degenerate temperature fit
warns rather than returning a quiet number. Benchmarks that measure explicit
non-goals are marked `non_goal = True` and published anyway.

**A calibrator is a proposal, not a result.** There are two of them — a
temperature and an isotonic map — and neither wins everywhere: a uniformly
overconfident head goes from ECE 0.2135 to 0.0247 under a temperature, and a
*tilted* head (overconfident where it is confident, under where it is not) has
no correct temperature at all and takes isotonic from 0.1232 to 0.0227.
`trigon train` and `trigon fit` therefore fit both per primitive, score them on
a slice of the calibration split neither was fitted on, and apply whichever
demonstrably helps — or neither. Do not add a third candidate without scoring
it the same way: `scripts/decline_rule.py` and `scripts/calibrator_choice.py`
construct heads whose true calibration is known, which a seed sweep cannot do.

Score candidates on `max(ECE, adaptive ECE)`, never on ECE alone. The two
estimators disagree — on one Noul head they read 0.0251 and 0.1625 over the
same answers — and the run is gated on both, so optimising the blinder one
declines the calibrator the gate is about to fail you for.

## Keeping the ledger

`docs/ledger.md` is the running record: what has been built, what has been
measured, what was believed and turned out to be wrong, and what is still open.
**Update it at every milestone**, in the same commit as the work.

A milestone is any of these, and nothing smaller:

- a capability that did not exist now works end to end;
- a claim moves between **believed** and **measured** in either direction;
- a release gate is added, removed, or changes what it reads;
- a published number is corrected;
- something in **Open** closes, or a new one opens.

Refactors, tests for existing behaviour and documentation passes are not
milestones. If every commit touches the ledger it stops being read.

**The disproved section is the point.** A ledger of successes is a changelog,
and this project's most expensive hours have gone into ideas that were
well-argued and wrong — the Rust rewrite, the CLIP-style cosine, four
successive theories of why one question would not train. Writing down what was
believed *and what measurement said instead* is what stops the fifth theory
being proposed with the same confidence as the first. When you disprove
something, add the row before you fix the cause; the fix is easier to describe
than the belief, so the belief is what gets lost.

**Record errors that ran in your favour separately**, under *Corrected in our
own favour*. They are the ones nobody else will find: a measurement that
flatters the project gets repeated rather than checked, and this repository has
shipped four of them — a cost figure that charged the test client to the
gateway, a price comparison that cached only our side, the same comparison
counting two different tokenizers, and a pooled ECE quoted as if it described
the model.

The counts in *State of the repository* are pinned by
`tests/test_ledger_drift.py` with deliberate tolerances, so the build fails when
the ledger has drifted far enough to mislead rather than on every commit. The
gate count and the use-case count are exact, because a reader checking what this
project enforces is entitled to a number that is not approximately true.

## Layout

The core package is dependency-light on purpose — `pydantic` only. The
calibration math, the metrics and the schema compiler are pure Python so they
can be imported by the gateway, by training code and by CI without a GPU stack.
`torch` lives behind the `train` extra and is imported lazily.

## Commands

```bash
pip install -e ".[dev,server]"     # add "train" for the reference model, "gpu" for Modal
pytest -q
ruff check src tests scripts
trigon eval all -n 200             # exits non-zero on a failed gate
python scripts/price.py            # cost per decision, per use case
trigon train --out reports/run.md --save-model reports/run.pt   # train, calibrate, gate
python scripts/seed_sweep.py --seeds 0 1 2 3 -n 8000 --epochs 8  # certify on the spread
python scripts/regate.py reports/run.pt --out reports/          # recalibrate, no retrain
trigon ask request.json --backend torch --weights reports/run.pt # one request
trigon serve --backend torch --weights reports/run.pt            # behind the API
trigon serve --compat --weights reports/run.pt   # the incumbent's shapes at the root
python scripts/train_corpus.py banking77 --out reports/banking77/run.md  # real data
python scripts/migrate.py traffic.jsonl --incumbent https://api.example.com
python scripts/export_openapi.py   # after ANY change to the contract
modal run scripts/modal_train.py --corpus helpsteer2 --n 12000 --epochs 6  # on a GPU
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

## When adding a corpus

**The licence tier is enforced in code, not in a document.**
`trigon.evals.corpora` refuses the uses a tier forbids — amber evals and never
trains, red ships in nothing — and `load()` takes `purpose` with no default,
because a default is the argument a caller least often thinks about and an
amber corpus in a training mix is not a mistake you can find later by reading
the weights. A test pins the committed tiers against `docs/data.md`.

Add a `CorpusSpec` with the attribution its licence requires. It is printed
into every report: a credit that lives only in a docs table is not attached to
the number it belongs to. Nothing is committed — corpora are fetched to an
ignored cache, because a checked-in copy of someone else's data is a second
source of truth that goes stale silently.

**Shuffle every split you slice.** Several of these corpora are ordered by
label. Taking `evaluation[:n]` off Banking77's test split produced an
evaluation set of one intent, a marginal predictor of 1.0000 and an
`accuracy_over_baseline` of −1.0000, and no gate can catch that — the model
really did lose to that baseline.

Loaders stay plain-file and stdlib. The `corpora` module imports without
`torch` for the same reason the compiler and the calibration math do.

## When adding a tokenizer

**Measure what it buys before believing it.** Digits were split one per token
on a well-evidenced diagnosis — whole-number BPE merges made 127 and 128
unrelated embedding rows, and the corpus contains a control showing a threshold
over 13 values is learned where one over 500 is not. It did not work: the
question it was for did not move, and two seeds that had certified stopped.
A tokenizer change alters every embedding the model has, so it can regress
questions that have nothing to do with the one it was aimed at.

**Python or Rust, chosen on measured performance and accuracy** — decided
2026-09-23 for A.3, replacing "encoding is pure Python and stays that way".
A pretrained backbone ships its own tokenizer, and an encoder that differs
from it by one token on one input feeds the model a sequence it was never
trained on; exactness against the reference is not negotiable, the language
is. What the old rule protected is still a cost to weigh, not waive: the
gateway budgets schemas on CPU nodes with no weights, and the compiler, the
calibration math and the drift test import without `torch`. A compiled
tokenizer is fine where it earns its place — keep it an optional dependency
behind the backend that needs it, and keep the import path of the compiler
and the calibration math free of it. Training a vocabulary lives in
`scripts/`.

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
