# Trigon

An open-source **System One**: typed, calibrated decisions in a single
prefill-only forward pass.

Send state plus a map of typed questions. Get one typed answer per question,
each a full distribution over exactly the labels you declared — in one pass, no
decode loop, no sampler, no parsing.

```python
from trigon import Engine, SystemOneRequest
from trigon.backends import LexicalBackend

response = Engine(LexicalBackend()).answer(
    SystemOneRequest(
        state="The customer writes: my card payment was declined at the store.",
        questions={
            "intent": {
                "type": "choice",
                "instructions": "Route this ticket.",
                "options": [
                    {"name": "card_declined", "criteria": "a card transaction was refused"},
                    {"name": "lost_luggage", "criteria": "baggage missing after a flight"},
                ],
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this for the customer?",
                "levels": [
                    {"name": "low", "value": 1.0},
                    {"name": "high", "value": 5.0},
                ],
            },
            "urgent": {"type": "noul", "instructions": "Needs a human within the hour?"},
        },
    )
)

response.answers["intent"].selected        # "card_declined"
response.answers["intent"].probabilities   # every declared option, summing to 1
response.answers["intent"].confidence      # derived from the calibrated distribution
response.answers["severity"].score         # expectation over the declared levels
response.answers["urgent"].probability     # no confidence field, by design
```

## Why

Most typed-output stacks bolt a JSON schema onto a generative model and
validate the result. This one has nothing to validate: the only vector a head
produces is a softmax over the option set the request declared. Schema
violations are not caught, they are unrepresentable.

And the probabilities are meant to be believed. Calibration is the product, not
a footnote — which is why the release gates are ECE numbers, why the eval
harness ships with the weights, and why a deployment that has not been
calibrated says so on `/healthz`.

## Three primitives

| Primitive | Question | Answer |
| --- | --- | --- |
| **Choice** | one option from a defined set | selected option, probability per option, confidence |
| **Score** | ordered descriptive levels | continuous score (expectation over levels), probability per level, confidence |
| **Noul** | yes/no against criteria | probability that the answer is yes |

Confidence is a **serving-layer statistic**, not a model output: it summarises
how concentrated the calibrated distribution is. Choice uses normalised
entropy, scaled so a 2-option and a 77-option answer read comparably. Score
uses dispersion, because entropy throws away the fact that levels are ordered.
Noul gets no confidence field at all — for a binary question the probability
already is the answer.

## Install and run

```bash
pip install -e ".[dev,server]"

trigon ask request.json            # one request, no weights required
trigon serve --port 8000           # the reference gateway
trigon eval all -n 200             # three suites; exits non-zero on a failed gate
trigon spec                        # the OpenAPI contract
```

Everything runs on a fresh clone with no weights and no GPU, because the
lexical floor backend is a real baseline. A reproducible-evals story that needs
a checkpoint before anyone can watch it run is not reproducible.

The reference architecture and the training loop need the extra:

```bash
pip install -e ".[train]"          # torch
pytest tests/test_independence.py  # proves the architectural claims
trigon train --out reports/run.md  # train, calibrate, and run the gates
```

`trigon train` is the loop the rest of the repo exists to support, end to end on
one machine: generate outcome-grounded data, fit the readout heads against
proper scoring rules, measure calibration on a held-out split, fit a
temperature, and put the result through the release gates. Its point is that
the gates are passed — or failed — by a model rather than asserted about one.

## What is proved, not claimed

`tests/test_independence.py` runs the reference model and asserts:

- **adding, removing or reordering questions moves no other answer** — exactly,
  to floating-point equality, including at twenty extra questions. This is what
  "added questions cause no context rot" has to mean to be worth saying.
- **the schema half of the sequence encodes identically regardless of state**,
  which is what makes it a cacheable cross-request KV prefix.

Both are properties of the layout, so they hold for an untrained model exactly
as they will for a trained one. Getting them took more than a block mask: see
`docs/decisions.md` on group-local positions and on why state does not attend
to the schema.

## Layout

```
src/trigon/
  types.py          the wire contract; rejects requests that could be ambiguous
  schema/           compiler: layout, block mask, cache keys, budgets
  calibration/      temperature, conformal wrappers, ECE/Brier/reliability
  confidence.py     the two statistics, and why Score needs its own
  backends/         lexical floor, LLM baseline, torch reference model
  retrieval.py      large-cardinality prefilter and its recall gate
  training/         proper scoring rules and the outcome-grounded training loop
  engine.py         the pipeline, in one place so nothing can drift
  server/           reference gateway; generates the OpenAPI spec
  evals/            three suites, one runner, release gates
spec/openapi.json   the contract, drift-tested against the gateway
docs/               architecture, training, data + licence audit, evals, decisions
```

## Documentation

- [`docs/decisions.md`](docs/decisions.md) — the four open questions answered,
  plus the calls the implementation forced
- [`docs/architecture.md`](docs/architecture.md) — sequence layout, attention
  rules, heads, tiers
- [`docs/training.md`](docs/training.md) — the two objectives, the schedule,
  the auxiliary losses
- [`docs/data.md`](docs/data.md) — five streams and the dataset licence audit
- [`docs/evals.md`](docs/evals.md) — the three suites and the release gates
- [`docs/roadmap.md`](docs/roadmap.md) — phases, staffing, cut order, risks

## Scope

**In:** typed decisions, calibrated probabilities, single-pass multi-question
inference, large option sets, self-hosting.

**Out of scope for v1, by design:** text generation, image and audio input,
multi-turn state. Also explicitly out: arithmetic, counting and date
comparison. Keep math in code — the jaggedness suite measures those anyway, so
the non-goal is a published number rather than a claim.

## Calibration, and how not to fool yourself with it

A calibration number is only evidence if a *calibrated* model could not have
produced it by chance. Both ECE estimators are biased upward at small samples,
and the bias is the same size as a typical gate: on 4-way predictions a
perfectly calibrated model scores a mean ECE of about 0.12 at n=60 and about
0.03 at n=1,000 — the latter being most of a 0.05 gate.

So every report here prints the measured ECE beside a simulated floor, and the
release gates check the measurement before they check the model: a run must
carry at least 5,000 scored questions, and its own floor must sit at or below
half the limit. A run that cannot separate a calibrated model from a
miscalibrated one certifies neither.

## Status

Phase 0 of a 16-week plan: contract, scaffolding, calibration layer, eval
harness, and a training loop that closes it. See
[`docs/roadmap.md`](docs/roadmap.md).

## Licence

Apache-2.0, for code and for weights when they exist. No field-of-use
restriction, no acceptable-use rider — see `docs/decisions.md` §3 for why, and
for why dataset licensing is governed by a stricter policy than the code.
