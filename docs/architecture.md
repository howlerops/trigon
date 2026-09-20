# Architecture

One forward pass. No decode loop. Every question answered in parallel, each
over exactly the labels the caller declared.

## The sequence

```
[ schema: q1 block | q2 block | ... ] [ state ] [ readout slots ]
```

Schema first, because schema tokens depend only on the question map: their KV
is a prefix identical across every request carrying that schema, and that
cross-request cache is what the serving economics rest on. State next, because
it varies per request. Readout slots last, so they see both — and so the causal
fallback layout still works if prefix-LM conversion costs more quality than it
is worth.

## What may attend to what

Three rules, enforced by an additive mask rather than by convention:

| Group | May attend to |
| --- | --- |
| `schema:q` | itself only (with `isolate_question_schemas`, the default) |
| `state` | itself only (with `state_attends_to_schema` off, the default) |
| `readout:q` | `schema:q`, `state`, `readout:q` |

The consequences, in order of how much they matter:

**Added questions are free.** A readout sees its own schema block and a state
encoding no other question can perturb, so adding, removing or reordering
questions does not move any other answer — exactly, to floating-point
equality. That is asserted on the reference model in
`tests/test_independence.py`, including at twenty extra questions.

**Noul answers are independent.** Same mechanism. A Noul cannot be dragged by
its neighbours because it never sees them.

**Both halves are cacheable.** Schema KV is reusable across requests;
per-question schema isolation makes it reusable across *different question
sets* containing that question. State KV is reusable across different question
sets over the same document.

What this costs: state is encoded question-agnostically, so all cross-
referencing between state and schema happens in the readout slots rather than
throughout the stack. `state_attends_to_schema=True` recovers that capacity and
gives up the guarantee. Which side wins is the phase-1 ablation; the default is
the one whose published claim we can stand behind.

## Positions are group-local

The mask alone is not enough. With sequence-global positions, inserting a
question shifts every later token's position and moves hidden states the mask
never let it see. So each schema block, the state, and each question's readout
slots all start at position 0, and a learned segment-type embedding keeps the
three kinds distinguishable despite the overlapping indices.

This is also what makes a cached prefix portable — a prefix computed at one
offset is wrong at another.

## Heads

Everything is categorical. There is no regression head anywhere.

| Primitive | Head | Slots |
| --- | --- | ---: |
| Choice, ≤ 64 options | one readout slot per option → scalar logit | *n* |
| Choice, > 64 options | one readout slot, dotted with pooled option states | 1 |
| Score | one readout slot, dotted with pooled level states | 1 |
| Noul | one readout slot → one logit | 1 |

The Choice crossover is the phase-1 ablation. A slot per option is more
expressive; the dot-product head costs one slot regardless of cardinality,
which is what makes large option sets affordable at all.

Score is a distribution over declared levels, and the reported score is its
expectation. A caller asking for a 1–5 rating gets a distribution over five
levels and a continuous number, not a regression output that could land
anywhere.

## Type safety is structural

A Choice answer can only ever be a softmax over the option set the request
declared, because that is the only vector the head produces. There is no
sampler that could emit anything else, so schema violations are not caught —
they are unrepresentable.

Two places make that real rather than rhetorical:

- `trigon.types` rejects requests that could produce ambiguous answers
  (duplicate option names, single-option Choices, unordered Score values).
- `trigon.backends.base.validate_output` runs on every path and raises if a
  backend returns the wrong number of logits, the wrong head for a primitive,
  or a non-finite value. A broken backend fails loudly; it never serves a
  malformed answer.

## The pipeline

`trigon.engine.Engine` owns the order of operations, so the gateway, the eval
harness and the SDKs cannot drift apart:

```
narrow oversized Choices  →  compile layout + mask  →  one forward pass
  →  validate structure  →  temperature  →  confidence  →  conformal
```

Confidence is derived from the **calibrated** distribution. Deriving it from
raw logits would produce a number that looks like a probability and is not
one — and the confidence-gated escalation policy would then spend premium-tier
money on the wrong requests.

## Tiers

The workhorse serves the traffic; the premium tier answers what the workhorse
could not settle. Escalation is decided **per question**, and the premium pass
carries only the unsettled questions — affordable precisely because questions
are independent by construction. Noul escalates on distance from 0.5, since it
has no confidence field: a Noul at 0.5 is the model saying it does not know.

## Large cardinality

Above the budgets in `trigon.limits`, a Choice goes embed → prefilter →
model rescoring of a 256-option shortlist. Options the prefilter dropped come
back with probability 0 rather than disappearing: the caller declared them, so
the answer mentions them.

The number that governs this is `recall_at_k`. Everything below the
prefilter's recall is accuracy no model quality can recover, so it is checked
before any model number is believed.
