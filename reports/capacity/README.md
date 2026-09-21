# Why `size` is unlearned — the ablations

`size` asks which tier a seat count falls in: `min(seats // 128, 3)` over
`seats ∈ [1, 500]`. It has sat on its own marginal predictor on **every seed of
every configuration** this project has trained, and it is the single largest
piece of accuracy left on the table — at 20% label noise a perfect answer
scores 0.85, the model scores 0.25, and it is one question of three.

Two explanations were offered before anything was measured. `docs/model-card.md`
said arithmetic is a documented non-goal. `docs/decisions.md` said the tokenizer
had fused whole numbers into single tokens, so 127 and 128 were unrelated
embedding rows. The second is demonstrably true and turned out not to be the
binding constraint: splitting digits one per token left `size` exactly where it
was and cost accuracy on two other questions.

These are the arms that followed. All use the digit tokenizer, so it is held
constant and is not the variable under test. Lift is over each question's own
marginal predictor.

## Capacity

| Arm | `size` seed 0 | `size` seed 1 |
| --- | ---: | ---: |
| 128×2 (reference shape) | −0.0055 | −0.0112 |
| 128×4 (twice the depth) | −0.0025 | −0.0182 |
| 256×4 (four times the parameters) | *running* | −0.0070 |

**Capacity is not the constraint.** Four times the parameters leaves `size`
exactly where it was. That is the useful shape of this result: a question the
model could compute but was short of capacity for would improve *somewhere*
along that row, and this does not move at all.

Doubling depth does nothing. It also made `plan` worse — 0.6112 and 0.4562,
against 0.848 and 0.646 at 128×2 — which is its own small finding: more layers
on this data is not free.

## The head

`size` is a **Score**, and a Score runs the dot-product head *without* the
residual. That is the configuration `docs/decisions.md` records as collapsing
to the marginal: the keys converge and the query freezes, and the head answers
the same thing regardless of input.

The symptom fits. `size` emits a near-constant answer — **sd 0.019 across the
whole seat range** — rather than a noisy wrong one. A model that cannot read
its input guesses; a head whose keys have converged says one thing forever.

The residual was scoped to Choice on the strength of one measurement that read
the *pooled* run metric on a single seed, and so could not see what the Score
question itself did. That comment ends "measure it on Score before extending it
there". Measured:

| Arm | `size` seed 0 | `size` seed 1 |
| --- | ---: | ---: |
| no Score residual (control) | −0.0055 | −0.0112 |
| **`--score-residual`** | +0.0002 | **+0.2425** |

**Seed 1 is the first time `size` has ever been above its marginal**, on any
seed of any configuration this project has trained — and it is not a nudge.
0.5008 against a 0.2582 marginal closes 41% of the distance to the 0.85 a
perfect answer scores at this noise level.

Seed 0 does not reproduce it, and that seed's `plan` also collapsed to chance,
so it is a bad draw rather than a clean refutation. What the pair establishes
is that the head **can** learn this question, which eight prior measurements
said it could not. How reliably is the next question, and it is the one this
project keeps having to ask: a fix that works on some draws is the same shape
of non-result as a configuration that certifies on some seeds.

## Which repair

Two candidates, and they differ in kind.

`--score-residual` makes collapse *less likely* by carrying each level's own
input embedding past the encoder, so the keys start distinct. The failure mode
still exists; descent is merely pushed away from it.

`--score-head linear` **removes the failure mode**. It reads every level off
one readout slot through a fixed-width head, the shape `max_levels` was
declared for and never used. The dot-product head is already a linear readout
of one vector through fixed directions — the schema half of the sequence
encodes identically regardless of state, which is the cacheability claim
`tests/test_independence.py` asserts, so the pooled level states it scores
against carry no state information at all. The linear head computes the same
function without the indirection that lets those directions converge.

Given that seed variance is the recurring finding of this whole project, a
repair that removes the failure mode should beat one that makes it rarer. That
is a prediction, and both arms are running against it.
