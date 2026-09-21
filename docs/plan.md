# From here to a drop-in open-source alternative

`docs/roadmap.md` is the build plan inherited from the brief, restated against
what has been measured. **This is the execution plan**: what actually has to be
true, in what order, and how each step is known to be done. It exists because
the roadmap's phases are sized in weeks and the real constraints turned out to
be different ones.

`docs/ledger.md` is the running record of what is done. This is the record of
what is not.

---

## What "drop-in" has to mean

A caller with a working integration against the incumbent should be able to
change a base URL and a key, and have it keep working. That decomposes into
five obligations, and only the first is close to met:

| Obligation | State |
| --- | --- |
| **Wire compatibility** — same request and response shapes | Contract is a superset; a compatibility layer is unwritten |
| **Semantic compatibility** — the same input yields a usable answer | Model answers one question of three well |
| **Envelope compatibility** — accepts anything they accept | `COMPAT_BUDGET` reproduces their limits exactly ✓ |
| **Operational compatibility** — latency and availability they can deploy on | Gateway measured; no model server, no KV cache |
| **Economic compatibility** — cheaper, provably | Token layout buys ~2×; $/token unmeasured |

**The hard one is semantic.** Everything else is engineering with a known
shape. A drop-in that returns well-formed, well-calibrated, *wrong* answers is
worse than no drop-in, because the calibration makes the wrongness credible.

---

## Stage 1 — Make the model learn what it is asked

**Why first.** Nothing downstream matters if the answers are not good. The
current model answers a copy task at Bayes-optimal, a conjunction on three
seeds of four, and a threshold over a wide range not at all. Two of those three
are model problems, not calibration problems, and calibration is already ahead
of the model.

**1.1 Finish the `size` investigation.** Four seeds are running on a
prose-independent vocabulary. Three hypotheses are dead (tokenizer alone,
depth, capacity) and one positive observation did not reproduce. Outcomes:

- it reproduces across seeds → the Score head's repair becomes the default and
  the 2×2 is the finding;
- it does not → the remaining candidate is that *one readout slot* is a real
  bottleneck for a question needing magnitude comparison, and the test is a
  Score with readout-per-level, which the Choice head already does.

**Done when** `worst_question_over_baseline` passes on the median seed, or the
question is documented as out of reach with a measurement rather than an
assertion.

**1.2 Make `at_risk` reliable.** Learned on three seeds of four, +0.09 against
a Bayes ceiling of +0.14. The failing seed sits exactly on its marginal, which
is the collapse signature again.

**1.3 Replace the spike with a real backbone.** 128-wide and two layers is a
spike, said so from the start, and every seed-variance finding in this project
is partly a symptom of it. Prefix-LM conversion of a small open base (0.5–1.5B)
is the phase-1 item the roadmap already carries. **This is the single largest
expected accuracy gain and it is also the one that invalidates the most
measurements** — every per-question number, every ECE, the whole `reports/`
directory. Sequence it *after* 1.1 so the diagnosis is not confounded.

**Done when** a real backbone certifies on four seeds with all three questions
above their marginals, and the jaggedness suite runs against it.

---

## Stage 2 — Make the numbers transfer

**2.1 Build the data streams.** Four of five are unbuilt; outcome grounding
rests on synthetic data alone. The licence audit has cleared nine corpora to
green, which is enough to start: Banking77, CLINC150, MASSIVE (Choice);
Circa (Noul); HelpSteer2, measuring_hate_speech (Score); GoEmotions,
Civil Comments (both, with distributions).

The distribution-carrying ones matter most. A model trained on hard labels
learns to be confident; a model trained on *annotator distributions* learns
what disagreement looks like, which is the product.

**2.2 Train on them and publish per-corpus calibration.** The per-primitive
slice already exists; per-corpus is the same machinery. A single pooled ECE
across nine corpora would hide exactly what a caller needs to know.

**2.3 Resolve CC BY-SA.** BoolQ and FEVER are blocked on counsel. The question
recurs for every share-alike corpus, so it is worth answering once.

**Done when** ECE ≤ 0.05 holds per corpus on held-out data, not pooled, and the
conformal wrapper's coverage holds per corpus too.

---

## Stage 3 — Make it deployable

**3.1 The KV cache.** The layout permits it and `tests/test_independence.py`
asserts the property; nothing caches it. `Usage.cached_schema_tokens` reports 0
and says so. This is the difference between the cold and cached columns in
`docs/pricing.md` — roughly a third of the token cost for these use cases.

**3.2 The model server.** The gateway is 1.5% of the p50 budget and saturates
at ~430 req/s per process; there is no model server behind it and no batching
across requests. Continuous batching over a prefill-only model is simpler than
the general case: no decode loop, no ragged generation, every request one pass.

**3.3 The L4 burn-in.** $/MTok is the entire cost argument beyond ~2×, and it
is arithmetic over unsourced inputs. **This is not a phase-3 nicety.** Until it
is measured, the economic claim is a guess, and the pricing tool says so.

**Done when** ≤150 ms p50 at a stated QPS on named hardware, with the
quantized-vs-float ECE delta inside 0.01 — which is already gated and already
measured at 0.0002 for weight-only int8.

---

## Stage 4 — Make it actually drop-in

**4.1 The compatibility adapter.** The contract is a superset, which means a
translation layer is small and mechanical: accept their request shape, map it
onto ours, map our response back. `COMPAT_BUDGET` already reproduces their
limits exactly, so rejection behaviour matches without special-casing.

**4.2 A migration harness.** Point it at both endpoints with the same traffic,
and report per-question agreement, calibration on each, and where they diverge.
A caller will not switch on a promise; they will switch on a diff over their
own traffic. **This is the most persuasive artifact in the whole plan and it is
about two hundred lines**, because both sides are already speaking a typed
contract.

**4.3 Publish weights and the harness.** Apache-2.0, no field-of-use
restriction, with the eval harness that produced every number.

**Done when** a caller can run the migration harness against their own traffic
and read a number.

---

## What could make this not work

Stated now, so the answer is not written after the fact:

- **The backbone conversion costs more quality than the architecture buys.**
  Prefix-LM conversion of a causal base is the least certain step. Mitigation:
  it is measurable early, and the jaggedness suite exists to measure it.
- **Calibration does not transfer to a caller's domain.** The biggest risk in
  the original plan. Mitigation already ships: `trigon fit --conformal-out`
  gives a distribution-free guarantee on a few hundred of their own labels, and
  it now verifies its own promise and exits non-zero when it fails.
- **$0.007/MTok is wrong.** If a small typed model does not serve tokens far
  cheaper than a frontier one, the economic case is a 2× premium, not an order
  of magnitude. This is the single most load-bearing unmeasured number in the
  project.
- **Seed variance survives the backbone.** Every configuration tried so far
  decides its own outcome by seed to some degree. If a real backbone does not
  fix it, certification needs more seeds and the cost of every experiment
  multiplies.

---

## Ordering, and why

Stage 1 before everything, because a well-calibrated wrong answer is the worst
product this project could ship. Stage 2 before 3, because serving a model
whose numbers do not transfer is optimising the wrong thing. Stage 3 before 4,
because a migration harness that shows worse latency is an argument against
switching.

The one item that does not wait is **3.3, the L4 burn-in**. It is cheap, it is
independent of everything else, and it decides whether the economic story is
"an order of magnitude" or "about twice" — which changes what the rest of this
is for.
