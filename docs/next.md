# The next plan

`docs/plan.md` was the execution plan for getting from a working phase-0 repo
to a drop-in alternative. Most of what it could reach without hardware, data
access or counsel has now been reached, and two of its items turned out not to
be blocked at all — they were blocked on a belief. This is the plan for what
is left, written after those findings rather than before them.

`docs/ledger.md` is the record of what is done. This is the record of what is
next, and what each step would have to show to count.

---

## What the last plan actually established

| Stage | State |
| --- | --- |
| 1.1 `size` | **Closed, negative.** Seven interventions, 22 runs, below its marginal on every seed |
| 1.2 `at_risk` | Learned on three seeds of four. Unchanged |
| 1.3 real backbone | Not started. Needs a GPU |
| 2.1 data streams | **One of five built.** Banking77 loads, licence-gated in code |
| 2.2 per-corpus calibration | Machinery built (`scripts/train_corpus.py`); one corpus measured |
| 2.3 CC BY-SA | Unresolved. Needs counsel |
| 3.1 KV cache | **Built, off by default**, for a measured reason |
| 3.2 model server | Not started |
| 3.3 L4 burn-in | Not started. Needs an L4 |
| 4.1 compat adapter | **Built.** Was never blocked — the wire format is published |
| 4.2 migration harness | **Built, and now able to reach an incumbent** |
| 4.3 publish weights | Deliberately not done. See below |

**The two most useful findings of the last plan were both about the plan.**
Stage 4.1 was parked on "needs the incumbent's real wire format" and the
format is public. Stage 2.1 was parked on "data streams unbuilt" and the first
corpus was a `curl` away, already cleared green by our own audit. Both were
recorded as blocked and neither was. That is a failure mode worth naming
before writing another plan: *a blocker asserted once and never re-checked is
indistinguishable from a blocker that is real*, and it costs more than a bad
estimate because nobody argues with it.

So every item below carries the thing that would unblock it, and whether that
thing has been **checked** or is **assumed**.

---

## Stage A — Make the accuracy claim transfer

This is now the only thing standing between the project and a usable product.
Wire, envelope, calibration machinery, gates and migration tooling are all
built; the model answers one synthetic question of three.

**A.1 Certify Banking77 on a seed spread.** A two-seed pilot is measured and
`accuracy_over_baseline` passes enormously: +0.4460 and +0.4027 against a
marginal predictor of 1.8%. Its ECE is **not** quotable — the pilot evaluated
on 1,500 rows, `sample_size` and `gate_is_testable` both failed, and a
perfectly calibrated model scores 0.0221 there against a measured 0.0324.
Blocker: none (checked — it runs on CPU in about 90 minutes per seed at the
corrected sizes).

**The done-condition below had to be repaired before it could be met.** Stage
2 of the last plan asks for "ECE ≤ 0.05 per corpus" and Banking77's test split
is 3,080 rows against a `MIN_CALIBRATION_SAMPLES` of 5,000, so the condition
was unreachable on this corpus as written. The evaluation set is topped up
from rows held out of train, capped at half the pool so the top-up cannot eat
the training set; when half is not enough, the floor is not reached and the
gate fails, which is the correct answer rather than a problem to route around.

**Done when** ECE ≤ 0.05 and `accuracy_over_baseline` ≥ 0.05 hold on the
median of four seeds, on an evaluation set at or above the sample-size floor,
with the report stating how much of it came from each split.

**A.2 Add the annotator-distribution corpora.** Blocker: **checked, and it
was half real.** HelpSteer2 is gzipped JSONL, needs nothing beyond stdlib, and
is now loaded — five ordered ratings over one state, the first real exercise of
the Score primitive. GoEmotions, measuring_hate_speech and Circa are
Parquet-only, and a reader for them would put a compiled dependency in the
import path of the calibration math and the drift tests. Those get converted
in `scripts/` first, or not at all.

**What is still missing is the thing the stream is for.** HelpSteer2's main
split carries *aggregated* integer ratings, not per-annotator distributions;
the distribution-carrying data is in its `disagreements/` split and is a
separate load. `Expectation.distribution` exists and nothing has ever used it.
A model trained on hard labels learns to be confident; a model trained on
annotator disagreement learns what disagreement looks like, which is the
product.

**Done when** the calibration report is published per corpus, never pooled,
and coverage holds per corpus too. Nine corpora pooled into one ECE would hide
exactly what a caller needs to know.

**A.3 Replace the spike.** Unchanged from the last plan and now the single
highest-value item, because it is also the only remaining explanation for
`size`. Prefix-LM conversion of a 0.5–1.5B open base. Blocker: a GPU
(**assumed**, not checked — nothing in this session tried to find one).

**This invalidates every number in `reports/`.** Sequence it after A.1 and A.2
so there is a real-data baseline to compare against, not only a synthetic one.

**Done when** four seeds certify with all three synthetic questions above
their marginals *and* per-corpus ECE holds on real data.

**A.4 Sweep the reference configuration on CI's hardware.** The certified
configuration certifies on four seeds *on one machine*, and GitHub's runners
produce a draw worse than any of them. `scripts/seed_sweep.py` already says
why: a perturbation the size of a matmul's summation order moves a seed from
one outcome to the other, and `tests/test_prefix_cache.py` now shows this
hardware has a different one. Hardware is a second axis of the seed problem
and nobody swept it. Blocker: none — it triples the CI job's wall clock, which
is a cost rather than an obstacle.

**Done when** `seed_sweep.py --require all` runs in CI and the reference-run
job blocks on its verdict again, or the configuration is replaced with one
whose outcome does not depend on which machine trains it.

---

## Stage B — Make the cost claim real

**B.1 The L4 burn-in.** Unchanged and still the most load-bearing unmeasured
number in the project: $/MTok decides whether the economic story is "an order
of magnitude" or "about twice". The token-layout advantage is measured at
~1.6–2.9× break-even and everything beyond that is arithmetic over an
unsourced input. Blocker: an L4 (**assumed**).

**B.2 The model server.** No batching across requests today. Continuous
batching over a prefill-only model is the easy case — no decode loop, no
ragged generation, one pass per request — and the gateway is already measured
at 1.5% of the p50 budget, so the batching layer is where the latency story is
won or lost. Blocker: none checked; the logic is testable on CPU.

**B.3 Measure the KV cache's wall clock.** It is built, it skips 53 of 69
positions on a typical request, and its saving has never been timed on an idle
machine. Until it is, "the schema is cacheable" is an architectural property
with no number attached. Blocker: none — it needs a quiet machine, not a
better one.

---

## Stage C — Make it something a stranger can adopt

**C.1 Publish weights — but not these.** Stage 4.3 of the last plan is
deliberately not done. The reference model is a spike whose accuracy is not a
result, said so from the first commit, and publishing its weights under the
project's name would be the most misleading thing in the repository: a
`model_version` that answers questions badly is worse than no weights at all.
This unblocks when A.3 does.

**C.2 A cookbook per use case.** Three use cases are committed
(`trigon.usecases`) and priced (`docs/pricing.md`). None has a worked
end-to-end example a reader can run. Blocker: none.

**C.3 Rate limiting and auth on the compat path.** `docs/compat.md` lists
`401`, `429` and `529` as codes the adapter never returns, which means a
caller's backoff path is untested against this server. Blocker: none.

---

## Ordering, and the one thing that does not wait

A before B before C, for the reason the last plan gave and which has only got
stronger: **a well-calibrated wrong answer is the worst product this project
could ship**, and calibration is now far ahead of accuracy. Serving a model
whose numbers do not transfer faster, or cheaper, or behind a nicer adapter,
is optimising the wrong thing three times over.

The exception is **B.3**, the KV cache timing. It needs a quiet machine rather
than a better one, it takes an afternoon, and it closes a claim that is
currently published as a token count with no time under it.

---

## What could make this not work

Restated from the last plan where it still holds, with one addition.

- **The backbone conversion costs more quality than the architecture buys.**
  Unchanged, and now also the last hypothesis standing for `size`.
- **Calibration does not transfer to a caller's domain.** The mitigation
  ships: `trigon fit --conformal-out` gives a distribution-free guarantee on a
  few hundred of their own labels and exits non-zero when it fails.
- **$0.007/MTok is wrong.** Unchanged.
- **Seed variance survives the backbone.** Unchanged, and the `size` record
  sharpens it: across seven interventions the spread *within* an intervention
  was larger than the difference *between* interventions. If a real backbone
  does not fix that, every experiment in this project costs four runs and the
  ones above get four times more expensive than written.
- **New: the real corpora do not behave like the synthetic one.** Banking77 is
  77 options and 606 tokens a request against four options and 70. Nothing in
  this repo's numbers — latency, cost, the attention tables, the retrieval
  trigger — was measured at that shape. The first per-corpus report is as
  likely to move an infrastructure number as an accuracy one.
