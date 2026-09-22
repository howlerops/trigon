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
`size`. Prefix-LM conversion of a 0.5–1.5B open base. Blocker: a GPU — **checked
now, and real**: `nvidia-smi` is absent, `torch.cuda.is_available()` is False,
and the machine is a 4-core Xeon with 15 GB and no accelerator.

**This invalidates every number in `reports/`.** Sequence it after A.1 and A.2
so there is a real-data baseline to compare against, not only a synthetic one.

**Done when** four seeds certify with all three synthetic questions above
their marginals *and* per-corpus ECE holds on real data.

**A.4 Sweep the reference configuration on CI's hardware.** 🔄 **In flight.**
The `reference-run` job now runs `seed_sweep.py --seeds 0 1 2 3 --jobs 4`
instead of training one seed, and publishes the spread to the job summary.
`--require none` for now, deliberately: the point of the change is to *get*
the four-seed measurement on that hardware before deciding what to gate on.

**It has still not produced one**, for two reasons, and the second is now the
blocker.

The first was the workflow: the job shared a `cancel-in-progress` concurrency
group with the fast tests, so every commit pushed during the half-hour run
discarded it. Twenty of one day's twenty-six runs were cancelled that way. It
now has its own group and does not cancel, so sweeps queue rather than vanish.

The second is that **CI stopped running at all**. Runs 26 and 27 failed with
every job ending in three to five seconds, zero steps recorded and their logs
returning 404 — the runner never reached `actions/checkout`. Run 12 was green
on substantially this workflow. Run 26 failed this way *before* the
concurrency change, so that edit is not the cause.

This is a private repository owned by an organization, so Actions minutes are
metered, and 27 runs in a day — several of them half-hour sweeps — is the
shape of a quota. That is a hypothesis, not a finding: the billing endpoint
returns 403 to this session and re-running a failed workflow is also refused,
so it cannot be confirmed from here. **It needs someone with billing access
to look.** Until then A.4 is blocked on infrastructure rather than on
anything in the repository.

The certified configuration certifies on four seeds *on one machine*, and
GitHub's runners produce a draw worse than any of them. `scripts/seed_sweep.py` already says
why: a perturbation the size of a matmul's summation order moves a seed from
one outcome to the other, and `tests/test_prefix_cache.py` now shows this
hardware has a different one. Hardware is a second axis of the seed problem
and nobody swept it. Blocker: none — it triples the CI job's wall clock, which
is a cost rather than an obstacle.

**Done when** `seed_sweep.py --require all` runs in CI and the reference-run
job blocks on its verdict again, or the configuration is replaced with one
whose outcome does not depend on which machine trains it.

---

**A.5 Bucket training chunks by length.** The trainer batches cases into
accumulation chunks and pads each to its longest member, and on a corpus of
free text that is expensive: HelpSteer2 compiles to 253–3,647 tokens, and at
the default chunk of 8 a randomly ordered epoch wastes **2.82×** of the
attention work on padding. Sorting within a shuffled epoch — sortish batching
— brings it to 1.04×.

🟡 **Implemented, measurement in flight.** `TrainingConfig.bucket_window`
draws a window of `accumulate × bucket_window` cases from the already-shuffled
epoch, sorts it by compiled length, and cuts it into chunks. The window is the
point: sorting the whole epoch would make every chunk uniform *and* show the
model all its short cases before any long one, which is a curriculum nobody
chose.

It was deferred the first time because it changes which cases share a gradient
step, and making that change mid-certification would invalidate the comparison
it exists to speed up. There is no certification in flight now — the three
HelpSteer2 attempts all died — so the objection is spent.

**The before-and-after is still owed** and this item is not closed without it.

**Done when** a four-seed sweep at the same configuration certifies with the
bucketing on, and the wall clock is published beside the unbucketed run.

---

## Stage B — Make the cost claim real

**B.1 The L4 burn-in.** Unchanged and still the most load-bearing unmeasured
number in the project: $/MTok decides whether the economic story is "an order
of magnitude" or "about twice". The token-layout advantage is measured at
~1.6–2.9× break-even and everything beyond that is arithmetic over an
unsourced input. Blocker: an L4 — **checked now, and real**: no GPU is present
or reachable from this machine.

B.3 moves this number, though, and in the right direction: a 6× wall-clock
saving at the served shape is 6× the requests per GPU-second, which is the
denominator of $/MTok. It does not replace the burn-in; it means the burn-in
starts from a different place than the one the pricing model assumed.

**B.2 The model server.** 🟡 **Built, measured, and it does not pay here.**
`Engine.answer_many` coalesces requests into one forward pass and
`tests/test_batching.py` asserts a batched answer is identical to the one the
request would have got alone — the block mask's guarantee extended across
requests, which is the only property worth testing, because a batcher that
mixes two callers' states produces well-formed answers to questions nobody
asked.

On this hardware it is **30% slower**: 3.43 ms per request alone against 4.14
ms in a batch of 16 (`reports/batching/README.md`). Not padding — the
synthetic corpus wastes 1.02× on that — but that batching fills parallel
capacity a small request leaves idle, and on one CPU thread there is none to
fill. Everything defaults to serial; `batch_size > 1` opts in.

**The throughput claim this item wanted needs the same GPU A.3 and B.1 need.**
What is measured is that the batching layer costs 10–20% on CPU, which is
worth knowing before building a scheduler on top of it. The remaining piece —
a gateway queue that coalesces concurrent requests — is deliberately not built
against a primitive that is currently a loss.

**B.3 Measure the KV cache's wall clock.** ✅ **Done, and it changed the
default.** 6× at the shape the certified Banking77 model serves and 23× at 256
options; `reports/cache/README.md`. The cache is now on by default, `/healthz`
reports it, and `TRIGON_CACHE_PREFIXES=0` turns it off.

---

## Stage C — Make it something a stranger can adopt

**C.1 Publish weights — but not these.** The argument has weakened and still
holds. When this was written the only weights were a spike whose accuracy was
not a result. `reports/banking77/` now has four certified checkpoints at
71–74% with ECE 0.0177–0.0361, which *are* a result, so the reason not to
publish is no longer "it answers badly" — it is that a 128-wide two-layer
model at 74% invites comparison with a fine-tuned BERT at 93% on a solved
benchmark, and loses. What would make publishing worth doing is A.3.

 Stage 4.3 of the last plan is
deliberately not done. The reference model is a spike whose accuracy is not a
result, said so from the first commit, and publishing its weights under the
project's name would be the most misleading thing in the repository: a
`model_version` that answers questions badly is worse than no weights at all.
This unblocks when A.3 does.

**C.2 A cookbook per use case.** ✅ **Done.**
`examples/usecase_cookbook.py <name>` runs any committed use case end to end
on a fresh clone: the state it reads, one answer per question in the terms its
primitive actually has, the confidence-routing decision, and what the request
cost in tokens and milliseconds.

One driver rather than three scripts, because the use cases already live in
one place and three copies of the same twenty lines diverge the first time one
is edited. A test runs every use case in `all_use_cases()`, so adding one
without a runnable example fails the build instead of quietly enlarging the
catalogue.

**C.3 Rate limiting and auth on the compat path.** ✅ **Done.** All three
codes are emitted, with the headers a client needs to act on them, on both the
native and the compat paths — and all three stay off unless an operator
configures them, because a self-hosted gateway should not invent a policy
nobody chose. `docs/compat.md` has the variables.

---

## Ordering, and the one thing that does not wait

A before B before C, for the reason the last plan gave and which has only got
stronger: **a well-calibrated wrong answer is the worst product this project
could ship**, and calibration is now far ahead of accuracy. Serving a model
whose numbers do not transfer faster, or cheaper, or behind a nicer adapter,
is optimising the wrong thing three times over.

The exception was **B.3**, the KV cache timing, and it is done. It needed a
quiet machine rather than a better one, and it closed a claim that had been
published as a token count with no time under it — then turned the default
around.

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
