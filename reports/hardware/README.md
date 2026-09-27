# The reference configuration on GitHub's hardware

`docs/next.md` A.4. The certified spike configuration — 8,000 cases, 8
epochs, d_model 128, 2 layers, noise 0.2, `--option-scoring auto` — had
certified on four seeds on **one machine**. Its first draw on GitHub's
runners failed: choice accuracy 0.530 and choice ECE 0.1076, where the four
local seeds read 0.648–0.849 and 0.0064–0.0240. Hardware changes the
summation order of a batched matmul, and `scripts/seed_sweep.py` already
documents that a change that size can move a seed from one outcome to the
other. So hardware is a second axis of the seed problem, and nobody had swept
it.

CI's `reference-run` job now sweeps seeds 0–3 on every push to `main`. Three
of those sweeps have finished.

## The three sweeps

| Commit | Job | Seed | Final loss | Kept epoch | Lift over baseline | ECE | Certified |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `4e3deae` | 108300548471 | 0–3 | *identical to `c23cedc`, bit for bit* | | | | 4 of 4 |
| `c23cedc` | 108469252018 | 0 | | | +0.1603 | 0.0387 | **yes** |
| | | 1 | | | +0.1316 | 0.0359 | **yes** |
| | | 2 | | | +0.0707 | 0.0141 | **yes** |
| | | 3 | | | +0.1513 | 0.0128 | **yes** |
| `efa7311` | 108489983080 | 0 | 1.1028 | 2 | +0.1632 | 0.0244 | **yes** |
| | | 1 | 0.8697 | 1 | +0.2200 | 0.0329 | **yes** |
| | | 2 | 1.0959 | 1 | +0.1295 | 0.0160 | **yes** |
| | | 3 | 0.9109 | 1 | +0.2135 | 0.0184 | **yes** |

Final loss for `c23cedc`: median 1.0028, range 0.9957–1.1279. For `efa7311`:
median 1.0034, range 0.8697–1.1028. Blank cells in the `c23cedc` rows were not
transcribed from that job's summary. Every seed's report, with its noise
floor, is in that run's `reference-run` artifact. A seed certifies only if
`gate_is_testable` passes, so every ECE above sits clear of its simulated
floor.

**Twelve seed-runs of twelve certify.** The configuration now holds on three
hardware draws: this machine, and at least two distinct kinds of GitHub runner.

| Where | Seeds | ECE | Lift over baseline |
| --- | ---: | --- | --- |
| Local (`reports/iso/`) | 4 of 4 | 0.0084–0.0247 | +0.1614 to +0.2277 |
| GitHub, `4e3deae` = `c23cedc` | 8 of 8 | 0.0128–0.0387 | +0.0707 to +0.1603 |
| GitHub, `efa7311` | 4 of 4 | 0.0160–0.0329 | +0.1295 to +0.2200 |

## GitHub's runners are not one machine

`4e3deae` and `c23cedc` produced bit-identical results for all four seeds.
`efa7311` produced a different set, and the only difference between it and
`c23cedc` is PR #6, which touched no file under `src/`, `scripts/` or
`.github/` — reports and documentation only. Identical code gave different
numbers, so the runner changed. `ubuntu-latest` is a pool of CPU types, and
the job gets whichever it lands on.

That has two consequences:

- **One CI sweep is one hardware draw.** It is not "GitHub's number". The
  12/12 above covers at least two draws. It does not cover every CPU in the
  pool.
- **Reproducibility across commits is not a regression test** on this job.
  A difference between two pushes can be the runner, not the change. The
  per-question independence tests already state this for sequence lengths
  (0.0 here, 1.4e-08 there). This is the same fact, one level up.

## What was different about the failed draw

Nothing in the three sweeps explains why the first GitHub draw failed. It ran
a single seed, on a runner whose kind was not recorded, before the sweep
existed, so it cannot be compared cell for cell. What the sweeps show is that
the failure is not typical of GitHub's hardware: sixteen seed-runs certify
across three machines, and one did not. That is a rate to watch, not a
guarantee. The job now blocks on it (below), so the next failure is a red
build with four seeds attached rather than an anecdote.

## What changed as a result

The `reference-run` job gates on the sweep again: `--require all`, which
exits non-zero unless every seed certifies. That is A.4's own done condition.
It runs on pushes to `main` only, so a failure marks `main` red without
blocking a pull request.

`worst_question_over_baseline` is advisory and fails on every seed on every
machine. That is the same finding as `reports/iso/`: the spike learns `plan`
and sits near the marginal on the other two questions. On `efa7311` seed 1,
`worst_primitive_workhorse_ece` also fails, and it is advisory too.
Certifying says this configuration's *calibration* reproduces across
hardware. It does not say the spike is a good model.
