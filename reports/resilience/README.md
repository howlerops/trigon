# Resilience on Modal: a launcher that is gone, and a container that is killed

Two claims in the ledger had been built and unit-tested but never exercised
on real infrastructure. First, a Modal run outlives the machine that launched
it. Second, a seed whose container dies resumes from its last finished epoch
instead of step 0. Both were tested on purpose on 2026-09-25. All three runs
are Banking77 on the spike: seed 0, 2,000 training cases, on `NVIDIA A10`, at
commit `6449526` with a clean tree. Their accuracy is not the point.

## The launcher is gone: `resilience-*`

A separate cloud session, a child of the one that wrote this, checked out the
branch and ran `modal_train.py launch`. It returned the run id and was then
**archived**, which releases its container. This session never launched the
run and never saw its call handles. It collected the run by id from the
Volume after the launcher was gone: return code 0, clean-tree commit
recorded, reports and checkpoint intact.

## A killed container resumes

`modal container stop` on the training container, three times. Each time
Modal started a new container for the same input within about 8 seconds
(`retries=modal.Retries(max_retries=2)` on `train_one`). The new container
read the seed's resume file from the Volume and continued. Each seed's log is
appended across its lives, so both halves are in one file.

| Run | Killed during | Second life says | Outcome |
| --- | --- | --- | --- |
| `resilience` (3 epochs) | evaluation, after epoch 3 | `resuming after epoch 3/3` | skipped training, evaluated, return code 0 |
| `resilience-midrun` (6 epochs) | evaluation, after epoch 6 | `resuming after epoch 6/6` | the same |
| **`resilience-epoch2`** (8 epochs) | **epoch 3, mid-training** | **`resuming after epoch 2/8`** | re-ran epoch 3 from its start, then epochs 4–8; return code 0 |

The third row is the one that matters: a mid-training kill on real hardware
lost only the unfinished epoch. `tests/test_training.py` already shows on CPU
that a resumed run ends bit-identical to an uninterrupted one. This run shows
the mechanism working end to end: the container dies, Modal retries the
input, and the resume file is read back from the Volume.

**Two things this found.**

- **`elapsed_s` timed only the last life.** A restarted seed's record said 196
  s for a run that took longer. Run records now carry `lives`, counted from the
  log. The three records here had it added after collection, from their logs,
  and they say so.
- **The status line lags by up to two minutes.** The log reaches the Volume on
  a two-minute commit, so a kill aimed at "mid-training" from the status line
  landed after training twice. The third kill was timed from the container's
  own start time instead.

**What is still not observed** is a preemption that *Modal* initiated. A
`container stop` is ours. Modal documents that it reschedules preempted
inputs, and the retry covers any container death, but the first real one has
not happened while anyone watched.
