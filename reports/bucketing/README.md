# Length bucketing in the trainer: 1.63×, not 2.82×

`docs/next.md` A.5. Each optimizer step pads its cases to the longest one and
attention is quadratic, so a chunk of mixed lengths pays the difference.
`TrainingConfig.bucket_window` sorts a window of the already-shuffled epoch by
compiled length before cutting it into chunks.

One seed, one epoch, 400 HelpSteer2 cases, chunk 8, one torch thread, idle
machine, 4-core Xeon @ 2.80GHz.

| Bucketing | Wall clock | Rate | Final loss |
| --- | ---: | ---: | ---: |
| off | 351.7 s | 1.1 cases/s | 1.2322 |
| window 8 | 215.8 s | 1.9 cases/s | 1.2190 |

**1.63× faster.**

## Why it is not 2.82×

The padding arithmetic says a random chunk of 8 wastes 2.82× of the
*attention* work on HelpSteer2, and bucketing brings that to 1.04×. It does
not say the whole step gets 2.8× faster, because attention is not the whole
step: embedding, the readout heads, the loss, the backward pass through all of
it and the optimizer are all linear in tokens and unaffected by padding.

1.63× overall from 2.7× on one term is the arithmetic working out, not a
disappointment — and it is the reason the measurement was owed. The padding
number alone would have predicted almost twice the saving that arrived.

## The loss difference is not a result

1.2190 against 1.2322 is one seed and one epoch. Bucketing changes which cases
share a gradient step, so a difference in either direction is expected and
this sample cannot tell it from noise. **Do not read it as bucketing helping
accuracy.** If it matters, it needs the same four-seed treatment as anything
else here.

## What is still owed

A.5's done-condition is a four-seed certification with bucketing on, beside
this wall clock. That has not run: HelpSteer2 has failed three times, most
recently to a VM reclamation, and `docs/gpu-access.md` explains why a run
longer than a session's idle window cannot finish in this environment at all.

## Reproduce

```python
from trigon.training import TrainingConfig, train
train(backend, cases, TrainingConfig(bucket_window=1))   # off
train(backend, cases, TrainingConfig(bucket_window=8))   # on, the default
```
