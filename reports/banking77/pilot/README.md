# The Banking77 pilot, and why its ECE is not quoted

Two seeds, 4,000 training cases, 4 epochs, evaluated on 1,500 rows of the
corpus's test split. **The accuracy result is real. The calibration result is
not quotable, and the gates said so before I did.**

| Seed | Accuracy | Marginal | Lift | ECE | Adaptive ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4640 | 0.0180 | **+0.4460** | 0.0324 | 0.0411 |
| 1 | 0.4193 | 0.0167 | **+0.4027** | 0.0351 | 0.0334 |

`accuracy_over_baseline` passes with enormous margin — a 77-way choice where
the marginal predictor scores 1.8%, answered correctly 42–46% of the time by a
128-wide two-layer model. That is the first evidence in this repository that
any of this transfers off the synthetic generator.

Both seeds failed two gates:

```
FAIL sample_size:      1500 (limit 5000) -- below this, ECE is dominated by estimator noise
FAIL gate_is_testable: 0.0348 (limit 0.0250) -- a perfectly calibrated model scores
                       ECE 0.0221 on this run, p95 0.0348
```

A perfectly calibrated model scores 0.0221 here. The measured ECE is 0.0324.
**Most of the gate is the instrument**, so 0.0324 is not a measurement of this
model's calibration — it is a number in the same range as the noise underneath
it. `CLAUDE.md` is explicit about what to do with that: fix the run, not the
gate, and never quote an ECE from a run below `MIN_CALIBRATION_SAMPLES`.

## What fixing the run took, and what it exposed

Banking77's test split is 3,080 rows against a floor of 5,000, so **the plan's
Stage 2 done-condition — "ECE ≤ 0.05 holds per corpus" — is not reachable on
this corpus from its own test split.** That is a defect in the plan, found by
measurement rather than by rereading it.

The evaluation set is now topped up from rows held out of train that neither
training nor calibration sees. They are held-out data by the only definition
that matters, they come from the train distribution rather than the test one,
and the report states how many of each it used instead of summing them.

The first version of that top-up was worse than the problem. It was capped at
`len(remaining) - 1`, so on a corpus with a small train split it reached the
floor by leaving **one** training case. Every count in the report was correct
and the report was worthless. The top-up is now capped at half the pool, and
when half is not enough the floor is not reached and `sample_size` fails —
which is the right answer. A corpus too small to supply both a trainable set
and a floor-sized evaluation cannot be certified by this gate, and
cannibalising training to make it pass is the same move as widening it.

These runs are kept because the accuracy number stands and because the failure
is the useful part. The certification run is in the parent directory.
