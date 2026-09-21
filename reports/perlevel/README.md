# A readout slot per Score level — and the end of the `size` investigation

`--score-readout-per-level` was the last structural hypothesis on the list in
`reports/capacity/README.md`:

> One slot is evidently not fatal on its own, or `at_risk` would fail too. One
> slot carrying two bits might be. `--score-readout-per-level` gives a Score
> the same layout the Choice head has, and is the last structural hypothesis
> in this list.

Four seeds, otherwise the certified configuration: 8,000 cases, 8 epochs,
d_model 128, 2 layers, noise 0.2, on the prose-independent vocabulary.

| Seed | `size` | its marginal | lift | baseline arm |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.2510 | 0.2565 | −0.0055 | −0.0150 |
| 1 | 0.2480 | 0.2558 | −0.0078 | −0.0120 |
| 2 | 0.2490 | 0.2613 | −0.0123 | −0.0012 |
| 3 | 0.2548 | 0.2660 | −0.0112 | −0.0140 |

Median −0.0095 against a baseline median of −0.0130. Better by a hair, below
the marginal predictor on every seed, and nowhere near the +0.05 the gate
wants. **It did not work.**

It also cost something. `at_risk` on seed 0 lands at 0.6672 against a marginal
of 0.6672 — exactly its marginal, to four places, which is the collapse
signature this project has learned to recognise. The baseline arm had
`at_risk` passing on three seeds of four; this arm has it passing on three of
four too, but the failing seed changed and the margin did not improve. Giving
a Score more slots did not take anything from the Noul on the other three
seeds, and on seed 0 it is hard to argue it helped.

## Seven interventions

This is the whole record, and it is the reason `size` is now documented as out
of reach rather than pursued further:

| Intervention | `size` lift, per seed |
| --- | --- |
| None (the certified arm) | −0.0062, −0.0112, −0.0023, −0.0230 |
| One token per digit | −0.0055, −0.0112, −0.0022, −0.0128 |
| Depth: 128×4 | −0.0025, −0.0182 |
| Capacity: 256×4 | −0.0055, −0.0070 |
| A fixed-width linear Score head | −0.0172, −0.0098 |
| The dot-product head's residual repair | −0.0150, −0.0120, −0.0012, −0.0140 |
| A readout slot per level | −0.0055, −0.0078, −0.0123, −0.0112 |

Twenty-two runs. Not one of them puts `size` above its own marginal predictor
on any seed. The spread within an intervention is larger than the difference
between interventions, which is the same finding this project keeps making
about itself: at this scale the seed decides more than the change does.

## What this does and does not establish

**Established, by measurement:** a threshold over 500 values is not learned by
a 128-wide two-layer prefill-only model on 8,000 synthetic cases, and it is not
learned by any of the six variations tried on that model. `at_risk` — a
conjunction plus a threshold over 13 values — *is* learned on most seeds, so
the failure is not "thresholds" and not "arithmetic is a non-goal", both of
which were believed here and neither of which survived contact with that
control.

**Not established:** that the architecture cannot learn it. Every run in the
table above shares a 128-wide two-layer backbone that was called a spike in the
first commit and has been the confound under every finding since. The honest
statement is the narrow one: *this* model does not learn *this* question, and
six attempts to fix it inside that model failed.

`docs/plan.md` Stage 1.3 is the experiment that would settle it, and it is
sequenced after this one precisely so the diagnosis is not confounded by a
backbone change. Until then `size` stays in **Open**, with a number under it
instead of a theory.
