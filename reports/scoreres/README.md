# The Score residual, first measurement

The arm that found it. `--score-residual` extends to Score heads the repair
that fixed Choice's dot-product collapse, and this is the run where `size`
first beat its own marginal predictor.

| Seed | `size` accuracy | Its marginal | Lift |
| ---: | ---: | ---: | ---: |
| 0 | 0.2567 | 0.2565 | +0.0002 |
| 1 | **0.5008** | 0.2582 | **+0.2425** |

Against a control of −0.0055 and −0.0112 on the same tokenizer, and against
every prior configuration, all of which sat on the marginal.

**These two runs used the digit tokenizer**, which is reverted. That does not
undermine them — the control they are measured against used it too, so the
residual is the only variable — but it does mean the absolute numbers are not
comparable with runs on the shipped vocabulary. `reports/fix/` repeats the arm
on the reverted tokenizer across three seeds, which is what the decision rests
on.

Seed 0 is a bad draw rather than a refutation: its `plan` also collapsed to
chance (−0.0005), so nothing on that seed learned anything. What the pair
establishes is that the head *can* learn this question, which eight prior
measurements said it could not.
