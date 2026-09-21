# The Score residual, on a vocabulary that stays still

The arm that settled it. `--score-residual` extends to Score heads the repair
that fixed Choice's dot-product collapse, run across four seeds on a vocabulary
that no longer moves when documentation is written.

| Seed | `size` | Its marginal | Lift | `at_risk` lift | `plan` lift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.2415 | 0.2565 | −0.0150 | +0.0948 | +0.3950 |
| 1 | 0.2438 | 0.2558 | −0.0120 | +0.0848 | +0.3837 |
| 2 | 0.2602 | 0.2613 | −0.0012 | **+0.0000** | +0.5940 |
| 3 | 0.2520 | 0.2660 | −0.0140 | +0.0940 | +0.1895 |

**`size` is at chance on all four.** The repair does not fix it.

That also closes the +0.2425. It was measured on this same intervention, one
seed, against a 4,712-token vocabulary that a later documentation commit
destroyed — `scripts/train_tokenizer.py` trained on `docs/*.md` until it was
fixed. On the vocabulary that exists, the same seed and flags give −0.0123.
One unreproducible observation under conditions that no longer exist is not a
result, and by this project's own rule it never was one.

## What is not a negative result

Three of four seeds certify with zero blocking failures, ECE 0.0095–0.0412, and
`at_risk` is learned at +0.09 on three of them. Seed 2 fails one gate and is
the seed where `at_risk` sits exactly on its marginal — the collapse signature,
on a different head.

`plan` is learned on every seed but its lift ranges from +0.1895 to +0.5940,
which is a threefold spread on a question the model can plainly do. That spread
is its own finding and is the reason this project certifies on a sweep rather
than a run.
