# Digit tokens × head repair, and the run that did not reproduce

The 2×2 that was supposed to settle `size`. It settled something else.

| | whole-number tokens | digit tokens |
| --- | --- | --- |
| collapsing Score head | chance, 8 observations | chance, 4 seeds |
| `--score-residual` | chance, 3 seeds | **+0.2425** then −0.0123 |
| `--score-head linear` | chance, 1 seed | chance, 2 seeds |

## What happened to the +0.2425

It was measured on seed 1 of digit tokens plus the Score residual, and it is
the only time in this project that `size` has beaten its own marginal
predictor. Re-run with the same seed and the same flags, it gives **−0.0123**.

The vocabulary changed underneath it. `scripts/train_tokenizer.py` trained on
`docs/*.md`, so the vocabulary moved 5,635 → 4,712 → 6,392 → 4,776 across one
working session, driven by documentation. The +0.2425 was measured against
4,712; the re-run happened on 4,776. Every checkpoint records the vocabulary it
was trained against and refuses to load under a different one — but a *training
run* does not refuse anything, it just quietly trains a different model.

So the result is one unreproducible observation under a vocabulary that no
longer exists. By this project's own rule — a single-seed run is one sample
from a distribution nobody measured — it was never evidence, and treating it as
a breakthrough was the mistake the rule exists to prevent. The corpus no longer
depends on prose, and `reports/stable/` re-runs the arm across four seeds on a
vocabulary that only moves when the data moves.

## What still stands

The diagnosis that produced the experiment is unaffected, because it rests on
things that reproduce:

- `size` emits a **near-constant** answer, sd 0.019 across the whole input
  range. A model that cannot read its input guesses; a head whose keys have
  converged says one thing forever. That distinguishes two failures and it is
  stable.
- `size` is a Score, and a Score ran the dot-product head **without** the
  residual — the configuration this repository documents as collapsing to the
  marginal, exempted from its own repair on the strength of one pooled-metric
  measurement.
- `open_tickets` (13 values) has its threshold learned where `seats` (500
  values) does not, in the same run.

What is *not* established is that either repair fixes it. Four seeds are
running on a stable vocabulary; until they land, `size` is unlearned on every
reproducible configuration this project has tried.
