# Seed sweeps

One configuration, several seeds, reported as a spread. Produced by
`scripts/seed_sweep.py`; see `docs/decisions.md`, "A single-seed training run is
not evidence", for why this exists.

## `reference-config` — 1,200 cases, 3 epochs, d_model 128, 2 layers

```bash
python scripts/seed_sweep.py --seeds 0 1 2 3 --name reference-config \
  -n 1200 --epochs 3 --lr 0.01 --d-model 128 --layers 2 --noise 0.2 \
  --option-scoring auto --eval-n 2000 --floor-trials 20 --validation-fraction 0
```

| Seed | Final loss | Lift over baseline | ECE | Certified | Blocked on |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1.1399 | +0.0000 | 0.0151 | **no** | `accuracy_over_baseline` |
| 1 | 0.9689 | **+0.1213** | 0.0122 | **no** | `workhorse_adaptive_ece` |
| 2 | 1.0692 | +0.0517 | 0.0095 | **no** | `workhorse_adaptive_ece` |
| 3 | 1.0438 | +0.0510 | 0.0223 | **no** | `workhorse_adaptive_ece` |

Final loss: median 1.0565, range 0.9689–1.1399, **spread 0.1710**. Chance is
1.1552.

**Read the lift column.** It is the gate's own headline metric, its limit is
0.05, and across four seeds of one configuration it takes the values 0.0000,
0.1213, 0.0517 and 0.0510 — from "does not beat a model that ignores the state"
to "beats it by more than twice the limit". Seed 0 fails that gate outright;
the other three clear it. Reporting any one of these as *the* result of this
configuration is reporting a draw.

They all fail to certify, on two different gates, which is its own useful
signal: this configuration is not a candidate, and a single run of it that
happened to pass would not have made it one.

The `.pt` files are deleted after each seed — a sweep is about the distribution,
not about keeping four models.
