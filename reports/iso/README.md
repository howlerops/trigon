# `iso-8k` — the first configuration that certifies on every seed

Four seeds of the 8,000-case configuration, gated with the calibrator selected
per primitive. `../sweeps/README.md` has the narrative and what changed to get
here; this directory is the evidence.

| Seed | ECE | Adaptive ECE | Lift over baseline | Certified |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.0247 | 0.0289 | +0.1614 | **yes** |
| 1 | 0.0235 | 0.0262 | +0.2203 | **yes** |
| 2 | 0.0084 | 0.0162 | +0.2277 | **yes** |
| 3 | 0.0087 | 0.0153 | +0.2141 | **yes** |

## Why these are `-regated.md` and not the training runs' own reports

The weights were trained before two defects in the calibration layer were
found — the selection scoring on plain ECE while the run is gated on both
estimators, and the Noul isotonic map being fitted on `max(p)` and applied to
`P(yes)`. A report written by a training run reflects the calibration rule that
ran at the time, so those reports would contradict this table.

`scripts/regate.py` refits the calibration layer on the saved checkpoint and
re-runs the gates against the same weights, so these reports reflect the rule
as it stands. Each one prints the command that produced it. The training runs'
own reports are gitignored rather than committed, because publishing two
reports about one set of weights that disagree about its calibration is worse
than publishing one.

```bash
python scripts/regate.py reports/iso/iso-seed0.pt --out reports/iso
```

Checkpoints are not committed (`*.pt` is ignored), so reproducing this means
training them first — the command is at the top of each training report, and
`../sweeps/README.md` records the sweep it came from.

## What this does not say

It is a 128-wide two-layer spike on one synthetic generator.
`worst_question_over_baseline` is advisory and still fails on every seed: the
model answers `plan` well and sits near the marginal on the other two.
Certifying on four seeds means this configuration's **calibration** is
reproducible, not that the model is good.
