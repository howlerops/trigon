# Release: trigon Banking77 intent model v2 — robust to injected instructions

The model the Modal deployment (`scripts/modal_serve.py`) serves since
2026-09-28, build name **`trigon-qwen2.5-1.5b-0.1.0+4e0792e7`**. It is the
certified Banking77 recipe trained with the adversarial + paired stream
(`reports/paired/README.md`, *Without negation*), the first configuration to
certify on four seeds under `injection_robustness` (Q34) and
`accuracy_over_chance` (Q35). It replaces v1, which loses 0.42 accuracy to one
injected sentence and no longer certifies.

Seed 1 of four: the median-accuracy seed, with the four-seed tie broken on
calibration (ECE 0.0175 against seed 2's 0.0318). Never the best draw.

| | |
| --- | --- |
| Backbone | `Qwen/Qwen2.5-1.5B` at revision `8faed761d45a263340a0528343f099c05c9a4323`, frozen bf16. **Not included**: fetched and pinned by `trigon.backends.hub` |
| This artifact | LoRA rank 16 plus the readout heads (`adapter.pt`, 90 MiB), then a temperature and an isotonic map |
| Trained at | commit `019425daee66`, clean tree, Modal `NVIDIA A10G`, lr 1e-4, 4 epochs, `--paired-mix 0.5 --paired-kinds injection,padding,paraphrase --consistency-weight 0` (`modal-run.json`) |
| Loads with | the main branch at or after the merge that adds this directory |

## What it scores (held out, 5,000 cases)

| | Uncalibrated | **Calibrated** |
| --- | ---: | ---: |
| Accuracy | 0.8756 | **0.8762** |
| ECE | 0.0477 | **0.0175** |
| Adaptive ECE | 0.0476 | **0.0235** |
| Brier | 0.1912 | **0.1918** |

A perfectly calibrated model scores ECE 0.0074 on this evaluation set, p95
0.0118, so 0.0175 is a real, small miscalibration and not noise. The marginal
predictor scores 0.0166; the model closes 87% of the distance from it to
perfect. Every release gate passes.

**Under attack** (1,000 held-out pairs each, templates training never saw):

| | v1 (undefended) | **v2** |
| --- | ---: | ---: |
| Accuracy with an instruction naming a wrong answer | 0.476 | **0.853** |
| Drop from the clean case (`injection_robustness`, limit 0.10) | 0.42 | **0.035** |
| Accuracy with 16 lines of irrelevant state | 0.064 | **0.754** |

The price is clean accuracy: the configuration's four-seed median is 0.8859
against v1's 0.9009. Read the whole spread in `reports/paired/README.md`.

**What it is not.** A general intent model: it knows Banking77's 77 intents,
calibrated on Banking77's distribution. Robust to the injection and padding
styles it was measured on, not to every attack: training and evaluation
templates share no wording but share a style. It does not answer negated
questions ("is it *not* X?") better than chance; negation was never learned
and is not trained. Its four-seed certification rests on the calibrator being
accepted on each seed; a fifth seed has not been run. On your own traffic,
refit the calibrators on your own labels with `trigon fit`.

## Use it

```bash
pip install -e ".[server,train]"        # from the main branch
TRIGON_TEMPERATURE_PATH=temperatures.json TRIGON_ISOTONIC_PATH=isotonic.json \
  trigon serve --backend torch --weights adapter.pt
```

The first load fetches the pinned backbone (3 GB) into `TRIGON_WEIGHTS_CACHE`.
`/healthz` reports whether the calibrators loaded.

## Licences and attribution

- **Qwen2.5-1.5B**, Alibaba Cloud: Apache-2.0. The adapter is a derivative of
  its weights.
- **Banking77 (Casanueva et al., 2020), PolyAI: CC BY 4.0.**
  https://github.com/PolyAI-LDN/task-specific-datasets. This is the data it
  was trained and evaluated on, and the source of every paired variant.

`SHA256SUMS` covers every file in the release bundle. Check with `sha256sum -c SHA256SUMS`.

## Where the files are

- `adapter.pt`: the Modal Volume `trigon-runs`, at
  `banking77-paired-qwen15b-mix05-nonneg-cw0-019425daee66-20260928T172328/seed1.pt`,
  the path `scripts/modal_serve.py` serves from. The whole bundle is at
  `releases/banking77-qwen15b-v2.tar.gz` on the same Volume (sha256 in
  `BUNDLE.sha256`). It becomes a GitHub Release through
  `.github/workflows/release.yml` with `release: banking77-qwen15b-v2`, which
  refuses the bundle unless it matches `BUNDLE.sha256`.
- `temperatures.json`, `isotonic.json`, `report.md`, `report.json`:
  `reports/paired/paired-qwen15b-mix05-nonneg-cw0-seed1-*` and
  `reports/paired/paired-qwen15b-mix05-nonneg-cw0-seed1.{md,json}`, byte for byte.
- `modal-run.json`: `reports/paired/paired-qwen15b-mix05-nonneg-cw0-modal-run.json`.
