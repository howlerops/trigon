# Release: trigon Banking77 intent model — Qwen2.5-1.5B + LoRA, calibrated

This is the certified Banking77 configuration from
`reports/banking77/README.md` on the main branch: seed 2, the median-accuracy
seed of four. It is the same adapter and calibrators the Modal deployment
(`scripts/modal_serve.py`) serves, with build name
**`trigon-qwen2.5-1.5b-0.1.0+cde26cc3`**.

| | |
| --- | --- |
| Backbone | `Qwen/Qwen2.5-1.5B` at revision `8faed761d45a263340a0528343f099c05c9a4323`, frozen bf16. **Not included**: fetched and pinned by `trigon.backends.hub` |
| This artifact | LoRA rank 16 plus the readout heads (`adapter.pt`, format `trigon-backbone-adapter-1`, 89 MiB), then a temperature and an isotonic map |
| Trained at | commit `872ed726edcd`, clean tree, Modal `NVIDIA A10`, lr 1e-4, 4 epochs (`modal-run.json`) |
| Loads with | the main branch at or after merge `4ce4de71b6fe` |

## What it scores (held out, 5,000 cases)

| | Uncalibrated | **Calibrated** |
| --- | ---: | ---: |
| Accuracy | 0.9020 | **0.9038** |
| ECE | 0.0534 | **0.0216** |
| Adaptive ECE | 0.0534 | **0.0216** |
| Brier | 0.1530 | **0.1476** |

A perfectly calibrated model scores ECE 0.0080 on this evaluation set, with a
p95 of 0.0115, so 0.0216 is a real, small overconfidence (+0.015), not
noise. The marginal predictor scores 0.0174. Every release gate passes,
including the per-question and per-primitive gates that now block on
backbone runs. On four seeds the configuration's median accuracy is 0.9009.
Read the whole spread in `report.md` and in the main branch's
`reports/banking77/README.md` rather than this one draw.

**What it is not:** a general intent model. It knows Banking77's 77 intents
and its calibration was measured on Banking77's distribution. On your own
traffic, refit the calibrators on your own labels with `trigon fit` and read
the coverage number.

## Use it

```bash
pip install -e ".[server,train]"        # from the main branch
TRIGON_TEMPERATURE_PATH=temperatures.json TRIGON_ISOTONIC_PATH=isotonic.json \
  trigon serve --backend torch --weights adapter.pt
```

The first load fetches the pinned backbone (3 GB) into `TRIGON_WEIGHTS_CACHE`.
`/healthz` reports whether the calibrators loaded; an uncalibrated server
says so instead of answering quietly.

## Licences and attribution

- **Qwen2.5-1.5B**, Alibaba Cloud: Apache-2.0. The adapter is a derivative of
  its weights.
- **Banking77 (Casanueva et al., 2020), PolyAI: CC BY 4.0.**
  https://github.com/PolyAI-LDN/task-specific-datasets. This is the data it
  was trained and evaluated on.

`SHA256SUMS` covers every file in the release bundle. Check with `sha256sum -c SHA256SUMS`.

## Where the files are

This directory is the release's record in the repository: the model card and
the checksums. The files themselves are:

- `adapter.pt`: the Modal Volume `trigon-runs`, at
  `banking77-qwen15b-e4-lr1e-4-872ed726edcd-20260923T145427/seed2.pt`. That is
  the path `scripts/modal_serve.py` serves from. The whole bundle,
  adapter included, is at `releases/banking77-qwen15b-v1.tar.gz` on the same
  Volume (sha256 `729a396e1dfe7c546c1b8239afc12f5e03661bd0d8d9c669d4baee1f5f086e18`).
  Get it with
  `modal volume get trigon-runs releases/banking77-qwen15b-v1.tar.gz .`
  It is not committed here: `*.pt` is ignored so the repository does not
  carry weights. It becomes a GitHub Release through `.github/workflows/release.yml`
  (run it by hand with `release: banking77-qwen15b-v1`), which fetches the bundle
  from the Volume and refuses it unless it matches `BUNDLE.sha256`.
- `temperatures.json`, `isotonic.json`, `report.md`, `report.json`:
  `reports/banking77/qwen15b-e4-lr1e-4-seed2-*` and
  `reports/banking77/qwen15b-e4-lr1e-4-seed2.{md,json}`, byte for byte.
- `modal-run.json`: `reports/banking77/qwen15b-e4-lr1e-4-modal-run.json`.
