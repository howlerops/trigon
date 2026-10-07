# Which model, and how to call it

Three published models, all Apache-2.0, every training corpus green-tier
(`docs/data.md`). Two are hosted behind one URL and chosen per request; all
three can be self-hosted.

| Model | Where | Public decision benchmark | Time in the model | Use it for |
| --- | --- | ---: | --- | --- |
| **`qwen3-4b-lms-agents` v1** (4B) | hosted default | 0.735 | ~80 ms short, ~230 ms agent step (L40S) | routing, moderation, guardrails, agent steps, agent-trajectory safety |
| **`qwen3-14b-lms-safe` v1** (14B) | hosted, `"model": "trigon-large"` | 0.756 | ~120 ms short, ~400 ms agent step (L40S) | knowledge-heavy questions; the most accurate |
| **`qwen3-0.6b-lms-safe` v1** (0.6B) | self-hosted CPU, `docker compose up` | 0.562 | ~0.3 s short on a CPU | no GPU, offline |

For reference on the same 8,016 cases: the incumbent 0.838, an open distilled
model 0.703, a compatible hosted service 0.486 (`reports/decision-bench/`).
Every hosted model answered as its evaluated bundle before it was served
(`reports/parity/`).

## Hosted

`POST https://trigon.jacobbeck-dev.workers.dev/v1/decide`, `Authorization: Bearer <key>`.
Omit `model` (or send any other name) for the default; send
`"model": "trigon-large"` for the 14B. The response's `model` names the build
that answered. Each tier's GPU sleeps after five idle minutes; the first
request after that waits for it to start (30--75 s).

```bash
curl -s https://trigon.jacobbeck-dev.workers.dev/v1/decide \
  -H "Authorization: Bearer $TRIGON_KEY" -H "Content-Type: application/json" \
  -d '{"model": "trigon-large",
       "state": "A 45-year-old with crushing chest pain radiating to the left arm.",
       "questions": {"dx": {"type": "choice", "instructions": "Most likely diagnosis?",
                            "options": [{"name": "myocardial infarction"}, {"name": "gastritis"}]}}}'
```

The incumbent's request shape is served at the incumbent's own path at the
Worker's root (`docs/compat.md`); its `model` field chooses the tier the same way.

## Self-hosted

The container fetches a published bundle and checks every file against one
pinned digest of its `SHA256SUMS` before serving it (`docker/entrypoint.py`):

| Model | `TRIGON_BUNDLE_BASE` | `TRIGON_BUNDLE_SUMS_SHA256` |
| --- | --- | --- |
| 0.6B | `https://huggingface.co/jacobbeckdev/trigon-qwen3-0.6b-lms-safe/resolve/a7c2f780afb324ca967f5606a197015fdcb944fa/` | `6ba0d6e66461b4ac9435f9c6160109b7eae5c337a8b6683b0af4f022c1641da2` |
| 4B | `https://huggingface.co/jacobbeckdev/trigon-qwen3-4b-lms-agents/resolve/12e0c30a32fe88012b7c7d3e0d8072c1a580a6d9/` | `e73c96ef89a2f0faeee69d59d6ff006f24565735a0b479b9c4751387ff91125e` |
| 14B | `https://huggingface.co/jacobbeckdev/trigon-qwen3-14b-lms-safe/resolve/f0f80d3a463991bef79c249c4091d8ad5d64719c/` | `9bc924690b701dcc55b88b8a6972456a72b5b35ee97dc70f4c7f5effb93709eb` |

`docker compose up` serves the 0.6B on a CPU. The 4B wants ~10 GB of GPU
memory and the 14B ~32 GB: run them with a CUDA PyTorch image and
`TRIGON_DEVICE=cuda`, or without Docker:

```bash
pip install -e ".[server,train]"
TRIGON_BACKEND=torch TRIGON_WEIGHTS=bundle/adapter.pt \
  TRIGON_TEMPERATURE_PATH=bundle/temperatures.json TRIGON_DEVICE=cuda trigon serve --port 8000
```

(`TRIGON_ISOTONIC_PATH=bundle/isotonic.json` too, for a bundle that has one.)

## What each is weaker at

- The 4B gives up a little calibration on held-out tasks for its agent-safety
  gain: CLINC150 ECE 0.115 against the previous default's 0.081, BoolQ 0.046
  against 0.017 (Banking77 improved, 0.071 to 0.029).
- Knowledge-heavy exam questions trail the incumbent on every model: MMLU-Pro
  0.50 on the 14B against 0.81.
- The 0.6B judges long agent transcripts near chance.
