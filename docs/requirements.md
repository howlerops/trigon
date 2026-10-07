# Requirements, and where each one stands

What "a drop-in alternative" has to mean, each requirement with the bar it is
held to and the evidence for where it stands. The incumbent's figures are from
its own published model page (version 1.13, read 2026-10-02); ours are from
`reports/`. Reviewed after every training run -- a run that moves a row moves
it here, and a row is only **met** on measured evidence.

| # | Requirement | The bar | Where it stands | Evidence |
| --- | --- | --- | --- | --- |
| 1 | **Wire-compatible** | a client changes only its base URL | **met** | `/compat` takes the incumbent's request as a real browser agent sends it -- object-valued instructions and criteria -- and returns the `choice`, `probabilities` and `confidence` it validates (`docs/self-host.md`); hosted, the Worker serves it at the incumbent's own path at its root (a base-URL change only), keys checked before the GPU is reached |
| 2 | **Reads the caller's question** | accuracy on option sets and wordings it was not trained on | **met** | published 4B: shuffled 50-option Banking77 0.862, renamed 0.838, order agreement 0.937, against a compatible service's 0.544, 0.395 and 0.64 (`reports/generality/lms-yn-q3-4b`) |
| 3 | **Held-out task accuracy** | at a compatible service's level on tasks no mix trains | **met** | the `trigon-large` tier (Qwen3-14B): 0.668 on the public benchmark's seven held-out slices against the hosted service's 0.362, an open distilled model's 0.659 and the incumbent's 0.838; CLINC150 0.942, BoolQ 0.892. The default 4B: 0.602, its recipe certified on three seeds (`reports/decision-bench/README.md`) |
| 4 | **Calibrated** | ECE within its noise floor | **met on held-out; not on Banking77** | published 4B: CLINC150 0.021 within its 0.028 floor, BoolQ 0.044 at 0.037; Banking77 0.059–0.077 against ~0.03, where no Choice calibrator was accepted |
| 5 | **Agent workload** | an agent's executed step right as often as the incumbent | **ahead of a compatible service; three seeds** | Mind2Web step success 0.597–0.617, median 0.600, on websites never trained on (a compatible service 0.050); the published safety model 0.622 (`reports/webact/`) |
| 6 | **Latency** | per-request at the incumbent's ~178 ms median | **met in the model; not end to end** | published 4B on an L40S: a 50-option request 79 ms in the model, an agent step (1,315 tokens, several questions) 229 ms; through the Worker 271 ms and 496 ms -- the rest is the Worker-to-GPU hop. Faster precision (merged or bfloat16 LoRA, 393–460 ms on an L4) failed parity and is not served (`reports/parity/`) |
| 7 | **Cost** | at or under the incumbent's $0.042 per million tokens | **met interactive; better batched** | $0.041–0.057/MTok interactive on an L4 (1.5B, batch 1), $0.012 at batch 8, $0.009 at batch 32 (`reports/burn-in/`); the 0.6B models are cheaper and unmeasured |
| 8 | **Runs without a GPU** | a CPU deployment that serves | **met** | `docker compose up` serves the published `qwen3-0.6b-lms-safe` v1 on CPU, fetched from Hugging Face and checked by digest; public benchmark 0.562 against the hosted service's 0.486; a CPU server answers exactly as the evaluated bundle (`reports/parity/qwen3-0.6b-lms-safe-v1-cpu/`). Agent-sized requests take seconds on a CPU; the GPU tier is for those |
| 9 | **Context and limits** | the incumbent's 64k / 32k token envelope | **met** | `COMPAT_BUDGET` reproduces it exactly; ours is a superset. Its per-question 50-option cap and required Noul criteria, observed on a compatible service, are handled outbound |
| 10 | **Independent questions** | adding a question does not move another's answer | **met** | asserted per architecture in `tests/test_independence.py` and `tests/test_qwen_backend.py`, Qwen3 included |
| 11 | **Throughput** | the incumbent's 80 requests / 100K tokens per second per account | **not measured** | two L4 containers at eight concurrent requests each is the current ceiling, by design for cost |
| 12 | **Open and durable** | weights anyone can use, long term | **met** | `qwen3-14b-lms-safe`, `qwen3-4b-lms-safe` (and `-lms-yn`) and `qwen3-0.6b-lms-safe` v1, Apache-2.0, every training corpus green-tier, on R2 behind the Worker and public on Hugging Face, checksums verified through the Worker; each served only after a parity check (`reports/parity/`) |
| 13 | **Broad** | one model across intents, yes/no, scores, documents and web actions | **met** | the published safety model leads the hosted service on 10 of 11 benchmark slices (all but agent-trajectory safety, 0.536 against 0.654) and on every generality task, and leads web actions |

## What to check on every run

1. `scripts/generality.py` on the bundle -- rows 2, 3 and 4. A run that gains
   held-out accuracy by losing order agreement or calibration has not improved.
2. `scripts/webact.py` -- row 5, and the executed step's calibration.
3. `/healthz` after loading -- `calibrated`, `schema_cache`, and the build name
   the card records.
4. The model card (`scripts/publish_bundle.py`) lists every corpus it trained
   on with its licence; a corpus not green in `docs/data.md` is a release
   blocker, not a footnote.
5. One seed is a measurement. A row moves to **met** on the median of several.
