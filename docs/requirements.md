# Requirements, and where each one stands

What "a drop-in alternative" has to mean, each requirement with the bar it is
held to and the evidence for where it stands. The incumbent's figures are from
its own published model page (version 1.13, read 2026-10-02); ours are from
`reports/`. Reviewed after every training run -- a run that moves a row moves
it here, and a row is only **met** on measured evidence.

| # | Requirement | The bar | Where it stands | Evidence |
| --- | --- | --- | --- | --- |
| 1 | **Wire-compatible** | a client changes only its base URL | **met** | `/compat` takes the incumbent's request as a real browser agent sends it -- object-valued instructions and criteria -- and returns the `choice`, `probabilities` and `confidence` it validates (`docs/self-host.md`) |
| 2 | **Reads the caller's question** | accuracy on option sets and wordings it was not trained on | **met on option sets; open on tasks** | shuffled 50-option Banking77 0.729–0.806 against a compatible service's 0.544; renamed 0.653–0.695 against 0.395; order agreement 0.83–0.92 against 0.64 (`reports/generality/`) |
| 3 | **Held-out task accuracy** | at a compatible service's level on tasks no mix trains | **not met** | CLINC150 0.585 against 0.849; BoolQ 0.565 against 0.758 -- the Noul head reads the question's wording but not the passage (correlation with the answer −0.09); WANLI added to fix it |
| 4 | **Calibrated** | ECE within its noise floor | **met on trained tasks; not on held-out** | Banking77 tasks 0.04–0.07 against floors ~0.04; CLINC150 0.11–0.17 and BoolQ 0.11–0.16 against ~0.045 |
| 5 | **Agent workload** | an agent's executed step right as often as the incumbent | **open** | Mind2Web step success 0.038 (no web training) against a compatible service's 0.050; broad model with 2,864 web steps training (`reports/webact/`) |
| 6 | **Latency** | per-request at the incumbent's ~178 ms median | **met in the model; not end to end** | 64–149 ms in the model on an L4; ~226 ms direct to the GPU kept alive, ~310 ms through the Worker; 45.6 s from zero (`docs/self-host.md`, *Hosted*) |
| 7 | **Cost** | at or under the incumbent's $0.042 per million tokens | **met interactive; better batched** | $0.041–0.057/MTok interactive on an L4 (1.5B, batch 1), $0.012 at batch 8, $0.009 at batch 32 (`reports/burn-in/`); the 0.6B models are cheaper and unmeasured |
| 8 | **Runs without a GPU** | a CPU deployment that serves | **met for small requests** | ~0.3 s for a two-question request on a laptop CPU; agent-sized requests take seconds (6.55 s on a Fly shared CPU), which is what the GPU path is for |
| 9 | **Context and limits** | the incumbent's 64k / 32k token envelope | **met** | `COMPAT_BUDGET` reproduces it exactly; ours is a superset. Its per-question 50-option cap and required Noul criteria, observed on a compatible service, are handled outbound |
| 10 | **Independent questions** | adding a question does not move another's answer | **met** | asserted per architecture in `tests/test_independence.py` and `tests/test_qwen_backend.py`, Qwen3 included |
| 11 | **Throughput** | the incumbent's 80 requests / 100K tokens per second per account | **not measured** | two L4 containers at eight concurrent requests each is the current ceiling, by design for cost |
| 12 | **Open and durable** | weights anyone can use, long term | **met; publishing pending** | Apache-2.0 bundles, every training corpus green-tier; R2 at immutable versioned paths with checksums and a generated model card (`scripts/publish_bundle.py`); a Hugging Face listing waits on a token |
| 13 | **Broad** | one model across intents, yes/no, scores, documents and web actions | **training** | the broad Qwen3-0.6B model (eight sources, 19,326 cases) |

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
