# Self-hosting: one service, no hosted dependency

Trigon runs wherever Python or Docker does. Nothing in serving needs Modal, a
GPU or any hosted API: Modal was only ever where training rented GPUs and one
optional deployment target (`scripts/modal_serve.py`). Every number in
`reports/generality/` was produced by the gateway running in process on a
laptop CPU.

## What a model is

A **bundle** is a directory:

| File | What it is |
| --- | --- |
| `adapter.pt` | the trained LoRA and its readout; names its kind, its backbone and the revision it was trained on. An LM-score adapter (`scripts/train_lm_score.py`) scores each answer as the backbone's own tokens; a readout-head adapter (`scripts/train_mix.py`) uses trained heads. The server reads which from the file |
| `temperatures.json`, `isotonic.json` | the calibrators, fitted on held-out data; either may be absent when none was accepted |
| `mix.json` | which corpora, how many cases, what was held out, what the calibrator was fitted on |
| `README.md`, `SHA256SUMS` | the generated model card and the checksums of every other file |

**Two hosted tiers behind one URL**, chosen by the request's `model`: the
default (`trigon-workhorse`, or anything else) is `qwen3-4b-lms-agents` v1
(public benchmark 0.735, agent-trajectory safety 0.840; `docs/models.md`);
`"model": "trigon-large"` is **`qwen3-14b-lms-safe` v1** (Hugging Face
`jacobbeckdev/trigon-qwen3-14b-lms-safe`), for knowledge-heavy questions --
public benchmark 0.756, 0.668 on its held-out slices, 117 ms in the model for
a 50-option request and 399 ms for an agent step on its own L40S.

**The current default model is `qwen3-4b-lms-safe` v1** (Hugging Face
`jacobbeckdev/trigon-qwen3-4b-lms-safe`; R2 `bundles/qwen3-4b-lms-safe/v1/`):
public decision benchmark 0.708 (0.602 on its seven held-out slices), CLINC150
0.930, BoolQ 0.843, Mind2Web step success 0.622 (`reports/decision-bench/README.md`).
`qwen3-4b-lms-yn` v1 is its predecessor without the safety corpora. Both want a
GPU for agent-sized requests. **For a CPU, `qwen3-0.6b-lms-safe` v1** (Hugging
Face `jacobbeckdev/trigon-qwen3-0.6b-lms-safe`; R2 `bundles/qwen3-0.6b-lms-safe/v1/`):
the same recipe on a 0.6B backbone, the default of `docker-compose.yml`.

The backbone (Qwen3, Qwen2.5 or MiniCPM5, all Apache-2.0) is not in the bundle.
It is fetched from Hugging Face at its pinned revision on first start and
cached (`TRIGON_WEIGHTS_CACHE` for readout-head bundles, `HF_HOME` for LM-score
ones).

## Run it

**Quickest, no GPU:** `docker compose up` serves the published CPU model,
`qwen3-0.6b-lms-safe` v1, fetched from Hugging Face at a pinned revision and
checked file by file. It answers on `:8000` -- `POST /v1/decide`. Public
decision benchmark 0.562 (the hosted service 0.486), CLINC150 0.812, BoolQ
0.699, Mind2Web step success 0.457. A CPU server answers exactly as the bundle
does in process (`reports/parity/qwen3-0.6b-lms-safe-v1-cpu/`).

Without Docker:

```bash
pip install -e ".[server,train]"
TRIGON_TEMPERATURE_PATH=bundle/temperatures.json TRIGON_ISOTONIC_PATH=bundle/isotonic.json \
  trigon serve --backend torch --weights bundle/adapter.pt
```

With Docker, the decision service alone (a bundle mounted at `/model` is served
as it is; `TRIGON_BUNDLE_URL` with `TRIGON_BUNDLE_SHA256` fetches and verifies
a published one instead):

```bash
docker build -t trigon:local .
docker run -p 8000:8000 -v "$PWD/bundle:/model:ro" -v trigon-cache:/cache trigon:local
```

Or the whole stack, decisions and free text, with `docker compose up`
(`docker-compose.yml`): trigon on `:8000`, and ollama on `:11434` with an
OpenAI-compatible `/v1` for the text a decision model deliberately never
writes.

`/healthz` reports whether the calibrators loaded, whether the schema cache is
on and whether the int8 path is serving -- check it before trusting a number.

## A cheap test endpoint

`deploy/fly/` runs the image on Fly.io with the bundle baked in, the backbone
cached on a volume, an API key required and machines stopped when idle -- the
cost is the seconds a machine is awake. Measured on `shared-cpu-4x`, 8 GB:
0.30 s round trip for a warm two-question request from Arizona to `lax`, 0.69 s
on the first. Cloudflare Containers is the other scale-to-zero option
(`standard-3`: 2 vCPU, 8 GiB, billed per 10 ms awake); its disk does not
persist, so the backbone would be baked into the image.

## Where the models live

A bundle is published to a Hugging Face model repository (revision-pinned,
with a model card naming every corpus and its licence) and mirrored to
Cloudflare R2, which charges nothing for egress, with the same SHA-256 checked
by `docker/entrypoint.py` from either. The backbone is never re-published: it
is fetched from its own repository at the pinned revision.

## Hosted: Cloudflare in front, a GPU behind

For agent-sized requests a GPU is the difference between the incumbent's per-request latency
and seconds per step. The published 4B answers a 50-option request in 79 ms and
a ~1,300-token web-action step in 229 ms in the model on an L40S (562 ms on an
L4, the same cost per busy request); the 0.6B readout-head model took 6.55 s on
a Fly `shared-cpu-4x`. The hosted shape:

```
caller ──► Cloudflare Worker (deploy/cloudflare)       ──► Modal Server (scripts/modal_gateway.py)
           API keys, CORS, 503 retry, Smart Placement       L40S, scale to zero, ≤2 containers,
           GET /models/* from R2 (trigon-models)            low-latency regional router, proxy auth
```

* **The Worker** checks the caller's key in constant time before anything is
  forwarded, holds the Modal proxy token and the gateway key as secrets, and
  retries the Server's 503s for up to 90 s, so a request that wakes a GPU from
  zero is one slow answer rather than an error. `/healthz` answers at the edge
  unless `?deep=1` from an authorised caller.
* **The Modal Server** runs the same gateway under uvicorn behind Modal's
  low-latency router. The general web-function ingress cost ~450 ms of a
  ~515 ms round trip; the Server, kept alive, ~226 ms with ~90 ms of it the
  model. It drains five minutes after its last request.
* **R2** holds every published bundle (`scripts/publish_bundle.py`) at an
  immutable `bundles/<name>/<version>/` path with `SHA256SUMS` and a model card,
  and charges nothing for downloads.

Measured end to end (Arizona, 2026-10-05, the 4B on an L40S): 223 ms median
through the Worker for a small request on a kept-alive connection, 188 ms
direct to Modal, 73 ms of it the model -- the rest is the network to the GPU's
region. A fresh TLS connection per request adds ~50 ms. The first request from
zero takes ~75 s. **Every deployment is checked before it counts:**
`scripts/serving_parity.py` sends the same requests to the Worker and to the
bundle in process and fails unless the answers agree
(`reports/parity/`). Deploy:

```bash
TRIGON_BUNDLE=<name>-<version> modal deploy scripts/modal_gateway.py   # TRIGON_GATEWAY_GPU to change the GPU
TRIGON_GATEWAY_APP=trigon-gateway-large TRIGON_BUNDLE=<large bundle> modal deploy scripts/modal_gateway.py
# secrets: CLIENT_KEYS, GATEWAY_KEY, MODAL_KEY, MODAL_SECRET. COMPAT_PATH puts the incumbent's
# path at the Worker's root, so a client changes only its base URL; it is never written here.
cd deploy/cloudflare && npx wrangler deploy --var COMPAT_PATH:$(python -c \
  "from trigon.server.compat_path import COMPAT_PATH; print(COMPAT_PATH)")
TRIGON_PARITY_KEY_FILE=<client key file> python scripts/serving_parity.py \
  --bundle <bundle dir> --url https://<worker> --out reports/parity/<name>
```

## Two front doors

* `POST /v1/decide` -- the native contract (`spec/openapi.json`).
* `POST /compat/<the incumbent's path>` -- the incumbent's request and
  response shapes (`docs/compat.md`). `trigon serve --compat` puts that path at
  the root instead, so a client that lets you change only its base URL works
  unmodified.

## A browser agent on the incumbent's API

The published browser agents built on the incumbent ask one request per step:
an `operation` Choice and one target Choice per operation over the page's
elements, with JSON-object `instructions` and criteria. The compat route
accepts that shape as sent and returns the `choice`, `probabilities` and
`confidence` such an agent validates. To run one against this stack:

1. Point its decision call at `http://localhost:8000/compat/<the incumbent's path>`
   (where the agent hard-codes the incumbent's URL, that one line).
2. Point its text helper at ollama: an OpenAI-compatible base URL of
   `http://localhost:11434/v1` and a local model, e.g. `qwen3:30b`.

`reports/webact/` scores this workload on real websites (Mind2Web): operation
and element accuracy, the step an agent would execute, its calibration and its
latency.

## CPU and memory

On an Apple M-series CPU, float32 and four threads, the 0.5–0.6B models answer
a request in ~0.3–0.7 s p50 when every request carries a new schema; repeated
schemas hit the prefix cache. `TRIGON_INT8=1` is slower on Apple silicon and
unmeasured on x86 (`reports/generality/README.md`, *CPU serving*). Give the
container ~4 GB for a 0.6B model in float32, and keep other large models out
of the same unified memory while it trains: an 18 GB ollama model alongside a
training run made it 5× slower.
