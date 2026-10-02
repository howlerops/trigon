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
| `adapter.pt` | the trained LoRA and readout heads; names its backbone and the revision it was trained on |
| `temperatures.json`, `isotonic.json` | the calibrators, fitted on held-out data; either may be absent when none was accepted |
| `mix.json` | for a model from `scripts/train_mix.py`: which corpora, how many cases, what was held out, what the calibrator was fitted on |

The backbone (Qwen3, Qwen2.5 or MiniCPM5, all Apache-2.0) is not in the bundle.
It is fetched from Hugging Face at its pinned revision on first start and
cached (`TRIGON_WEIGHTS_CACHE`).

## Run it

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
