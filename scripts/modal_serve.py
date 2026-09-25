"""Serve the certified Banking77 model behind the real gateway, on Modal.

    modal deploy scripts/modal_serve.py

The model is `reports/banking77/`'s certified configuration -- Qwen2.5-1.5B
with LoRA, lr 1e-4 -- seed 2, the median-accuracy seed and one whose
calibrator was applied. Its adapter is read from the `trigon-runs` Volume
where training wrote it, and its temperatures and isotonic map ship from the
repository beside the report they were fitted for.

**It is the gateway, not a second serving path.** `trigon.server.app.build_app`
configured through the same `TRIGON_*` environment variables `trigon serve`
reads, so `/v1/systemone`, `/compat`, `/healthz`, auth, rate limits and the
schema KV cache are the ones the tests cover. A deployment that answered
through its own code would be a second thing to keep honest.

**Closed twice, and the outer lock is the one that saves money.**

1. *Modal proxy auth* (`requires_proxy_auth=True`). A request without a
   `Modal-Key` / `Modal-Secret` pair -- a proxy auth token created in the
   Modal dashboard -- is refused at Modal's edge with a 401 **before any
   container starts**. The gateway's own key check could only run inside a
   container, so on a public URL every stranger's request used to be able to
   wake an A10G to be told no. That was tolerable while the repository was
   private and is not once the URL is published in `reports/`.
2. *The gateway's API key.* `TRIGON_API_KEYS` comes from the
   `trigon-serve-auth` Modal Secret, so a caller holding the Modal token still
   needs a trigon key, and the rate limit (`TRIGON_RATE_PER_MINUTE`) is per
   key. The two are different credentials on purpose: the proxy token belongs
   to whoever may reach the deployment, the API key to whoever may use it.

**Spend is capped, not merely scaled.** At most `MAX_CONTAINERS` GPUs, each
taking eight concurrent requests; a flood waits in Modal's queue rather than
buying hardware. It scales to zero when idle: a GPU that nobody is calling
costs nothing, and the first request after a quiet spell pays a cold start of
loading 3 GB of backbone.

Calling it:

    curl -H "Modal-Key: $MODAL_PROXY_KEY" -H "Modal-Secret: $MODAL_PROXY_SECRET" \
         -H "Authorization: Bearer $TRIGON_API_KEY" https://<workspace>--trigon-serve-gateway.modal.run/healthz

or `trigon_client.Client(..., headers={"Modal-Key": ..., "Modal-Secret": ...})`.
"""

from __future__ import annotations

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
RUN = "banking77-qwen15b-e4-lr1e-4-872ed726edcd-20260923T145427"
SEED = 2
# The spend ceiling: the most A10Gs this deployment can hold at once, whatever
# arrives. Two is one warm container plus headroom for a burst.
MAX_CONTAINERS = 2
REPORT = f"/root/trigon/reports/banking77/qwen15b-e4-lr1e-4-seed{SEED}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
        "pydantic>=2",
        "fastapi>=0.110",
        "uvicorn>=0.27",
        "pyyaml>=6",
        "safetensors>=0.4",
        "regex>=2023",
    )
    .env({"TRIGON_WEIGHTS_CACHE": "/weights/backbones", "PYTHONPATH": "/root/trigon/src"})
    .add_local_dir(
        REPO,
        "/root/trigon",
        ignore=["**/.venv/**", "**/corpora/**", "**/.git/**", "**/.claude/**", "_site/**", "**/__pycache__/**", "**/*.pt"],
    )
)

app = modal.App("trigon-serve")
runs = modal.Volume.from_name("trigon-runs")
weights = modal.Volume.from_name("trigon-weights")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/runs": runs, "/weights": weights},
    secrets=[modal.Secret.from_name("trigon-serve-auth")],
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def gateway():
    os.environ.update(
        {
            "TRIGON_BACKEND": "torch",
            "TRIGON_WEIGHTS": f"/runs/{RUN}/seed{SEED}.pt",
            "TRIGON_TEMPERATURE_PATH": f"{REPORT}-temperatures.json",
            "TRIGON_ISOTONIC_PATH": f"{REPORT}-isotonic.json",
            "TRIGON_RATE_PER_MINUTE": os.environ.get("TRIGON_RATE_PER_MINUTE", "120"),
        }
    )
    from trigon.server.app import build_app

    return build_app()
