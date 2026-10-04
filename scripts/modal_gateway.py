"""Serve any trained bundle behind the real gateway, on a scale-to-zero GPU.

    modal volume put trigon-runs <dir>/adapter.pt bundles/<name>/adapter.pt  # + calibrators
    modal secret create trigon-gateway-auth TRIGON_API_KEYS=<key>
    TRIGON_BUNDLE=<name> modal deploy scripts/modal_gateway.py

The GPU half of the hosted service (`docs/self-host.md`, *Hosted*): a
Cloudflare Worker in front holds the keys and routes, R2 holds the models, and
this answers when a request needs a GPU. `scripts/modal_serve.py` is the same
thing hard-wired to the certified Banking77 model on an A10G; this takes any
bundle by name and runs on an L4, which a 0.6-1.5B model does not outgrow.

Everything that makes `modal_serve.py` safe to leave deployed carries over,
for the reasons it gives: it is the gateway (`build_app` from the same
``TRIGON_*`` variables `trigon serve` reads), Modal's proxy auth refuses a
request at the edge before a container wakes, the gateway's own API key and
per-key rate limit apply behind it, at most ``MAX_CONTAINERS`` GPUs exist at
once, and it scales to zero.

Calling it directly (the Worker does this for its callers):

    modal curl https://<workspace>--trigon-gateway-gateway.modal.run/healthz
"""

from __future__ import annotations

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
#: Which bundle under /runs/bundles/ to serve, fixed at deploy time.
BUNDLE = os.environ.get("TRIGON_BUNDLE", "mix-q3-06b")
GPU = os.environ.get("TRIGON_GATEWAY_GPU", "L4")
#: The spend ceiling: however much arrives, no more GPUs than this.
MAX_CONTAINERS = int(os.environ.get("TRIGON_GATEWAY_MAX_CONTAINERS", "2"))

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
        "httpx2",  # the TestClient `bench` measures the gateway through
        "transformers>=4.51",  # LM-score bundles load their backbone through it
    )
    .env(
        {
            "TRIGON_WEIGHTS_CACHE": "/weights/backbones",
            # An LM-score backbone downloads once, onto the volume, not per cold start.
            "HF_HOME": "/weights/hf",
            "PYTHONPATH": "/root/trigon/src",
            "TRIGON_BUNDLE": BUNDLE,
        }
    )
    .add_local_dir(
        REPO,
        "/root/trigon",
        ignore=[
            "**/.venv*/**",
            "**/corpora/**",
            "**/.git/**",
            "**/.claude/**",
            "**/.omc/**",
            "_site/**",
            "**/__pycache__/**",
            "**/*.pt",
            "sdk/**",
            "site/**",
        ],
    )
)

app = modal.App("trigon-gateway")
runs = modal.Volume.from_name("trigon-runs")
weights = modal.Volume.from_name("trigon-weights")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/runs": runs, "/weights": weights},
    secrets=[modal.Secret.from_name("trigon-gateway-auth")],
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def gateway():
    bundle = pathlib.Path("/runs/bundles") / os.environ["TRIGON_BUNDLE"]
    if not (bundle / "adapter.pt").exists():
        raise RuntimeError(f"no adapter.pt in {bundle}; upload it to the trigon-runs volume")
    env = {
        "TRIGON_BACKEND": "torch",
        "TRIGON_WEIGHTS": str(bundle / "adapter.pt"),
        "TRIGON_RATE_PER_MINUTE": os.environ.get("TRIGON_RATE_PER_MINUTE", "600"),
    }
    for name, variable in (
        ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
        ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
    ):
        if (bundle / name).exists():
            env[variable] = str(bundle / name)
    os.environ.update(env)
    from trigon.server.app import build_app

    return build_app()


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/runs": runs, "/weights": weights},
    timeout=1800,
)
def bench(requests: list[dict], repeats: int = 2) -> dict:
    """The gateway in process on this GPU: per-request latency, no network.

    What a caller sees is this plus the round trip to the Worker and from it to
    Modal; this is the part the model owns, comparable to a published per-request
    model latency.
    """
    import statistics
    import time

    import torch
    from fastapi.testclient import TestClient

    bundle = pathlib.Path("/runs/bundles") / os.environ["TRIGON_BUNDLE"]
    os.environ.update({"TRIGON_BACKEND": "torch", "TRIGON_WEIGHTS": str(bundle / "adapter.pt")})
    if (bundle / "temperatures.json").exists():
        os.environ["TRIGON_TEMPERATURE_PATH"] = str(bundle / "temperatures.json")
    from trigon.server.app import build_app

    client = TestClient(build_app())
    client.post("/v1/decide", json=requests[0])  # warm the kernels
    seconds, tokens = [], []
    for _ in range(repeats):
        for body in requests:
            torch.cuda.synchronize()
            started = time.perf_counter()
            response = client.post("/v1/decide", json=body)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            response.raise_for_status()
            tokens.append(response.json()["usage"]["prefill_tokens"])
    seconds.sort()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "n": len(seconds),
        "p50_ms": round(1000 * statistics.median(seconds), 1),
        "p95_ms": round(1000 * seconds[int(0.95 * (len(seconds) - 1))], 1),
        "prefill_tokens_median": statistics.median(tokens),
        "bundle": os.environ["TRIGON_BUNDLE"],
    }


#: Where requests enter Modal. A Modal Server routes through a low-latency
#: proxy pinned here instead of the general web-function ingress, which cost
#: ~450 ms of every ~515 ms round trip while the model took 64 ms.
ROUTING_REGION = os.environ.get("TRIGON_GATEWAY_ROUTING_REGION", "us-west")
PORT = 8000


@app.server(
    image=image,
    gpu=GPU,
    volumes={"/runs": runs, "/weights": weights},
    secrets=[modal.Secret.from_name("trigon-gateway-auth")],
    port=PORT,
    routing_region=ROUTING_REGION,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    target_concurrency=8,
    # The default drained a ready replica a minute after its first request;
    # five minutes keeps it across the gaps between an agent's steps.
    scaledown_window=300,
    startup_timeout=600,
    exit_grace_period=10,
)
class Gateway:
    """The same gateway as `gateway`, as a process behind Modal's low-latency router.

    Authenticated by Modal by default, like the web function: the Worker's
    proxy token is what reaches it.
    """

    @modal.enter()
    def start(self) -> None:
        import subprocess
        import time
        import urllib.request

        bundle = pathlib.Path("/runs/bundles") / os.environ["TRIGON_BUNDLE"]
        env = dict(
            os.environ,
            TRIGON_BACKEND="torch",
            TRIGON_WEIGHTS=str(bundle / "adapter.pt"),
            TRIGON_RATE_PER_MINUTE=os.environ.get("TRIGON_RATE_PER_MINUTE", "600"),
        )
        for name, variable in (
            ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
            ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
        ):
            if (bundle / name).exists():
                env[variable] = str(bundle / name)
        self.process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "trigon.server.app:build_app",
                "--factory",
                "--host",
                "0.0.0.0",
                "--port",
                str(PORT),
            ],
            env=env,
        )
        deadline = time.monotonic() + 590
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"gateway exited with {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=2):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("gateway did not become healthy")

    @modal.exit()
    def stop(self) -> None:
        self.process.terminate()
