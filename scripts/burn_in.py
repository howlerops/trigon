#!/usr/bin/env python
"""The L4 burn-in: latency, prefill throughput and $/MTok on named hardware.

    python scripts/burn_in.py --usd-per-hour 0.80 --out reports/burn-in/

`docs/next.md` B.1. `$/MTok` is dollars per hour over tokens per second, and
the tokens-per-second half has never been measured: the inherited ~30k prefill
tok/s on an L4, and the ~$0.007/MTok built on it, are arithmetic over
unsourced inputs. This measures the half that can be measured and takes the
price as an input, because the price is a quote rather than a property of the
hardware.

**It measures shapes, not the spike.** The only weights in this repository are
128 wide and two layers deep, and they would serve many times faster than any
backbone this project would ship -- publishing their throughput as the
product's would flatter `$/MTok` by the ratio of the two models' compute.
Latency and throughput depend on a model's shape and not on its weight values,
so each shape below is a randomly initialised encoder at a real backbone's
width and depth. The spike is measured too, labelled as the spike.

Each backbone preset keeps its width, depth and head count, and sets `d_ff` to
1.5x the backbone's intermediate size. The backbones use a gated MLP with
three matrices where `nn.TransformerEncoderLayer` has two, so 1.5x is what
makes the per-token FLOPs equal rather than three-quarters.

**It measures the path that serves.** `Engine.answer` with the schema KV cache
on, as the gateway defaults it (`reports/cache/README.md`), at the Banking77
shape -- 77 options, one Choice -- because that is the certified deployment.
The HTTP layer is left out: it is measured at 2.28 ms a request on its own and
is not what an accelerator changes.

**Two throughputs, because they answer different questions.** *Billed* tokens
are what a caller sends and what `$/MTok` is quoted over: the whole request,
schema included. *Computed* tokens exclude the schema served from the cache.
The gap between them is the cache's saving and is the reason the typed path
can be cheap; quoting only the billed number without saying so would hide
where the saving comes from.

Everything that makes the number reproducible is recorded beside it: the GPU
name and driver from `nvidia-smi`, the torch and CUDA versions, the commit and
whether the tree was clean.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cache_bench import STATES, _request  # noqa: E402

from trigon.engine import Engine  # noqa: E402
from trigon.limits import DEFAULT_LATENCY_TARGET  # noqa: E402

# (d_model, layers, heads, d_ff). d_ff is 1.5x the backbone's intermediate size
# so a two-matrix MLP does the work of its three-matrix gated one.
SHAPES = {
    "spike": (128, 2, 4, 256),
    "0.5b": (896, 24, 14, 7296),  # Qwen2.5-0.5B: 896 wide, 24 layers, 14 heads, 4864 ffn
    "1.5b": (1536, 28, 12, 13440),  # Qwen2.5-1.5B: 1536 wide, 28 layers, 12 heads, 8960 ffn
}


def _hardware(device: str) -> dict:
    import torch

    info: dict = {
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
    }
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        try:
            info["nvidia_smi"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            info["nvidia_smi"] = f"unavailable: {exc}"
    for key, args in (("commit", ["rev-parse", "HEAD"]), ("dirty", ["status", "--porcelain"])):
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
            info[key] = bool(out) if key == "dirty" else out
        except (OSError, subprocess.CalledProcessError):
            info[key] = None
    return info


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def measure(name: str, device: str, args) -> dict:
    import torch

    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend

    d_model, layers, heads, d_ff = SHAPES[name]
    backend = TorchReadoutBackend(
        ReadoutConfig(d_model=d_model, n_layers=layers, n_heads=heads, d_ff=d_ff), seed=0
    ).to(device)
    backend.cache_prefixes = True
    parameters = sum(p.numel() for p in backend.model.parameters())
    engine = Engine(backend, compiler=backend.make_compiler())
    requests = [_request(s, args.options, 0) for s in STATES]

    # Warm, uncounted: the first request on a schema fills the prefix, and the
    # first CUDA calls pay for kernel selection. Timing either is how a
    # throughput number flatters or damns itself by accident.
    for request in requests * args.warmup:
        engine.answer(request)
    _sync(device)

    rows = []
    for batch in args.batch_sizes:
        latencies: list[float] = []
        billed = computed = answered = 0
        started = time.perf_counter()
        while answered < args.requests:
            chunk = [requests[(answered + i) % len(requests)] for i in range(batch)]
            t0 = time.perf_counter()
            responses = (
                [engine.answer(chunk[0])] if batch == 1 else engine.answer_many(chunk, batch)
            )
            _sync(device)
            elapsed = (time.perf_counter() - t0) * 1000.0
            # A batch's requests all wait for the whole pass, so each of them
            # sees its wall clock as latency -- not the per-request share.
            latencies.extend([elapsed] * len(responses))
            for response in responses:
                billed += response.usage.prefill_tokens
                computed += response.usage.prefill_tokens - response.usage.cached_schema_tokens
            answered += len(responses)
        wall = time.perf_counter() - started
        ordered = sorted(latencies)
        row = {
            "batch": batch,
            "requests": answered,
            "p50_ms": statistics.median(latencies),
            "p99_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
            "requests_per_s": answered / wall,
            "billed_tokens_per_s": billed / wall,
            "computed_tokens_per_s": computed / wall,
            "tokens_per_request": billed / answered,
        }
        if args.usd_per_hour:
            row["usd_per_mtok_billed"] = args.usd_per_hour / 3600 / row["billed_tokens_per_s"] * 1e6
        rows.append(row)
        print(
            f"  {name:>5} batch {batch:>3}: p50 {row['p50_ms']:8.2f} ms  p99 "
            f"{row['p99_ms']:8.2f} ms  {row['requests_per_s']:8.1f} req/s  "
            f"{row['billed_tokens_per_s']:>10,.0f} billed tok/s",
            flush=True,
        )
    del backend, engine
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "shape": name,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "d_ff": d_ff,
        "parameters": parameters,
        "rows": rows,
    }


def render(hardware: dict, results: list[dict], args) -> str:
    target = DEFAULT_LATENCY_TARGET
    lines = [
        "# Burn-in",
        "",
        f"**{hardware.get('gpu') or hardware['cpu']}** — "
        f"`{hardware.get('nvidia_smi', 'no nvidia-smi')}`",
        "",
        f"torch {hardware['torch']}, CUDA {hardware['cuda']}, Python {hardware['python']}; "
        f"commit `{(hardware.get('commit') or 'unknown')[:12]}`"
        + (" **(dirty tree)**" if hardware.get("dirty") else ""),
        "",
        f"{args.options} options, one Choice, schema KV cache on, "
        f"{args.requests} requests per row after {args.warmup} warm rounds. "
        "Weights are random at each shape; latency and throughput depend on shape only.",
        "",
        f"Latency target: p50 ≤ {target.p50_ms:.0f} ms, p99 ≤ {target.p99_ms:.0f} ms.",
        "",
        "| Shape | Params | Batch | p50 ms | p99 ms | req/s | billed tok/s "
        "| computed tok/s | $/MTok billed | p50 target |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        for row in result["rows"]:
            usd = f"{row['usd_per_mtok_billed']:.4f}" if "usd_per_mtok_billed" in row else "—"
            lines.append(
                f"| {result['shape']} | {result['parameters'] / 1e6:,.1f}M | {row['batch']} "
                f"| {row['p50_ms']:.2f} | {row['p99_ms']:.2f} | {row['requests_per_s']:,.1f} "
                f"| {row['billed_tokens_per_s']:,.0f} | {row['computed_tokens_per_s']:,.0f} "
                f"| {usd} | {'pass' if row['p50_ms'] <= target.p50_ms else 'FAIL'} |"
            )
    lines += [
        "",
        f"`$/MTok` uses **${args.usd_per_hour}/hour**, an input rather than a measurement."
        if args.usd_per_hour
        else "No `--usd-per-hour` given, so no `$/MTok` is computed.",
        "",
        "The spike row is the reference model's shape and is **not** the product's cost; "
        "the backbone rows are what a shipped model of that size would cost to serve.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="spike,0.5b,1.5b")
    parser.add_argument("--options", type=int, default=77, help="the certified Banking77 shape")
    parser.add_argument("--requests", type=int, default=2000, help="per row")
    parser.add_argument("--warmup", type=int, default=10, help="rounds over the request set")
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--usd-per-hour", type=float, default=None, help="what the box costs")
    parser.add_argument("--device", default="cuda", help="cuda by default: this is a GPU burn-in")
    parser.add_argument("--out", default=None, help="a directory for burn-in.md and .json")
    args = parser.parse_args(argv)
    args.batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA device: a burn-in on the wrong hardware is not a burn-in")
    hardware = _hardware(args.device)
    print(f"burn-in on {hardware.get('gpu') or hardware['cpu']}", flush=True)
    results = [measure(name, args.device, args) for name in args.shapes.split(",")]
    markdown = render(hardware, results, args)
    print()
    print(markdown)
    if args.out:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "burn-in.md").write_text(markdown)
        (out / "burn-in.json").write_text(
            json.dumps({"hardware": hardware, "results": results}, indent=2) + "\n"
        )
        print(f"wrote {out}/burn-in.md and burn-in.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
