#!/usr/bin/env python
"""Drive the gateway concurrently and report the latency distribution.

    python scripts/load_test.py                      # lexical, 8 workers
    python scripts/load_test.py --workers 32 --seconds 20
    python scripts/load_test.py --backend torch --weights reports/run.pt

`scripts/gateway_cost.py` measures one request on an idle box, which is a
median and says nothing about the tail. `docs/decisions.md` leans on that
measurement to drop the planned Rust rewrite, and names its own falsifier:

> p99 under real load showing GC or GIL pauses rather than model queueing.
> The gateway's contribution to p99 is unmeasured — a median on an idle box,
> and tail behaviour under contention is a different question.

This is that measurement. It runs the real ASGI app over a real socket, so the
event loop, the serialiser and the GIL are all in the path.

**It has been run, and the falsifier did not fire** — see `docs/decisions.md`,
"The tail falsifier was the real test". Read p99/p50 rather than p99: a GC or
GIL pause widens the tail relative to the median as pressure rises, and here
that ratio *narrows*, from 1.6× at one client to 1.5× at thirty-two, while
latency tracks `concurrency / throughput` to within a millisecond. That is
queueing, which the falsifier explicitly excluded. Re-run it after anything
that touches the request path, and read the ratio.

**What it does not measure.** A real deployment's p99 is dominated by the model
server and by queueing in front of it, neither of which exists here — the
lexical backend answers in microseconds. So this isolates the gateway's *own*
tail, which is the number the language decision turns on, and is not a
prediction of production p99.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

BODY = {
    "state": "the card payment was declined at the till and the customer is upset",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Route this ticket.",
            "options": [{"name": n} for n in ("billing", "shipping", "account", "other")],
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this?",
            "levels": [{"name": "low", "value": 1.0}, {"name": "high", "value": 5.0}],
        },
        "urgent": {"type": "noul", "instructions": "Needs a human within the hour?"},
    },
}


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank, like the eval harness. Exact on small samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=8971)
    parser.add_argument("--backend", default="lexical")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    import urllib.error
    import urllib.request

    import uvicorn

    from trigon.limits import DEFAULT_LATENCY_TARGET
    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    app = build_app(ServerConfig(backend=args.backend, weights=args.weights))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        print("the gateway did not come up", file=sys.stderr)
        return 1

    import json

    url = f"http://127.0.0.1:{args.port}/v1/decide"
    payload = json.dumps(BODY).encode()
    latencies: list[float] = []
    errors = [0]
    lock = threading.Lock()
    stop_at = time.time() + args.seconds

    def worker() -> None:
        mine: list[float] = []
        failed = 0
        while time.time() < stop_at:
            request = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    response.read()
                mine.append((time.perf_counter() - started) * 1000)
            except (urllib.error.URLError, OSError):
                failed += 1
        with lock:
            latencies.extend(mine)
            errors[0] += failed

    # Warm one request so import-time work is not in the sample.
    worker_warm = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(worker_warm, timeout=30) as response:
        response.read()

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - started
    server.should_exit = True
    thread.join(timeout=10)

    if not latencies:
        print("no successful requests", file=sys.stderr)
        return 1

    p50 = statistics.median(latencies)
    p99 = _percentile(latencies, 0.99)
    target = DEFAULT_LATENCY_TARGET

    print(f"backend {args.backend}, {args.workers} concurrent workers, {elapsed:.1f}s")
    print(f"{'requests':<28} {len(latencies):>10,}   ({len(latencies) / elapsed:,.0f}/s)")
    print(f"{'errors':<28} {errors[0]:>10,}")
    print(f"{'p50':<28} {p50:>10.2f} ms   (target {target.p50_ms:.0f})")
    print(f"{'p90':<28} {_percentile(latencies, 0.90):>10.2f} ms")
    print(f"{'p99':<28} {p99:>10.2f} ms   (target {target.p99_ms:.0f})")
    print(f"{'p99 / p50':<28} {p99 / p50:>10.1f}x")
    print(f"{'max':<28} {max(latencies):>10.2f} ms")
    print()
    print(
        "This is the gateway's own tail: the lexical backend answers in "
        "microseconds,\nso there is no model server or queue in front of it. A "
        "production p99 is\ndominated by those, not by this."
    )
    if p99 > target.p99_ms:
        print("\n**The gateway alone exceeds the p99 target.** That is the falsifier in")
        print("docs/decisions.md — revisit the language choice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
