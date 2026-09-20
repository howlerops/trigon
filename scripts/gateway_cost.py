#!/usr/bin/env python
"""Measure what the gateway itself costs, so the language choice is evidence.

    python scripts/gateway_cost.py

The build plan budgeted phase-3 time to rewrite this gateway in Rust. That is
a claim about where time goes, and it is checkable in a few seconds, so it
should not be taken on faith in either direction. See `docs/decisions.md`,
"Python for the gateway, Rust for one function, Go for nothing".

Run it again whenever the answer might have changed — a heavier validation
path, a slower tokenizer, a real backend — because the decision it supports is
a measurement, not a preference.
"""

from __future__ import annotations

import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.bpe import BPETokenizer  # noqa: E402
from trigon.limits import DEFAULT_BUDGET, DEFAULT_LATENCY_TARGET  # noqa: E402
from trigon.schema.compiler import SchemaCompiler  # noqa: E402
from trigon.types import SystemOneRequest  # noqa: E402

TYPICAL = {
    "state": "the card payment was declined at the till and the customer is upset",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Route this ticket.",
            "options": [{"name": n} for n in ("billing", "shipping", "account", "other")],
        },
        "urgent": {"type": "noul", "instructions": "Needs a human within the hour?"},
    },
}


def _median_ms(fn, n: int) -> float:
    samples = []
    for _ in range(n):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> int:
    from fastapi.testclient import TestClient

    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    client = TestClient(build_app(ServerConfig(backend="lexical")))
    client.post("/v1/systemone", json=TYPICAL)  # warm

    whole = _median_ms(lambda: client.post("/v1/systemone", json=TYPICAL), 200)

    compiler = SchemaCompiler()
    wide = SystemOneRequest.model_validate(
        {
            "state": "a short ticket about a declined card",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Pick one.",
                    "options": [
                        {"name": f"opt_{i:05d}", "criteria": f"situation {i} in the taxonomy"}
                        for i in range(1000)
                    ],
                }
            },
        }
    )
    compiler.compile_request(wide)
    compile_wide = _median_ms(lambda: compiler.compile_request(wide), 20)

    tokenizer = BPETokenizer.load()
    # Novel text: a merge cache must not be able to carry it.
    import random

    rng = random.Random(0)
    novel = " ".join(
        f"{rng.choice('abcdefghijklmnop')}{rng.randrange(10**6)}zq{rng.randrange(10**4)}"
        for _ in range(20000)
    )
    counted = tokenizer.count(novel)
    per_ms = counted / _median_ms(lambda: tokenizer.encode(novel), 3)
    ceiling_ms = DEFAULT_BUDGET.state_tokens / per_ms

    p50 = DEFAULT_LATENCY_TARGET.p50_ms
    print(
        f"{'whole request, over HTTP':<44} {whole:>8.2f} ms   {1000 / whole:>7,.0f} req/s per core"
    )
    print(f"{'compiling a 1,000-option schema':<44} {compile_wide:>8.2f} ms")
    print(
        f"{'tokenizing a ceiling-size state':<44} {ceiling_ms:>8.1f} ms   "
        f"({per_ms * 1000:,.0f} tok/s, {DEFAULT_BUDGET.state_tokens:,} tokens)"
    )
    print(f"{'p50 latency target':<44} {p50:>8.1f} ms")
    print()
    print(f"The gateway's own work is {whole / p50:.1%} of the p50 budget.")
    print(f"A gateway that cost nothing at all would move p50 by {whole:.2f} ms.")
    if whole / p50 > 0.15:
        print("\nThat is no longer a rounding error — revisit docs/decisions.md.")
    else:
        print("\nRewriting it optimises the wrong thing. The tokenizer is the only")
        print("piece worth native code, and it arrives as a dependency in phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
