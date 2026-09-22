"""Auth, rate limiting and overload shedding — the three codes a caller retries on.

`docs/compat.md` listed `401`, `429` and `529` under "what is not there", and
that is a real gap rather than a cosmetic one: a caller migrating brings a
retry loop written against those codes, and a server that never emits them
leaves that loop untested. A load test against a gateway that *queues* where
the incumbent *sheds* measures the wrong thing and measures it optimistically.

**All three are off unless configured**, because the honest default for a
self-hosted gateway is not to invent a policy the operator did not choose. An
unset key means no `401`; an unset rate means no `429`; an unset concurrency
cap means no `529`. What this module guarantees is that when an operator does
set them, the codes and the headers match the contract a caller already has.

The limiter is a fixed-window counter, in memory, per process. That is stated
rather than hidden because it decides what it is good for: it protects one
process from one caller's burst, it does not coordinate across replicas, and a
deployment that needs a global budget puts a real limiter in front. A sliding
window would be more accurate at the boundary and is not worth the bookkeeping
for something whose job is to shed load rather than to meter it precisely.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

__all__ = ["RateLimiter", "install_guards"]


class RateLimiter:
    """Fixed-window request counting, per key, in one process.

    Not a token bucket: a bucket smooths bursts, and smoothing is the opposite
    of what a shedder is for. The window resets on a wall-clock boundary, so a
    caller can send `rate` requests at the end of one window and `rate` at the
    start of the next. That is the known cost of a fixed window and it is
    acceptable for shedding; it would not be for billing.
    """

    def __init__(self, rate: int, window_seconds: float = 60.0) -> None:
        if rate <= 0:
            raise ValueError("a rate limit of zero would reject every request")
        self.rate = rate
        self.window = window_seconds
        self._counts: dict[str, tuple[float, int]] = {}
        # A gateway serves requests on a thread pool, so two callers sharing a
        # key can read-modify-write the same counter. Without this the limit
        # is approximately enforced, which for a shedder means occasionally
        # not enforced at the moment it matters most.
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int, float]:
        """``(allowed, remaining, seconds_until_reset)``."""
        now = time.monotonic() if now is None else now
        window_start = now - (now % self.window)
        reset_in = self.window - (now % self.window)
        with self._lock:
            start, count = self._counts.get(key, (window_start, 0))
            if start < window_start:
                start, count = window_start, 0
            if count >= self.rate:
                self._counts[key] = (start, count)
                return False, 0, reset_in
            self._counts[key] = (start, count + 1)
            return True, self.rate - count - 1, reset_in


def _key(request: Request) -> str:
    """Who to count against.

    The bearer token when there is one, since that is the unit an operator
    thinks in, and the peer address otherwise. Never both concatenated: a
    caller rotating keys behind one address would then get a fresh budget per
    rotation, which is the failure mode a limiter exists to prevent.
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.client.host if request.client else "anonymous"


def _error(status: int, kind: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"type": kind, "message": message}},
        headers=headers or {},
    )


def install_guards(
    app: FastAPI,
    *,
    api_keys: frozenset[str] | None = None,
    rate_per_minute: int | None = None,
    max_concurrent: int | None = None,
    protected_prefixes: tuple[str, ...] = ("/v1/", "/compat/"),
) -> None:
    """Add 401, 429 and 529 to ``app``, each only if it was configured.

    `/healthz` stays open on purpose: a liveness probe that needs a credential
    is a liveness probe that reports the credential's health, and an operator
    debugging a 401 storm needs one endpoint that answers.
    """
    limiter = RateLimiter(rate_per_minute) if rate_per_minute else None
    inflight = {"n": 0}
    gate = threading.Lock()

    @app.middleware("http")
    async def _guard(request: Request, call_next: Callable):
        path = request.url.path
        if not any(path.startswith(p) for p in protected_prefixes):
            return await call_next(request)

        if api_keys is not None:
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else ""
            if token not in api_keys:
                # WWW-Authenticate because the contract this mirrors is a
                # bearer scheme, and a 401 without it is one a client cannot
                # act on programmatically.
                return _error(
                    401,
                    "authentication_error",
                    "invalid or missing API key",
                    {"WWW-Authenticate": "Bearer"},
                )

        if limiter is not None:
            allowed, remaining, reset_in = limiter.check(_key(request))
            if not allowed:
                # Retry-After in whole seconds, rounded up: a client that
                # retries at exactly the boundary races the window and is
                # rejected again, which reads to them as the limiter lying.
                return _error(
                    429,
                    "rate_limit_error",
                    f"more than {limiter.rate} requests in {limiter.window:.0f}s",
                    {
                        "Retry-After": str(int(reset_in) + 1),
                        "X-RateLimit-Limit": str(limiter.rate),
                        "X-RateLimit-Remaining": "0",
                    },
                )

        if max_concurrent is not None:
            with gate:
                if inflight["n"] >= max_concurrent:
                    # 529, not 503: the contract this mirrors uses 529 for
                    # overload, and a caller's backoff branches on that number.
                    return _error(
                        529,
                        "overloaded_error",
                        "the server is at capacity; retry with backoff",
                        {"Retry-After": "1"},
                    )
                inflight["n"] += 1
            try:
                return await call_next(request)
            finally:
                with gate:
                    inflight["n"] -= 1

        return await call_next(request)
