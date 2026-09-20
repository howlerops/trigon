"""The gateway. Pins the contract and generates the OpenAPI spec the SDKs are
generated from.

The build plan called for rewriting this in Rust for production. Measured, the
gateway's own work is 2.4% of the p50 latency budget and a rounding error
against GPU cost, so a rewrite buys three milliseconds of a hundred and fifty
and costs a second implementation of the contract. See docs/decisions.md,
"Python for the gateway, Rust for one function, Go for nothing" -- including
the three measurements that would change that."""

from .config import ServerConfig
from .routing import RoutingPolicy, TieredRouter

__all__ = ["RoutingPolicy", "ServerConfig", "TieredRouter", "build_app"]


def __getattr__(name: str):
    if name == "build_app":
        from .app import build_app

        return build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
