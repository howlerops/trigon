"""Reference gateway. The production gateway is Rust; this one pins the
contract and generates the OpenAPI spec. The Python and TypeScript SDKs are a
phase-4 deliverable (docs/roadmap.md) and will be generated from that spec --
which is why it is checked in and drift-tested now, before anything reads it."""

from .config import ServerConfig
from .routing import RoutingPolicy, TieredRouter

__all__ = ["RoutingPolicy", "ServerConfig", "TieredRouter", "build_app"]


def __getattr__(name: str):
    if name == "build_app":
        from .app import build_app

        return build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
