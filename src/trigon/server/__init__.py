"""Reference gateway. The production gateway is Rust; this one pins the
contract and generates the OpenAPI spec the SDKs are built from."""

from .config import ServerConfig
from .routing import RoutingPolicy, TieredRouter

__all__ = ["RoutingPolicy", "ServerConfig", "TieredRouter", "build_app"]


def __getattr__(name: str):
    if name == "build_app":
        from .app import build_app

        return build_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
