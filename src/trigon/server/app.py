"""The gateway. Validates, compiles, routes, calibrates, answers.

The production gateway in the build plan is Rust; this is the reference
implementation that pins the contract and generates the OpenAPI spec both SDKs
are built from. Keeping one executable definition of the contract is what makes
"drop-in compatible" checkable instead of aspirational --
``tests/test_openapi_drift.py`` fails when the checked-in spec and this app
disagree.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..backends.lexical import LexicalBackend
from ..engine import Engine, EngineConfig
from ..limits import DEFAULT_BUDGET, DEFAULT_LATENCY_TARGET
from ..schema.compiler import SchemaTooLarge
from ..types import SystemOneRequest, SystemOneResponse
from .config import ServerConfig
from .routing import RoutingPolicy, TieredRouter

__all__ = ["build_app", "create_router"]

DESCRIPTION = """
Typed, calibrated decisions in one prefill-only forward pass.

Send state plus a map of typed questions; get one typed answer per question,
each a distribution over exactly the labels you declared. Answers cannot
violate the schema: the only vector the model produces is a softmax over your
option set.
"""


def create_router(config: ServerConfig | None = None) -> TieredRouter:
    """Assemble the serving stack described by ``config``."""
    config = config or ServerConfig.from_env()
    scaler = config.load_scaler()
    conformal = config.load_conformal()

    workhorse = Engine(
        _backend(config.backend),
        scaler=scaler,
        conformal=conformal,
        config=EngineConfig(domain=config.domain, tier="workhorse"),
    )
    premium = (
        Engine(
            _backend(config.premium_backend),
            scaler=scaler,
            conformal=conformal,
            config=EngineConfig(domain=config.domain, tier="premium"),
        )
        if config.premium_backend
        else None
    )
    return TieredRouter(
        workhorse,
        premium,
        RoutingPolicy(escalate_below_confidence=config.escalate_below_confidence),
    )


def _backend(name: str) -> Any:
    if name == "lexical":
        return LexicalBackend()
    if name == "torch":
        from ..backends.torch_readout import TorchReadoutBackend

        return TorchReadoutBackend()
    raise ValueError(f"unknown backend {name!r}; known backends are 'lexical' and 'torch'")


def build_app(config: ServerConfig | None = None, router: TieredRouter | None = None) -> FastAPI:
    config = config or ServerConfig.from_env()
    router = router or create_router(config)

    app = FastAPI(
        title="Trigon System One API",
        version=__version__,
        description=DESCRIPTION.strip(),
        openapi_tags=[
            {"name": "inference", "description": "Typed decisions."},
            {"name": "ops", "description": "Health and capability reporting."},
        ],
    )
    app.state.router = router
    app.state.config = config

    @app.exception_handler(SchemaTooLarge)
    async def _too_large(_: Request, exc: SchemaTooLarge) -> JSONResponse:
        # 413, not 422: the request is well-formed, it just does not fit.
        return JSONResponse(
            status_code=413, content={"error": {"type": "schema_too_large", "message": str(exc)}}
        )

    @app.post(
        "/v1/systemone",
        response_model=SystemOneResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        summary="Answer a map of typed questions about one state",
    )
    def systemone(request: SystemOneRequest) -> SystemOneResponse:
        try:
            return router.answer(request)
        except SchemaTooLarge:
            # Re-raised rather than handled here: the app-level handler turns
            # it into a 413. Catching it below as a ValueError -- which it is --
            # would report the caller's oversized request as our bug.
            raise
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            # A backend that broke the structural guarantee is our bug, not the
            # caller's, and it must never surface as a malformed answer.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/v1/models", tags=["ops"], summary="Serving tiers and their versions")
    def models() -> dict:
        tiers = [{"tier": "workhorse", "model": router.workhorse.backend.model_version}]
        if router.premium is not None:
            tiers.append({"tier": "premium", "model": router.premium.backend.model_version})
        return {"data": tiers}

    @app.get("/healthz", tags=["ops"], summary="Liveness and calibration status")
    def healthz() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            # Deliberately exposed: an uncalibrated deployment is allowed, a
            # silently uncalibrated one is not.
            "calibrated": config.is_calibrated,
            "context_tokens": DEFAULT_BUDGET.context_tokens,
            "latency_target_ms": {
                "p50": DEFAULT_LATENCY_TARGET.p50_ms,
                "p99": DEFAULT_LATENCY_TARGET.p99_ms,
            },
        }

    return app


def __getattr__(name: str):
    # ``uvicorn trigon.server.app:app`` without building an app at import time
    # for every test that only wants ``build_app``.
    if name == "app":
        return build_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
