"""The gateway. Validates, compiles, routes, calibrates, answers.

This is the gateway, not a reference for one: the build plan's Rust rewrite
does not survive measurement (docs/decisions.md). Keeping one executable
definition of the contract is what makes "drop-in compatible" checkable
instead of aspirational -- ``tests/test_openapi_drift.py`` fails when the
checked-in spec and this app disagree, and both SDKs are generated from it.
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
from ..types import DecisionRequest, DecisionResponse
from .config import ServerConfig
from .limits_middleware import install_guards
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
    isotonic = config.load_isotonic()
    conformal = config.load_conformal()

    workhorse = Engine(
        _backend(
            config.backend, config.weights, config.cache_prefixes, config.unsupervised_evidence
        ),
        scaler=scaler,
        isotonic=isotonic,
        conformal=conformal,
        config=EngineConfig(domain=config.domain, tier="workhorse"),
    )
    premium = (
        Engine(
            _backend(
                config.premium_backend,
                config.premium_weights,
                config.cache_prefixes,
                config.unsupervised_evidence,
            ),
            scaler=scaler,
            isotonic=isotonic,
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


def _backend(
    name: str,
    weights: str | None = None,
    cache_prefixes: bool = True,
    unsupervised_evidence: str | None = None,
) -> Any:
    if name == "lexical":
        if weights:
            raise ValueError("the lexical backend has no weights to load")
        return LexicalBackend()
    if name == "torch":
        from ..backends.torch_readout import TorchReadoutBackend

        backend = TorchReadoutBackend.load(weights) if weights else TorchReadoutBackend()
        # A schema prefix belongs to the weights that produced it. Those are
        # fixed for this process's lifetime, which is what makes reuse safe
        # here and unsafe in the trainer.
        backend.cache_prefixes = cache_prefixes
        if unsupervised_evidence:
            backend.unsupervised_evidence = unsupervised_evidence
            # At startup, not on the first evidence request: a misspelt
            # method should stop the deployment, not 500 a caller.
            backend._resolved_evidence_mode()
        return backend
    raise ValueError(f"unknown backend {name!r}; known backends are 'lexical' and 'torch'")


def build_app(config: ServerConfig | None = None, router: TieredRouter | None = None) -> FastAPI:
    config = config or ServerConfig.from_env()
    router = router or create_router(config)

    app = FastAPI(
        title="Trigon API",
        version=__version__,
        description=DESCRIPTION.strip(),
        openapi_tags=[
            {"name": "inference", "description": "Typed decisions."},
            {"name": "ops", "description": "Health and capability reporting."},
        ],
    )
    app.state.router = router
    app.state.config = config
    install_guards(
        app,
        api_keys=config.api_keys,
        rate_per_minute=config.rate_per_minute,
        max_concurrent=config.max_concurrent,
    )

    @app.exception_handler(SchemaTooLarge)
    async def _too_large(_: Request, exc: SchemaTooLarge) -> JSONResponse:
        # 413, not 422: the request is well-formed, it just does not fit.
        return JSONResponse(
            status_code=413, content={"error": {"type": "schema_too_large", "message": str(exc)}}
        )

    @app.post(
        "/v1/decide",
        response_model=DecisionResponse,
        response_model_exclude_none=True,
        tags=["inference"],
        summary="Answer a map of typed questions about one state",
    )
    def decide(request: DecisionRequest) -> DecisionResponse:
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
            # silently uncalibrated one is not. Same for an untrained one,
            # which is the worse failure and looks identical from outside.
            "calibrated": config.is_calibrated,
            "trained": config.is_trained,
            # Reported for the same reason as the two above: the cache moves
            # answers by ~5e-08 and wall clock by 6x, and an operator
            # comparing two deployments should not have to guess which of
            # them is running it.
            "schema_cache": config.cache_prefixes,
            # What a checkpoint never trained on rationales answers
            # `include_evidence` with; null for a backend that cannot
            # attribute. The response names the method too, but an operator
            # comparing deployments should not need a request to find out.
            "unsupervised_evidence": getattr(
                router.workhorse.backend, "unsupervised_evidence", None
            ),
            "context_tokens": DEFAULT_BUDGET.context_tokens,
            "latency_target_ms": {
                "p50": DEFAULT_LATENCY_TARGET.p50_ms,
                "p99": DEFAULT_LATENCY_TARGET.p99_ms,
            },
        }

    # Their shapes, on their path, under a prefix -- so one process serves both
    # and `tests/test_compat.py` can assert the two return the same numbers.
    # A caller actually migrating points at `build_compat_app`, whose root is
    # their base URL; see docs/compat.md.
    from .compat import compat_router

    app.include_router(compat_router(router), prefix="/compat")

    return app


def __getattr__(name: str):
    # ``uvicorn trigon.server.app:app`` without building an app at import time
    # for every test that only wants ``build_app``.
    if name == "app":
        return build_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
