"""Environment-driven serving configuration.

Kept tiny and explicit: a deployment should be able to read this file and know
every knob that exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..calibration.conformal import ConformalPredictor
from ..calibration.isotonic import IsotonicCalibrator
from ..calibration.temperature import TemperatureScaler

__all__ = ["ServerConfig"]


def _keys(raw: str | None) -> frozenset[str] | None:
    """Comma-separated keys, or None for "no authentication".

    An empty or whitespace-only value is None rather than an empty set. An
    empty set would reject every request, which is a configuration mistake
    that looks exactly like a total outage.
    """
    if not raw or not raw.strip():
        return None
    keys = frozenset(part.strip() for part in raw.split(",") if part.strip())
    return keys or None


def _positive(raw: str | None) -> int | None:
    if not raw or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"expected a positive limit, got {value}")
    return value


@dataclass
class ServerConfig:
    backend: str = "lexical"
    # Path to a checkpoint written by ``trigon train``. Without it a torch
    # backend serves randomly initialised weights, which answer every question
    # with noise -- so the gateway reports whether it has any.
    weights: str | None = None
    domain: str | None = None
    # The cross-request schema KV cache. A gateway is the one place the
    # weights are fixed for a process's lifetime, which is what makes a prefix
    # safe to reuse at all.
    #
    # **On by default, now that it has been timed.** At the shape the
    # certified Banking77 model actually serves -- 77 options, 707 of 725
    # tokens schema -- the cache is a 6x speedup: 91.8 ms to 15.3 ms p50. At
    # 256 options it is 23x, and the uncached tail is twice its own median
    # because it recomputes a 2,474-token self-attention whose answer cannot
    # change. reports/cache/README.md has the table.
    #
    # It was off for two reasons and both are spent. The first, that the cache
    # traded exact independence for compute, is wrong: on GitHub's runners the
    # *cached* path is the exact one and the uncached drifts 2.4e-08, because
    # what decides it is whether a sequence length lands on a kernel that
    # reduces in the same order. The second, that nobody had timed it, was
    # honest and is no longer true.
    #
    # Set TRIGON_CACHE_PREFIXES=0 to turn it off. /healthz reports which way
    # it is set, for the same reason it reports whether the deployment is
    # calibrated: an operator should not have to guess which numbers their
    # gateway is producing.
    cache_prefixes: bool = True
    # Paths to fitted artifacts. Absent means "serve uncalibrated and say so".
    temperature_path: str | None = None
    # `trigon train` picks a calibrator per primitive and writes whichever
    # it chose. A deployment given only the temperatures would silently
    # serve any isotonic-calibrated primitive raw.
    isotonic_path: str | None = None
    conformal_dir: str | None = None
    premium_backend: str | None = None
    premium_weights: str | None = None
    escalate_below_confidence: float = 0.35
    # -- the three codes a caller's retry loop branches on -------------------
    #
    # All three are None by default, and that is the honest default rather
    # than a lax one: a self-hosted gateway should not invent an auth or
    # capacity policy its operator did not choose. What the guards guarantee
    # is that when an operator *does* set them, the codes and headers match
    # the contract a migrating caller already has. See docs/compat.md.
    api_keys: frozenset[str] | None = None
    rate_per_minute: int | None = None
    max_concurrent: int | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServerConfig:
        source = env if env is not None else dict(os.environ)
        return cls(
            backend=source.get("TRIGON_BACKEND", "lexical"),
            weights=source.get("TRIGON_WEIGHTS") or None,
            domain=source.get("TRIGON_DOMAIN") or None,
            cache_prefixes=source.get("TRIGON_CACHE_PREFIXES", "1") not in ("0", "false"),
            temperature_path=source.get("TRIGON_TEMPERATURE_PATH") or None,
            isotonic_path=source.get("TRIGON_ISOTONIC_PATH") or None,
            conformal_dir=source.get("TRIGON_CONFORMAL_DIR") or None,
            premium_backend=source.get("TRIGON_PREMIUM_BACKEND") or None,
            premium_weights=source.get("TRIGON_PREMIUM_WEIGHTS") or None,
            escalate_below_confidence=float(source.get("TRIGON_ESCALATE_BELOW_CONFIDENCE", "0.35")),
            api_keys=_keys(source.get("TRIGON_API_KEYS")),
            rate_per_minute=_positive(source.get("TRIGON_RATE_PER_MINUTE")),
            max_concurrent=_positive(source.get("TRIGON_MAX_CONCURRENT")),
        )

    def load_scaler(self) -> TemperatureScaler:
        if not self.temperature_path:
            return TemperatureScaler()
        return TemperatureScaler.load(self.temperature_path)

    def load_isotonic(self) -> IsotonicCalibrator:
        if not self.isotonic_path:
            return IsotonicCalibrator()
        return IsotonicCalibrator.load(self.isotonic_path)

    def load_conformal(self) -> dict[str, ConformalPredictor]:
        if not self.conformal_dir:
            return {}
        from pathlib import Path

        profiles = {}
        for path in sorted(Path(self.conformal_dir).glob("*.json")):
            profiles[path.stem] = ConformalPredictor.load(path)
        return profiles

    @property
    def is_calibrated(self) -> bool:
        """Surfaced on /healthz: serving uncalibrated is allowed, hiding it is not."""
        return bool(self.temperature_path or self.isotonic_path)

    @property
    def is_trained(self) -> bool:
        """Whether **every** configured tier has learned anything.

        Surfaced on /healthz. A torch backend with no checkpoint answers every
        question from randomly initialised weights, which is a far worse
        failure than being uncalibrated and looks identical from outside.

        Both tiers count. An untrained premium tier is the harder one to
        notice: it answers only the questions the workhorse could not settle,
        so it is exactly the traffic nobody is watching, and a deployment that
        reported ``trained`` for the workhorse alone would say nothing about
        it.
        """
        return self._tier_is_trained(self.backend, self.weights) and self._tier_is_trained(
            self.premium_backend, self.premium_weights
        )

    @staticmethod
    def _tier_is_trained(backend: str | None, weights: str | None) -> bool:
        if backend is None:
            return True  # a tier that is not configured cannot be untrained
        return backend != "torch" or bool(weights)
