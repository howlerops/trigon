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
    # **Off by default, and that is a deliberate trade rather than caution.**
    # The saving requires attending with only the non-schema positions as
    # queries, which changes the GEMM shape, so float32 rounds differently.
    # Per-question independence -- one of the two claims this product rests on,
    # asserted to exact equality in `tests/test_independence.py` -- degrades
    # from exact to 4.6e-08 measured on the served path. That is far below
    # anything a caller could act on, and it is still the difference between a
    # guarantee and a tolerance. An operator who wants the compute back opts
    # in; nobody gets a weaker promise than the one they read about by
    # accident.
    cache_prefixes: bool = False
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

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServerConfig:
        source = env if env is not None else dict(os.environ)
        return cls(
            backend=source.get("TRIGON_BACKEND", "lexical"),
            weights=source.get("TRIGON_WEIGHTS") or None,
            domain=source.get("TRIGON_DOMAIN") or None,
            cache_prefixes=source.get("TRIGON_CACHE_PREFIXES", "0") not in ("0", "false"),
            temperature_path=source.get("TRIGON_TEMPERATURE_PATH") or None,
            isotonic_path=source.get("TRIGON_ISOTONIC_PATH") or None,
            conformal_dir=source.get("TRIGON_CONFORMAL_DIR") or None,
            premium_backend=source.get("TRIGON_PREMIUM_BACKEND") or None,
            premium_weights=source.get("TRIGON_PREMIUM_WEIGHTS") or None,
            escalate_below_confidence=float(source.get("TRIGON_ESCALATE_BELOW_CONFIDENCE", "0.35")),
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
