"""Environment-driven serving configuration.

Kept tiny and explicit: a deployment should be able to read this file and know
every knob that exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..calibration.conformal import ConformalPredictor
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
    # Paths to fitted artifacts. Absent means "serve uncalibrated and say so".
    temperature_path: str | None = None
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
            temperature_path=source.get("TRIGON_TEMPERATURE_PATH") or None,
            conformal_dir=source.get("TRIGON_CONFORMAL_DIR") or None,
            premium_backend=source.get("TRIGON_PREMIUM_BACKEND") or None,
            premium_weights=source.get("TRIGON_PREMIUM_WEIGHTS") or None,
            escalate_below_confidence=float(source.get("TRIGON_ESCALATE_BELOW_CONFIDENCE", "0.35")),
        )

    def load_scaler(self) -> TemperatureScaler:
        if not self.temperature_path:
            return TemperatureScaler()
        return TemperatureScaler.load(self.temperature_path)

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
        return bool(self.temperature_path)

    @property
    def is_trained(self) -> bool:
        """Whether the serving backend has learned anything.

        Also surfaced on /healthz. A torch backend with no checkpoint answers
        every question from randomly initialised weights, which is a far worse
        failure than being uncalibrated and looks identical from outside.
        """
        return self.backend != "torch" or bool(self.weights)
