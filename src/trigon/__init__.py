"""Trigon: open-source typed, calibrated decisions in one pass.

The public surface is small on purpose. ``Engine`` is the whole pipeline;
everything else is a part of it you can replace.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .confidence import choice_confidence, score_confidence
from .engine import Engine, EngineConfig
from .limits import DEFAULT_BUDGET, Budget
from .schema import SchemaCompiler, compile_request
from .types import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

__all__ = [
    "Budget",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DEFAULT_BUDGET",
    "Engine",
    "EngineConfig",
    "NoulAnswer",
    "NoulQuestion",
    "SchemaCompiler",
    "ScoreAnswer",
    "ScoreQuestion",
    "DecisionRequest",
    "DecisionResponse",
    "__version__",
    "choice_confidence",
    "compile_request",
    "score_confidence",
]
