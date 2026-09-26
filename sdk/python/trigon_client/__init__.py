"""Trigon — Python client.

Generated from ``spec/openapi.json``, which is the contract. Nothing here is
hand-maintained: ``scripts/generate_sdk.py`` writes ``_generated.py`` and
``tests/test_sdk.py`` fails if it is stale, so a contract change that is not
regenerated cannot land.

Standard library only. An SDK is the first thing anyone installs, and one that
drags in an HTTP stack to call three endpoints is a worse first impression than
forty lines of ``urllib``.
"""

from ._generated import (
    CONTRACT_VERSION,
    ChoiceAnswer,
    EvidenceSpan,
    NoulAnswer,
    Response,
    ScoreAnswer,
    Timing,
    TrigonClient,
    TrigonError,
    Usage,
    choice,
    noul,
    parse_answer,
    score,
)

__all__ = [
    "CONTRACT_VERSION",
    "ChoiceAnswer",
    "EvidenceSpan",
    "NoulAnswer",
    "Response",
    "ScoreAnswer",
    "Timing",
    "TrigonClient",
    "TrigonError",
    "Usage",
    "choice",
    "noul",
    "parse_answer",
    "score",
]
