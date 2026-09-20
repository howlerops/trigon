"""Token counting, behind a protocol so the compiler never imports a tokenizer.

The gateway compiles and budgets schemas on CPU nodes that have no model
weights loaded, so the default estimator is a heuristic. Serving replicas pass
the real tokenizer in. The heuristic is deliberately biased to over-count: a
schema wrongly admitted overflows a replica, a schema wrongly rejected returns
a 413 the caller can act on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol, runtime_checkable

__all__ = ["CharHeuristicEstimator", "CallableEstimator", "TokenEstimator"]


@runtime_checkable
class TokenEstimator(Protocol):
    def count(self, text: str) -> int:
        """Number of tokens ``text`` occupies."""
        ...

    @property
    def exact(self) -> bool:
        """True when counts come from the serving tokenizer, not a heuristic."""
        ...


class CharHeuristicEstimator:
    """~4 characters per token, rounded up, with a per-segment overhead.

    ``chars_per_token`` of 4.0 is the usual English figure; JSON-shaped state
    and terse option names run denser, which the ceiling and the per-segment
    overhead absorb.
    """

    exact = False

    def __init__(self, chars_per_token: float = 4.0, segment_overhead: int = 2) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self.chars_per_token = chars_per_token
        self.segment_overhead = segment_overhead

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) + self.segment_overhead


class CallableEstimator:
    """Wraps a real tokenizer, e.g. ``CallableEstimator(tok.encode, exact=True)``."""

    def __init__(self, encode: Callable[[str], list], *, exact: bool = True) -> None:
        self._encode = encode
        self._exact = exact

    @property
    def exact(self) -> bool:
        return self._exact

    def count(self, text: str) -> int:
        return len(self._encode(text))
