"""Run the jaggedness suite and collect one result per failure mode."""

from __future__ import annotations

from collections.abc import Sequence

from ...engine import Engine
from ..harness import SuiteResult, run_cases, summarize
from .benchmarks import Benchmark, all_benchmarks

__all__ = ["run_jaggedness"]


def run_jaggedness(
    engine: Engine,
    benchmarks: Sequence[Benchmark] | None = None,
    *,
    n: int = 40,
    seed: int = 0,
) -> list[SuiteResult]:
    """One ``SuiteResult`` per benchmark, each carrying its own metrics.

    Benchmark-specific metrics land in ``extra``; ``accuracy`` is ``None`` for
    the benchmarks that have no correct answer by construction.
    """
    results = []
    for benchmark in benchmarks or all_benchmarks(n=n, seed=seed):
        outcomes = run_cases(engine, benchmark.cases())
        extra = benchmark.score(outcomes)
        result = summarize(
            f"jaggedness/{benchmark.name}",
            engine.backend.model_version,
            outcomes,
            extra=extra,
        )
        results.append(result)
    return results
