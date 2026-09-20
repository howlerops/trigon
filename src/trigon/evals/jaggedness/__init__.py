"""The jaggedness suite: one benchmark per documented failure mode."""

from .benchmarks import (
    Benchmark,
    ContextRotBenchmark,
    ContradictoryCriteriaBenchmark,
    CountingBenchmark,
    DateComparisonBenchmark,
    IndirectionDepthBenchmark,
    InjectionSteeringBenchmark,
    LiteralReadingBenchmark,
    NegationCoherenceBenchmark,
    NoulChoiceAgreementBenchmark,
    all_benchmarks,
)
from .runner import run_jaggedness

__all__ = [
    "Benchmark",
    "ContextRotBenchmark",
    "ContradictoryCriteriaBenchmark",
    "CountingBenchmark",
    "DateComparisonBenchmark",
    "IndirectionDepthBenchmark",
    "InjectionSteeringBenchmark",
    "LiteralReadingBenchmark",
    "NegationCoherenceBenchmark",
    "NoulChoiceAgreementBenchmark",
    "all_benchmarks",
    "run_jaggedness",
]
