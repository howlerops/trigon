"""The calibration suite, and the release gates it enforces.

This is the suite the build plan says never to cut, and the one whose output is
the differentiator: the competitor has published no reliability curve, no ECE
number and no ablation, so the first reproducible calibration report sets the
standard. Which means the report has to be honest about its own weaknesses --
both ECE estimators, per-primitive and per-domain breakdowns, and a stated
gate that a run either passes or fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..calibration.metrics import CalibrationReport, report
from ..engine import Engine
from ..limits import CALIBRATION_GATES
from .harness import Case, CaseOutcome, SuiteResult, run_cases, summarize

__all__ = ["GateResult", "check_gates", "run_calibration_suite", "slice_reports"]


@dataclass(frozen=True)
class GateResult:
    """Whether a run may ship, and why not if it may not."""

    name: str
    value: float
    limit: float
    passed: bool

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return f"{verdict} {self.name}: {self.value:.4f} (limit {self.limit:.4f})"


def run_calibration_suite(
    engine: Engine, cases: Sequence[Case], *, suite: str = "calibration"
) -> tuple[SuiteResult, dict[str, CalibrationReport]]:
    """Run the suite and return the summary plus per-domain breakdowns."""
    outcomes = run_cases(engine, cases)
    result = summarize(suite, engine.backend.model_version, outcomes)
    return result, slice_reports(outcomes)


def slice_reports(outcomes: Sequence[CaseOutcome]) -> dict[str, CalibrationReport]:
    """Per-domain reports. An aggregate ECE can hide a domain that is badly
    miscalibrated, and per-domain is the granularity users actually deploy at."""
    buckets: dict[str, list[tuple[tuple[float, ...], int]]] = {}
    for outcome in outcomes:
        domain = outcome.case.domain
        for question in outcome.questions.values():
            if question.expected is None:
                continue
            truth = question.expected.hard_label
            if truth is None:
                continue
            buckets.setdefault(domain, []).append((question.probabilities, truth))
    return {
        domain: report([p for p, _ in rows], [y for _, y in rows], slice_name=domain)
        for domain, rows in sorted(buckets.items())
        if rows
    }


def check_gates(
    result: SuiteResult,
    tier: str = "workhorse",
    quantized: SuiteResult | None = None,
) -> list[GateResult]:
    """Apply the release gates from the build plan.

    ``quantized`` is the same suite run through a quantized serving path. The
    delta gate exists because probabilities degrade well before argmax does --
    an accuracy-only check would wave through a KV bit-width that quietly
    destroyed calibration.
    """
    gates: list[GateResult] = []
    if result.calibration is None:
        raise ValueError("calibration suite produced no scored questions")

    limit = CALIBRATION_GATES[f"{tier}_max_ece"]
    gates.append(
        GateResult(f"{tier}_ece", result.calibration.ece, limit, result.calibration.ece <= limit)
    )
    gates.append(
        GateResult(
            f"{tier}_adaptive_ece",
            result.calibration.adaptive_ece,
            limit,
            result.calibration.adaptive_ece <= limit,
        )
    )
    if quantized is not None:
        if quantized.calibration is None:
            raise ValueError("quantized run produced no scored questions")
        delta = abs(quantized.calibration.ece - result.calibration.ece)
        delta_limit = CALIBRATION_GATES["max_quantization_ece_delta"]
        gates.append(GateResult("quantization_ece_delta", delta, delta_limit, delta <= delta_limit))
    return gates
