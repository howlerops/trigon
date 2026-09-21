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
from ..limits import (
    CALIBRATION_GATES,
    MAX_FLOOR_FRACTION_OF_GATE,
    MIN_ACCURACY_OVER_BASELINE,
    MIN_CALIBRATION_SAMPLES,
)
from .harness import Case, CaseOutcome, SuiteResult, run_cases, summarize

__all__ = ["blocking", "GateResult", "check_gates", "run_calibration_suite", "slice_reports"]


@dataclass(frozen=True)
class GateResult:
    """Whether a run may ship, and why not if it may not."""

    name: str
    value: float
    limit: float
    passed: bool
    #: Set when the gate is about the measurement rather than the model.
    note: str = ""
    #: Measured and printed, but not counted in the run's verdict. An advisory
    #: gate is one we intend to enforce and cannot yet -- it is reported so it
    #: cannot be quietly forgotten, which a gate that is merely switched off
    #: always is. Nothing may become advisory to make a run pass; see
    #: ``check_gates`` for the only one, and the condition that ends it.
    advisory: bool = False

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        if self.advisory:
            verdict = f"{verdict} (advisory)"
        base = f"{verdict} {self.name}: {self.value:.4f} (limit {self.limit:.4f})"
        return f"{base} -- {self.note}" if self.note else base


def blocking(gates: Sequence[GateResult]) -> list[GateResult]:
    """The gates that decide whether a run ships.

    One helper because the verdict is computed in four places -- the CLI's exit
    code twice, the markdown report and the JSON report -- and an advisory gate
    counted in one of them would be a blocking gate with extra steps.
    """
    return [g for g in gates if not g.advisory]


def run_calibration_suite(
    engine: Engine,
    cases: Sequence[Case],
    *,
    suite: str = "calibration",
    floor_trials: int = 200,
) -> tuple[SuiteResult, dict[str, CalibrationReport]]:
    """Run the suite and return the summary plus per-domain breakdowns."""
    outcomes = run_cases(engine, cases)
    result = summarize(suite, engine.backend.model_version, outcomes, floor_trials=floor_trials)
    return result, slice_reports(outcomes)


def slice_reports(outcomes: Sequence[CaseOutcome]) -> dict[str, CalibrationReport]:
    """Calibration broken out by domain **and** by primitive.

    An aggregate ECE hides where the miscalibration is. Per-domain was here
    from the start because that is the granularity users deploy at; per
    primitive was documented from the start and never implemented, and the
    absence had a cost.

    A temperature is fitted per primitive, so a primitive is exactly the unit
    at which a fit can go wrong -- and one did. On an 8,000-case run the Score
    head drew a fitted temperature of 0.20, sharpening a head that had learned
    nothing by five times, while Choice and Noul sat near 1.0. The pooled
    number said "ECE 0.0516, gate is 0.05" and gave a reader nothing to act
    on. It is the same argument that made `worst_question_over_baseline`
    advisory rather than absent: a pooled figure that cannot name the part
    that failed is a summary, not a measurement.

    Keys are prefixed, because a domain called `choice` would otherwise
    silently overwrite the primitive of that name.
    """
    by_domain: dict[str, list[tuple[tuple[float, ...], int]]] = {}
    by_primitive: dict[str, list[tuple[tuple[float, ...], int]]] = {}
    for outcome in outcomes:
        domain = outcome.case.domain
        for question in outcome.questions.values():
            if question.expected is None:
                continue
            truth = question.expected.hard_label
            if truth is None:
                continue
            row = (question.probabilities, truth)
            by_domain.setdefault(domain, []).append(row)
            by_primitive.setdefault(question.primitive, []).append(row)

    def build(buckets: dict[str, list], prefix: str) -> dict[str, CalibrationReport]:
        return {
            f"{prefix}{name}": report(
                [p for p, _ in rows],
                [y for _, y in rows],
                slice_name=f"{prefix}{name}",
                simulate_floor=False,
            )
            for name, rows in sorted(buckets.items())
            if rows
        }

    return build(by_domain, "domain:") | build(by_primitive, "primitive:")


def check_gates(
    result: SuiteResult,
    tier: str = "workhorse",
    quantized: SuiteResult | None = None,
    *,
    require_per_question: bool = False,
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
    cal = result.calibration

    # Gate the measurement before gating the model. A run that cannot tell a
    # calibrated model from a miscalibrated one must not certify either.
    gates.append(
        GateResult(
            "sample_size",
            float(cal.n),
            float(MIN_CALIBRATION_SAMPLES),
            cal.n >= MIN_CALIBRATION_SAMPLES,
            note="below this, ECE is dominated by estimator noise",
        )
    )
    if cal.floor is not None:
        ceiling = limit * MAX_FLOOR_FRACTION_OF_GATE
        gates.append(
            GateResult(
                "gate_is_testable",
                cal.floor.p95,
                ceiling,
                cal.floor.p95 <= ceiling,
                note=(
                    f"a perfectly calibrated model scores ECE {cal.floor.mean:.4f} "
                    f"on this run, p95 {cal.floor.p95:.4f}"
                ),
            )
        )
    # Before the calibration gates: did the model use the input at all? A
    # marginal predictor is calibrated by construction, so ECE cannot reject
    # it -- and a gate that certifies the one model guaranteed to be useless is
    # worse than no gate, because it looks like evidence.
    if result.baseline_accuracy is not None and result.accuracy is not None:
        lift = result.accuracy - result.baseline_accuracy
        gates.append(
            GateResult(
                "accuracy_over_baseline",
                lift,
                MIN_ACCURACY_OVER_BASELINE,
                lift >= MIN_ACCURACY_OVER_BASELINE,
                note=(
                    f"model {result.accuracy:.4f} vs marginal predictor "
                    f"{result.baseline_accuracy:.4f}; calibration cannot reject a "
                    f"model that ignores the state"
                ),
            )
        )

    worst = result.worst_question_lift
    if worst is not None:
        # The pooled gate above averages over questions, so a model that has
        # learned one question of three and answers the rest by rote clears it
        # -- the reference run does exactly that at +0.0639 pooled while two of
        # its three questions sit below their own marginal predictor. This gate
        # reads the worst single question instead.
        #
        # Advisory until phase 1. Not because the bar is wrong, but because at
        # a 128-wide two-layer spike every per-question gate fails, and a gate
        # that nothing can pass measures model capacity rather than model
        # honesty. It becomes blocking the moment a real backbone lands --
        # ``require_per_question=True``, which the phase-1 training command
        # sets -- and it is reported in the meantime so that flip cannot be
        # quietly skipped.
        offender = min(result.per_question.values(), key=lambda q: q.lift)
        gates.append(
            GateResult(
                "worst_question_over_baseline",
                worst,
                MIN_ACCURACY_OVER_BASELINE,
                worst >= MIN_ACCURACY_OVER_BASELINE,
                note=(
                    f"worst is {offender.question_id!r} at {offender.accuracy:.4f} vs its own "
                    f"marginal {offender.baseline_accuracy:.4f}; the pooled gate hides this"
                ),
                advisory=not require_per_question,
            )
        )

    gates.append(GateResult(f"{tier}_ece", cal.ece, limit, cal.ece <= limit))
    gates.append(
        GateResult(f"{tier}_adaptive_ece", cal.adaptive_ece, limit, cal.adaptive_ece <= limit)
    )
    if quantized is not None:
        if quantized.calibration is None:
            raise ValueError("quantized run produced no scored questions")
        delta = abs(quantized.calibration.ece - cal.ece)
        delta_limit = CALIBRATION_GATES["max_quantization_ece_delta"]
        gates.append(GateResult("quantization_ece_delta", delta, delta_limit, delta <= delta_limit))
    return gates
