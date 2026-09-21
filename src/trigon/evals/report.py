"""Render eval results as Markdown and JSON.

Every release ships both: JSON so a run is machine-diffable against the last
one, Markdown so a reader can see the reliability diagram without running
anything. The reliability diagram is drawn as text on purpose -- it belongs in
the repo and in a terminal, not only in a notebook.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..calibration.metrics import CalibrationReport
from .calibration_suite import GateResult, blocking
from .harness import SuiteResult

__all__ = ["render_json", "render_markdown", "render_reliability"]


def render_reliability(calibration: CalibrationReport, width: int = 40) -> str:
    """A text reliability diagram: confidence vs accuracy, per bin.

    ``gap`` is signed. Positive is overconfident, which is the direction an
    RLHF'd baseline fails in and the direction users get hurt by.
    """
    lines = [
        "  bin            n    conf     acc     gap",
        f"  {'-' * (width + 30)}",
    ]
    for b in calibration.bins:
        if not b.count:
            continue
        filled = int(round(b.accuracy * width))
        bar = "#" * filled + "." * (width - filled)
        marker = int(round(b.mean_confidence * width))
        bar = bar[:marker] + "|" + bar[marker + 1 :] if marker < width else bar
        lines.append(
            f"  [{b.lower:.2f},{b.upper:.2f}) {b.count:5d}  {b.mean_confidence:.3f}  "
            f"{b.accuracy:.3f}  {b.gap:+.3f}  {bar}"
        )
    lines.append("  '#' = accuracy, '|' = mean confidence; aligned bars mean calibrated.")
    return "\n".join(lines)


def render_markdown(
    results: Sequence[SuiteResult],
    gates: Sequence[GateResult] = (),
    slices: dict[str, CalibrationReport] | None = None,
    cardinality: Sequence = (),
    workflows: Sequence = (),
    gated: SuiteResult | None = None,
) -> str:
    """The release report.

    ``gated`` is the suite the gates were computed from, and passing it is how
    the per-question table stops disagreeing with the gate above it. It used
    to take the first result carrying a breakdown, which is the *uncalibrated*
    run -- so on Banking77 seed 2 the gate said `intent` scored 0.7126 and the
    table three sections below said 0.7194, both true, neither labelled.

    Nobody noticed for as long as calibration never changed a decision. An
    isotonic map is monotone per class and not jointly, so it can reorder two
    classes and move the argmax; on the synthetic corpus it did not, and on a
    77-way choice it did.
    """
    out: list[str] = ["# Eval report", ""]
    models = sorted({r.model for r in results} | {w.model for w in workflows})
    out.append(f"Model(s): {', '.join(models)}")
    out.append("")

    if not results:
        out.append("_No calibration or jaggedness suites in this run._")
        out.append("")
        _append_cardinality(out, cardinality)
        _append_workflows(out, workflows)
        return "\n".join(out)

    out.append("## Suites")
    out.append("")
    out.append(
        "| Suite | Cases | Accuracy | ECE | Adaptive ECE | Brier | p50 ms | p99 ms | Tokens |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        c = r.calibration
        out.append(
            f"| {r.suite} | {r.n_cases} | {_fmt(r.accuracy)} | "
            f"{_fmt(c.ece if c else None)} | {_fmt(c.adaptive_ece if c else None)} | "
            f"{_fmt(c.brier if c else None)} | {r.latency_p50_ms:.1f} | "
            f"{r.latency_p99_ms:.1f} | {r.mean_prefill_tokens:.0f} |"
        )
    out.append("")

    extras = [r for r in results if r.extra]
    if extras:
        out.append("## Benchmark-specific metrics")
        out.append("")
        for r in extras:
            values = ", ".join(f"`{k}` {v:.3f}" for k, v in sorted(r.extra.items()))
            out.append(f"- **{r.suite}** — {values}")
        out.append("")

    if gates:
        out.append("## Release gates")
        out.append("")
        for gate in gates:
            out.append(f"- {gate}")
        out.append("")
        blockers = [g.name for g in blocking(gates) if not g.passed]
        if blockers:
            out.append(f"**Blocked**: {', '.join(blockers)}.")
        else:
            out.append("All blocking gates passed.")
        advisory_failures = [g.name for g in gates if g.advisory and not g.passed]
        if advisory_failures:
            out.append("")
            out.append(
                f"**Advisory, not blocking**: {', '.join(advisory_failures)}. "
                "Reported so the run does not read as clean when it is not — "
                "see `docs/evals.md` for what turns each one on."
            )
        out.append("")

    # The gated suite when the caller says which it is, and otherwise the
    # first with a breakdown -- the old behaviour, kept so `trigon eval`'s
    # single-suite calls are unaffected.
    source = gated if gated is not None and gated.per_question else None
    per_question = source.per_question if source else None
    if per_question is None:
        per_question = next((r.per_question for r in results if r.per_question), None)
    if per_question:
        out.append("## Accuracy per question")
        out.append("")
        out.append(
            "The pooled lift above averages over questions. A model that has learned "
            "one question and answers the rest by rote clears a pooled gate, so the "
            "breakdown is printed whether or not it is gated on."
        )
        if source is not None:
            out.append("")
            out.append(
                f"Measured on `{source.suite}`, the same run the gates read. "
                "Calibration can move a decision -- an isotonic map is monotone "
                "per class and not jointly -- so this can differ from the "
                "uncalibrated accuracy in the suites table above."
            )
        out.append("")
        out.append("| Question | n | Accuracy | Its marginal predictor | Lift |")
        out.append("| --- | ---: | ---: | ---: | ---: |")
        for q in sorted(per_question.values(), key=lambda q: q.lift):
            mark = "" if q.lift >= 0 else " ⚠"
            out.append(
                f"| `{q.question_id}` | {q.n:,} | {q.accuracy:.4f} | "
                f"{q.baseline_accuracy:.4f} | {q.lift:+.4f}{mark} |"
            )
        out.append("")

    # Two cuts, two tables. The per-primitive one exists because a temperature
    # is fitted per primitive, so that is the unit at which a fit can go wrong,
    # and the pooled number cannot name the part that failed.
    for prefix, heading, label in (
        ("domain:", "## Per-domain calibration", "Domain"),
        ("primitive:", "## Per-primitive calibration", "Primitive"),
    ):
        rows = {
            k.removeprefix(prefix): v for k, v in (slices or {}).items() if k.startswith(prefix)
        }
        if not rows:
            continue
        out.append(heading)
        out.append("")
        # Adaptive ECE beside ECE, because the two disagree and the run is
        # gated on both. On one seed's Noul head they read 0.0251 and 0.1625
        # over the same answers: a binary head's confidences cluster, and
        # equal-width bins average the cluster into one number while
        # equal-mass bins resolve it. A table carrying only the first says
        # that head is fine while the adaptive gate fails the run.
        out.append(
            f"| {label} | n | Accuracy | Mean confidence | Overconfidence | ECE | Adaptive ECE |"
        )
        out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, rep in rows.items():
            out.append(
                f"| {name} | {rep.n} | {rep.accuracy:.3f} | {rep.mean_confidence:.3f} | "
                f"{rep.overconfidence:+.3f} | {rep.ece:.4f} | {rep.adaptive_ece:.4f} |"
            )
        out.append("")

    _append_cardinality(out, cardinality)
    _append_workflows(out, workflows)

    primary = next((r for r in results if r.calibration is not None), None)
    if primary and primary.calibration is not None and primary.calibration.floor:
        floor = primary.calibration.floor
        cal = primary.calibration
        out.append("## Is this number evidence?")
        out.append("")
        out.append(
            f"On these {floor.n} predictions a perfectly calibrated model scores a mean "
            f"ECE of {floor.mean:.4f} (95th percentile {floor.p95:.4f}), simulated over "
            f"{floor.trials} resamples. The measured ECE is {cal.ece:.4f}."
        )
        out.append("")
        if cal.distinguishable:
            out.append(
                "The measured error is **above** that floor, so the miscalibration is real "
                "rather than sampling noise."
            )
        else:
            out.append(
                "The measured error is **within** that floor, so this model is "
                "statistically indistinguishable from perfectly calibrated at this "
                "sample size. That is the strongest claim the data supports — it is not "
                "the same as proving the error is zero."
            )
        out.append("")

    if primary and primary.calibration is not None:
        out.append("## Reliability diagram")
        out.append("")
        out.append("```")
        out.append(render_reliability(primary.calibration))
        out.append("```")
        out.append("")
    return "\n".join(out)


def render_json(
    results: Sequence[SuiteResult],
    gates: Sequence[GateResult] = (),
    slices: dict[str, CalibrationReport] | None = None,
    cardinality: Sequence = (),
    workflows: Sequence = (),
    gated: SuiteResult | None = None,
) -> str:
    """The same run as ``render_markdown``, machine-readable.

    It takes the same arguments for a reason, and a test pins that: a report
    and its ``.json`` sibling that disagree about which suites ran is worse
    than having only one of them. ``gated`` is emitted as a name rather than
    used to filter, because this file carries every suite in full and a
    consumer comparing two of them needs to know which one the verdict came
    from -- the uncalibrated and calibrated runs can differ in *accuracy*, not
    only in ECE, when a calibrator reorders two classes.

    ``passed`` is the CLI's own exit condition, so a consumer reading this file
    reaches the same verdict the command did.

    Each gate carries ``advisory``, which the markdown has always printed and
    this file used to omit. That omission was not cosmetic: the top-level
    ``passed`` is computed over ``blocking(gates)`` alone, so a consumer
    re-deriving the verdict from the gate list got a stricter answer than the
    command's -- `scripts/seed_sweep.py` did exactly that and reported a
    configuration as certifying on none of four seeds when it certified on
    three.
    """
    return json.dumps(
        {
            "results": [r.to_dict() for r in results],
            "gates": [
                {
                    "name": g.name,
                    "value": g.value,
                    "limit": g.limit,
                    "passed": g.passed,
                    "advisory": g.advisory,
                }
                for g in gates
            ],
            "gated": gated.suite if gated is not None else None,
            "slices": {k: v.to_dict() for k, v in (slices or {}).items()},
            "cardinality": [c.to_dict() for c in cardinality],
            "workflows": [w.to_dict() for w in workflows],
            "passed": all(g.passed for g in blocking(gates)) and all(c.passed for c in cardinality),
        },
        indent=2,
        sort_keys=True,
    )


def _append_cardinality(out: list[str], cardinality: Sequence) -> None:
    """Decision D1's falsifier, swept across option count and query difficulty."""
    if not cardinality:
        return
    out.append("## Cardinality recall gate")
    out.append("")
    out.append("| Options | Fields stated | Shortlist | Recall | Limit | |")
    out.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    for r in cardinality:
        out.append(
            f"| {r.options:,} | {4 - r.drop_slots} of 4 | {r.shortlist:,} | "
            f"{r.recall:.4f} | {r.limit:.2f} | {'PASS' if r.passed else '**FAIL**'} |"
        )
    out.append("")


def _append_workflows(out: list[str], workflows: Sequence) -> None:
    """Scored against resolved outcomes, with cost on the same run."""
    if not workflows:
        return
    out.append("## Workflows")
    out.append("")
    out.append(
        "| Workflow | Cases | Outcome accuracy | Model calls / case | Tokens | p50 ms | p99 ms |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for w in workflows:
        out.append(
            f"| {w.workflow} | {w.n_cases} | {_fmt(w.outcome_accuracy)} | "
            f"{w.mean_model_calls:.2f} | {w.mean_prefill_tokens:.0f} | "
            f"{w.latency_p50_ms:.1f} | {w.latency_p99_ms:.1f} |"
        )
    out.append("")


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"
