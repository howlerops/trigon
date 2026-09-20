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
from .calibration_suite import GateResult
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
) -> str:
    """The release report."""
    out: list[str] = ["# Eval report", ""]
    models = sorted({r.model for r in results})
    out.append(f"Model(s): {', '.join(models)}")
    out.append("")

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
        if all(g.passed for g in gates):
            out.append("All gates passed.")
        else:
            failed = [g.name for g in gates if not g.passed]
            out.append(f"**Blocked**: {', '.join(failed)}.")
        out.append("")

    if slices:
        out.append("## Per-domain calibration")
        out.append("")
        out.append("| Domain | n | Accuracy | Mean confidence | Overconfidence | ECE |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, rep in slices.items():
            out.append(
                f"| {name} | {rep.n} | {rep.accuracy:.3f} | {rep.mean_confidence:.3f} | "
                f"{rep.overconfidence:+.3f} | {rep.ece:.4f} |"
            )
        out.append("")

    primary = next((r for r in results if r.calibration is not None), None)
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
) -> str:
    return json.dumps(
        {
            "results": [r.to_dict() for r in results],
            "gates": [
                {"name": g.name, "value": g.value, "limit": g.limit, "passed": g.passed}
                for g in gates
            ],
            "slices": {k: v.to_dict() for k, v in (slices or {}).items()},
        },
        indent=2,
        sort_keys=True,
    )


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"
