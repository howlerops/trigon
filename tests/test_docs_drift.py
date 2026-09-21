"""The numbers in the docs are the numbers in the code, or the build fails.

`CLAUDE.md`: *one source of truth per fact -- budgets live in
`src/trigon/limits.py` ... do not duplicate any of them.* The prose duplicates
them anyway, because a decisions document that says "see limits.py" is not a
decisions document. `tests/test_openapi_drift.py` already settles the same
tension for the contract: keep the copy, and make the copy's correctness a
test.

What this does **not** do is read a number out of the docs and use it. It
reads the docs and asserts they agree with `trigon.limits`. The direction
matters -- if this ever became the source, the duplication would be real.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from trigon.limits import (
    CALIBRATION_GATES,
    COMPAT_BUDGET,
    CONFORMAL_COVERAGE_SIGMAS,
    DEFAULT_BUDGET,
    MAX_FLOOR_FRACTION_OF_GATE,
    MIN_ACCURACY_OVER_BASELINE,
    MIN_CALIBRATION_SAMPLES,
)

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

# The row label in `docs/decisions.md` -> the `Budget` field it restates.
BUDGET_ROWS = {
    "Total request": "context_tokens",
    "State + longest question": "single_question_envelope",
    "State": "state_tokens",
    "Schema": "schema_tokens",
    "Readout slots": "readout_tokens",
    "Questions per request": "max_questions",
}


def _number(cell: str) -> int:
    """`**524,288**` -> 524288. Bold is emphasis, not data."""
    return int(cell.replace("**", "").replace(",", "").strip())


def test_the_budget_table_matches_the_budgets():
    """`docs/decisions.md` §1 restates both budgets in full."""
    text = (DOCS / "decisions.md").read_text()
    labels = "|".join(re.escape(label) for label in BUDGET_ROWS)
    rows = re.findall(rf"^\| ({labels}) \| ([\d,*]+) \| ([\d,*]+) \|", text, re.M)
    found = {label: (compat, extended) for label, compat, extended in rows}

    # A row silently renamed out of the table is the drift this exists to
    # catch, so absence is a failure rather than a skip.
    assert set(found) == set(BUDGET_ROWS), f"missing rows: {set(BUDGET_ROWS) - set(found)}"

    for label, field in BUDGET_ROWS.items():
        compat, extended = found[label]
        assert _number(compat) == getattr(COMPAT_BUDGET, field), f"{label}, compat column"
        assert _number(extended) == getattr(DEFAULT_BUDGET, field), f"{label}, extended column"


def test_the_prose_option_trigger_and_shortlist_match():
    """§1's opening paragraph states the retrieval triggers in words."""
    text = (DOCS / "decisions.md").read_text()
    trigger = re.search(r"more than \*\*([\d,]+) options\*\*", text)
    shortlist = re.search(r"\*\*([\d,]+)-option shortlist\*\*", text)
    envelope = re.search(r"over \*\*([\d,]+) tokens\*\*", text)
    assert trigger and shortlist and envelope, "the prose triggers were reworded"
    assert _number(trigger.group(1)) == DEFAULT_BUDGET.retrieval_option_trigger
    assert _number(shortlist.group(1)) == DEFAULT_BUDGET.shortlist_size
    # A derived number, not a configured one: `max_question_tokens` is the
    # per-question envelope less state, so this 65,536 goes stale the moment
    # *either* input moves and the prose keeps a figure nothing computes.
    assert _number(envelope.group(1)) == DEFAULT_BUDGET.max_question_tokens


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        (r"\| `sample_size` \| ≥ ([\d,]+) \|", MIN_CALIBRATION_SAMPLES),
        (r"\| `accuracy_over_baseline` \| ≥ \+([\d.]+) \|", MIN_ACCURACY_OVER_BASELINE),
        (
            r"\| Workhorse ECE \(and adaptive ECE\) \| ≤ ([\d.]+) \|",
            CALIBRATION_GATES["workhorse_max_ece"],
        ),
        (r"\| Premium ECE \| ≤ ([\d.]+) \|", CALIBRATION_GATES["premium_max_ece"]),
        (
            r"\| Quantized-vs-BF16 ECE delta \| ≤ ([\d.]+) \|",
            CALIBRATION_GATES["max_quantization_ece_delta"],
        ),
    ],
)
def test_the_published_gate_limits_are_the_enforced_ones(pattern, expected):
    """`docs/evals.md` §1 is the table a reader trusts before running anything.

    A gate quietly relaxed in code and left alone in the docs is worse than an
    unpublished gate: the docs would then certify a threshold nothing enforces.
    `CLAUDE.md` forbids relaxing a gate to make a run pass; this makes the
    forbidden move visible in a diff rather than only in a review.
    """
    text = (DOCS / "evals.md").read_text()
    match = re.search(pattern, text)
    assert match, f"the gate table row matching {pattern!r} was reworded or removed"
    assert float(match.group(1).replace(",", "")) == pytest.approx(expected)


def test_the_conformal_coverage_gate_is_published_as_it_is_enforced():
    """Stated as a sigma count, because the floor is derived from the run.

    A fixed number here would be wrong at every sample size but one, which is
    the whole argument of the section it appears in.
    """
    text = (DOCS / "evals.md").read_text()
    assert re.search(r"\| `conformal_coverage` \| ≥ target − 3σ \|", text)
    assert CONFORMAL_COVERAGE_SIGMAS == 3.0


def test_the_testability_rule_is_stated_as_it_is_enforced():
    """`gate_is_testable` is published as a fraction, not a number."""
    text = (DOCS / "evals.md").read_text()
    assert re.search(r"\| `gate_is_testable` \| floor p95 ≤ ½ × limit \|", text)
    # Spelled "½" in prose and 0.5 in code; if the constant moves, the prose
    # has to be rewritten rather than have a digit changed, which is the point.
    assert MAX_FLOOR_FRACTION_OF_GATE == 0.5
