"""The cost tables in docs/architecture.md must be measurements, not prose.

They are the evidence for the claim ``trigon.limits`` is built on -- that
schema is nearly free to grow and state is the one axis that costs
quadratically. The first version of that section was hand-written and did not
reproduce from the compiler, which is the failure this test exists to prevent:
a number in a doc is only worth reading if something re-derives it.
"""

from __future__ import annotations

import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = _ROOT / "docs" / "architecture.md"
SCRIPT = _ROOT / "scripts" / "attention_table.py"

_spec = importlib.util.spec_from_file_location("attention_table", SCRIPT)
attention_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(attention_table)


def _checked_in() -> str:
    text = DOC.read_text()
    _, _, rest = text.partition(attention_table.START)
    body, marker, _ = rest.partition(attention_table.END)
    assert marker, f"{DOC} has no {attention_table.END} marker"
    return body.strip()


def test_checked_in_tables_match_what_the_code_produces():
    assert _checked_in() == attention_table.render().strip(), (
        "docs/architecture.md is stale -- run: python scripts/attention_table.py --write"
    )


def test_state_is_the_axis_that_costs():
    """The claim the budgets rest on, asserted rather than eyeballed."""
    state_heavy, schema_heavy, *_ = (attention_table.measure(*r) for r in attention_table.ROWS)
    # When state dominates, the mask can save almost nothing: state attends to
    # itself and that term is quadratic in the whole of it.
    assert state_heavy["saving"] < 0.15
    # When schema dominates, it saves nearly everything, because each
    # question's block is isolated and the cost is a sum of small squares.
    assert schema_heavy["saving"] > 0.80


def test_the_same_token_count_costs_very_differently_by_shape():
    """Why there is no single 'a full-size request costs X' number."""
    rows = attention_table.envelope()
    tokens = [r["tokens"] for r in rows]
    assert max(tokens) - min(tokens) < 0.01 * min(tokens), "rows should carry ~equal tokens"
    costs = [r["dense"] for r in rows]
    # Fewest, largest questions is the expensive end -- monotonically so.
    assert costs == sorted(costs, reverse=True)
    assert max(costs) > 2 * min(costs)
