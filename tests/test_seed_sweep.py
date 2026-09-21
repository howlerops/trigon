"""The sweep's verdict rule, tested without training anything.

`scripts/seed_sweep.py` is the script that decides whether a configuration is
allowed to be called certified, and once CI gates on it, whether main goes red.
That rule was reachable only by running the sweep -- twenty minutes of
training per invocation -- which is the same shape of problem as a release
gate whose only input is synthetic: the expensive part is the model, the part
that can be wrong is the arithmetic around it.

So the rule is a pure function over the rows, and this is it under test.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SWEEP = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "seed_sweep.py"
_spec = importlib.util.spec_from_file_location("seed_sweep", SWEEP)
seed_sweep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(seed_sweep)


def _row(seed: int, certified: bool, blocked: tuple[str, ...] = ()) -> dict:
    return {
        "seed": seed,
        "final_loss": 1.0,
        "kept_epoch": 1,
        "lift": 0.06,
        "ece": 0.01,
        "certified": certified,
        "blocked": list(blocked),
    }


def test_none_reports_without_gating():
    """The exploratory default: a sweep run to choose a configuration.

    Stopping at the first configuration that disappoints is how you end up
    having measured only the ones that worked.
    """
    rows = [_row(0, False, ("accuracy_over_baseline",)), _row(1, True)]
    code, line = seed_sweep.verdict(rows, "none")
    assert code == 0
    assert line == ""


def test_all_passes_only_when_every_seed_certifies():
    code, line = seed_sweep.verdict([_row(i, True) for i in range(4)], "all")
    assert code == 0
    assert "all 4 seeds" in line


@pytest.mark.parametrize("certified", [0, 1, 2, 3])
def test_all_fails_on_any_uncertified_seed(certified):
    """Three of four is a failure, not a pass with a caveat.

    This is the whole argument of `docs/decisions.md`, "A single-seed training
    run is not evidence": a configuration whose outcome depends on the draw has
    not been shown to work, and the majority reading of it is exactly the
    mistake -- the committed reference run was one lucky draw out of four.
    """
    blocked = ("workhorse_adaptive_ece",)
    rows = [_row(i, i < certified, () if i < certified else blocked) for i in range(4)]
    code, line = seed_sweep.verdict(rows, "all")
    assert code == 1
    assert "`workhorse_adaptive_ece`" in line


def test_an_empty_sweep_certifies_nothing():
    """Zero of zero is not "every seed passed".

    A sweep whose seeds all failed to launch would otherwise report success,
    which is the failure mode where a green CI job means the job did nothing.
    """
    code, line = seed_sweep.verdict([], "all")
    assert code == 1
    assert "No seeds ran" in line


def test_median_is_not_an_option():
    """Deliberately absent, and the absence is part of the design.

    The median seed certifying means half the draws do not. A release gate
    half of runs fail is the same broken instrument as an ECE gate sitting
    below its own noise floor, which this project refuses elsewhere.
    """
    assert "median" not in seed_sweep.REQUIREMENTS
    assert set(seed_sweep.REQUIREMENTS) == {"none", "all"}


def test_a_row_cannot_be_certified_and_blocked_at_once():
    """The two fields come from different places, so they can disagree.

    `certified` is the report's own `passed`; `blocked` is re-derived from its
    gate list. Re-deriving is what broke: the old code filtered on an
    `advisory` key the report format did not carry, counted every advisory
    failure as blocking, and reported an 8,000-case configuration as
    certifying on none of four seeds when three of four did.

    Both are kept, because the derived list is what a reader wants to see. So
    the contradiction has to be loud rather than resolved by preferring one.
    """
    row = _row(0, True)
    row["blocked"] = ["workhorse_ece"]
    with pytest.raises(RuntimeError, match="says it passed but lists blocking"):
        seed_sweep.verdict([row], "all")

    # And it is checked even in the mode that gates on nothing, because a
    # report contradicting itself is a broken report either way.
    with pytest.raises(RuntimeError):
        seed_sweep.verdict([row], "none")
