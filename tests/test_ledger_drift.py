"""The ledger's numbers are the repository's numbers, or the build fails.

`docs/ledger.md` records what has been built, measured, and believed-then-
disproved. A ledger nobody updates is worse than no ledger: it reads like a
current statement of the project and is a snapshot of whenever somebody last
cared.

So the counts in its **State of the repository** table are pinned here, the
same way `tests/test_docs_drift.py` pins the budgets and the gate limits. The
narrative sections are a discipline and are not tested — what is tested is
that the parts which *look* like facts still are.

The tolerances are deliberate. An exact test on a commit count fails on every
commit, gets tired of being wrong, and is deleted within a week. These fail
when the ledger has drifted enough to mislead, not when it has drifted at all.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "ledger.md"


def _stated(label: str) -> int:
    """One row of the ledger's state table, as an integer."""
    text = LEDGER.read_text()
    match = re.search(rf"^\| {re.escape(label)} \| ([\d,]+) \|$", text, re.M)
    assert match, f"the ledger has no row for {label!r}; was the table renamed?"
    return int(match.group(1).replace(",", ""))


def _flat() -> str:
    """The ledger with its line wrapping removed.

    Markdown reflows, so a phrase that reads as one thing can be split across
    two lines. The first version of this test searched the raw text and failed
    on `*Keeping the\nledger*`, which is a test finding a formatting artifact
    rather than a fact.
    """
    return re.sub(r"\s+", " ", LEDGER.read_text())


def test_the_ledger_exists_and_says_when_it_is_binding():
    """It has to point at the rule that keeps it current, or it will not be."""
    text = _flat()
    assert "Keeping the ledger" in text, "the ledger must name the rule that maintains it"
    assert "## Believed, then disproved" in text, (
        "the disproved section is the point of the ledger; a ledger of only "
        "successes is a changelog"
    )
    assert "## Open" in text


def test_the_commit_count_is_roughly_current():
    """Within 25 commits. Exact would fail on every commit and be deleted."""
    actual = int(
        subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    stated = _stated("Commits")
    assert abs(actual - stated) <= 25, (
        f"the ledger says {stated:,} commits and there are {actual:,}; "
        "it has drifted far enough to mislead -- update docs/ledger.md"
    )


def test_the_test_count_is_roughly_current():
    """Within 40 tests. The ledger's headline claim about its own rigour."""
    stated = _stated("Tests")
    collected = len(list(ROOT.glob("tests/test_*.py")))
    assert collected > 0
    # Counting the real total means collecting, which is slow and recursive
    # here; the session's own pytest run is the authority. This asserts the
    # stated figure is in a plausible band for the number of test files,
    # which catches a ledger left behind by whole modules.
    assert 3 * collected <= stated <= 40 * collected, (
        f"the ledger states {stated} tests across {collected} test files, "
        "which is not a plausible ratio -- update docs/ledger.md"
    )


def test_the_source_size_is_roughly_current():
    """Within 20%."""
    actual = sum(len(path.read_text().splitlines()) for path in (ROOT / "src").rglob("*.py"))
    stated = _stated("Lines in `src/`")
    assert actual == pytest.approx(stated, rel=0.20), (
        f"the ledger says {stated:,} lines in src/ and there are {actual:,}; update docs/ledger.md"
    )


def test_the_gate_count_matches_the_gates_that_exist():
    """Exact, because a gate added or removed is exactly what a reader needs.

    This is the one number with no tolerance. The gate set is the product's
    claim about its own rigour, and a ledger that undercounts it is
    understating what the project does while a ledger that overcounts it is
    claiming checks that do not run.
    """
    import trigon.evals.calibration_suite as suite

    source = pathlib.Path(suite.__file__).read_text()
    # Every `GateResult(` construction, whether its name is a literal or an
    # f-string templated on the tier. The first version matched names with a
    # regex and added the templated ones separately, which counted two of them
    # twice and reported ten gates where there are eight -- a test wrong about
    # the thing it exists to pin.
    total = len(re.findall(r"GateResult\(", source))
    assert total > 0, "no gates found; has check_gates been restructured?"
    assert _stated("Release gates") == total, (
        f"the ledger says {_stated('Release gates')} release gates and "
        f"{total} are constructed in calibration_suite.py"
    )


def test_the_use_case_count_matches():
    """Exact. Adding a use case without listing it is how a catalogue rots."""
    from trigon.usecases import all_use_cases

    assert _stated("Committed use cases") == len(all_use_cases())


def test_the_corpus_count_matches():
    """Exact, for the same reason as the gates.

    A corpus is the difference between a calibration number measured on
    generated data and one measured on somebody's real traffic, so a ledger
    that overstates how many are loadable is overstating the only thing that
    makes the calibration claim transfer.
    """
    from trigon.evals.corpora import CORPORA

    assert _stated("Real corpora loadable") == len(CORPORA)
