"""The first code anyone runs is the README's example. It has to run.

A quickstart that raises is worse than no quickstart, and nothing else in the
suite would notice: the example is prose as far as Python is concerned. This
executes the block and checks the claims its trailing comments make, so a
rename or a contract change cannot quietly break the front page.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _first_python_block() -> str:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert blocks, "the README no longer has a Python example"
    return blocks[0]


def test_the_readme_quickstart_runs_and_says_what_it_claims():
    namespace: dict = {}
    exec(compile(_first_python_block(), str(README), "exec"), namespace)  # noqa: S102
    answers = namespace["response"].answers

    # "card_declined" -- the trailing comment on the first line.
    assert answers["intent"].selected == "card_declined"
    # "every declared option, summing to 1"
    assert set(answers["intent"].probabilities) == {"card_declined", "lost_luggage"}
    assert sum(answers["intent"].probabilities.values()) == pytest.approx(1.0)
    # "derived from the calibrated distribution" -- a real number in [0, 1].
    assert 0.0 <= answers["intent"].confidence <= 1.0
    # "expectation over the declared levels" -- inside the 1.0-5.0 scale, and
    # not a regression output that could land anywhere.
    assert 1.0 <= answers["severity"].score <= 5.0
    # "no confidence field, by design"
    assert not hasattr(answers["urgent"], "confidence")
    assert 0.0 <= answers["urgent"].probability <= 1.0


DOCS = sorted(
    [README, README.parent / "CLAUDE.md", *(README.parent / "docs").glob("*.md")],
)


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_documented_trigon_command_exists(doc):
    """A quickstart that names a subcommand the CLI does not have, or a flag it
    does not take, fails on the reader's first try and on nobody else's."""
    from trigon.cli import build_parser

    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices

    for block in re.findall(r"```bash\n(.*?)```", doc.read_text(), re.S):
        # Join continuations so a wrapped command is checked as one.
        text = block.replace("\\\n", " ")
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if not line.startswith("trigon "):
                continue
            name, *rest = line.split()[1:] or [""]
            assert name in subcommands, f"{doc.name}: no such command {name!r}"
            flags = [tok for tok in rest if tok.startswith("--")]
            known = {
                option for action in subcommands[name]._actions for option in action.option_strings
            }
            unknown = [f for f in flags if f not in known]
            assert not unknown, f"{doc.name}: 'trigon {name}' has no {unknown}"


def test_the_cookbook_runs():
    """A worked example that has stopped working is worse than none."""
    import subprocess

    script = README.parent / "examples" / "triage_cookbook.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=README.parent,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # The four things it exists to demonstrate.
    assert "tokens, one pass" in out
    assert "held for a human" in out
    assert "adding a fifth question moved the other answers: False" in out
    assert "out of pocket" in out


def test_every_committed_use_case_has_a_cookbook_that_runs():
    """A worked example that has stopped working is worse than none, and a
    use case with no worked example is a catalogue entry.

    Parameterised over `all_use_cases()` rather than a hardcoded list, so
    adding a use case without a runnable example fails here instead of
    quietly enlarging the catalogue.
    """
    import subprocess

    from trigon.usecases import all_use_cases

    script = README.parent / "examples" / "usecase_cookbook.py"
    for case in all_use_cases():
        result = subprocess.run(
            [sys.executable, str(script), case.name],
            capture_output=True,
            text=True,
            cwd=README.parent,
            check=False,
        )
        assert result.returncode == 0, f"{case.name}: {result.stderr}"
        out = result.stdout
        assert f"# {case.name}" in out
        # One answer per declared question, and the cost of the whole thing.
        for qid in case.questions:
            assert qid in out, f"{case.name}: no answer printed for {qid}"
        assert "prefill tokens" in out
        assert "## Routing" in out


def test_the_cookbook_never_prints_a_confidence_for_a_noul():
    """The contract gives a Noul no confidence field, on purpose. An example
    that invented `max(p, 1-p)` would put a number on screen that no model
    produced, in the one place a reader is learning what the shape is.
    """
    import subprocess

    from trigon.types import NoulQuestion
    from trigon.usecases import all_use_cases

    script = README.parent / "examples" / "usecase_cookbook.py"
    for case in all_use_cases():
        nouls = [q for q, spec in case.questions.items() if isinstance(spec, NoulQuestion)]
        if not nouls:
            continue
        out = subprocess.run(
            [sys.executable, str(script), case.name],
            capture_output=True,
            text=True,
            cwd=README.parent,
            check=False,
        ).stdout
        for qid in nouls:
            block = out.split(f"  {qid}\n", 1)[1].split("\n\n", 1)[0]
            assert "probability" in block
            # The *field*, not the word. The first version of this asserted
            # "confidence" was absent from the block and failed on the
            # cookbook's own parenthetical explaining that a Noul has none --
            # a test tripping over the sentence documenting the rule it
            # enforces, which is the second time that has happened here.
            fields = [line.strip().split()[0] for line in block.splitlines() if line.strip()]
            assert "confidence" not in fields, f"{case.name}.{qid} printed a confidence field"
        return
