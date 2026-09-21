"""The migration harness: the artifact a caller actually decides on.

Nobody switches on a promise. They switch on a diff over their own traffic, and
the one number that decision rests on is *who is right where the two systems
disagree* — not agreement, which can be 100% between two systems that are both
always wrong in the same direction.

These pin the reductions that make heterogeneous answers comparable, and the
verdict arithmetic that turns a diff into a decision.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

MIGRATE = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "migrate.py"
_spec = importlib.util.spec_from_file_location("migrate", MIGRATE)
migrate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(migrate)


def test_every_primitive_reduces_to_one_decision():
    """Choice, Score and Noul say different things; comparing needs one notion.

    A Choice names a `selected` option, a Noul carries only a probability, and
    a Score carries **neither** -- it reports a `score` and a distribution over
    levels, so its decision is the modal level.

    This test used to assert `_selected({"level": "gold"}) == "gold"`, matching
    the implementation. No answer in this contract has a `level` key, so both
    sides of every Score comparison were None, None equalled None, and the
    harness reported 100% agreement on a question it had never compared. The
    test agreed with the code and both were wrong about the contract, which is
    why it is now written against `trigon.types` rather than against memory.
    """
    from trigon.types import ScoreAnswer

    assert "level" not in ScoreAnswer.model_fields
    assert migrate._selected({"selected": "billing", "confidence": 0.8}) == "billing"
    assert migrate._selected({"score": 2.0, "probabilities": {"silver": 0.3, "gold": 0.7}}) == (
        "gold"
    )
    assert migrate._selected({"probability": 0.81}) == "yes"
    assert migrate._selected({"probability": 0.19}) == "no"
    # Exactly even odds resolves to yes, and does so deterministically: a
    # coin-flip here would make two runs of the harness disagree with
    # themselves.
    assert migrate._selected({"probability": 0.5}) == "yes"
    assert migrate._selected({"unrecognised": 1}) is None


def test_a_noul_has_no_confidence_and_the_stand_in_is_labelled():
    """The contract gives a Noul no confidence field, on purpose.

    Its distance from even odds is a different quantity, and the harness says
    so wherever it reports it. What is asserted here is the arithmetic: even
    odds is no information, and certainty either way is full confidence.
    """
    assert migrate._confidence({"probability": 0.5}) == pytest.approx(0.0)
    assert migrate._confidence({"probability": 1.0}) == pytest.approx(1.0)
    assert migrate._confidence({"probability": 0.0}) == pytest.approx(1.0)
    assert migrate._confidence({"confidence": 0.42}) == pytest.approx(0.42)
    assert migrate._confidence({"selected": "x"}) is None


def test_the_harness_runs_end_to_end_against_a_command(tmp_path, capsys):
    """A baseline that is not an HTTP service still gets compared.

    A prompted LLM behind a script, or a rules engine, should not have to be
    wrapped in a server before it can be diffed against.
    """
    import json
    import sys

    requests = [
        {
            "state": {"plan": plan, "seats": 100, "payment_failed": True, "open_tickets": 5},
            "questions": {
                "plan": {
                    "type": "choice",
                    "instructions": "Which plan?",
                    "options": [{"name": n} for n in ("free", "standard", "pro")],
                }
            },
        }
        for plan in ("free", "pro", "pro")
    ]
    labels = [{"plan": plan} for plan in ("free", "pro", "pro")]

    request_file = tmp_path / "reqs.jsonl"
    label_file = tmp_path / "labels.jsonl"
    request_file.write_text("\n".join(json.dumps(r) for r in requests))
    label_file.write_text("\n".join(json.dumps(item) for item in labels))

    # An incumbent that always says "free": right once, wrong twice.
    baseline = tmp_path / "baseline.py"
    baseline.write_text(
        "import json,sys\n"
        "json.load(sys.stdin)\n"
        'print(json.dumps({"answers": {"plan": {"selected": "free", "confidence": 0.9}}}))\n'
    )

    argv = sys.argv
    sys.argv = [
        "migrate.py",
        str(request_file),
        "--incumbent-cmd",
        f"{sys.executable} {baseline}",
        "--labels",
        str(label_file),
    ]
    try:
        assert migrate.main() == 0
    finally:
        sys.argv = argv

    printed = capsys.readouterr().out
    assert "3 of 3 requests compared" in printed
    # The verdict section is the point of the tool.
    assert "disagreements with a label" in printed
    assert "this project right" in printed


def test_mismatched_labels_are_refused_rather_than_zipped(tmp_path):
    """Requests and labels join by position and nothing else.

    Silently truncating to the shorter would score one system against another
    system's answers, which is worse than refusing.
    """
    import json
    import sys

    request_file = tmp_path / "r.jsonl"
    label_file = tmp_path / "l.jsonl"
    request_file.write_text(
        "\n".join(json.dumps({"state": "x", "questions": {}}) for _ in range(3))
    )
    label_file.write_text(json.dumps({"q": "a"}))

    argv = sys.argv
    sys.argv = [
        "migrate.py",
        str(request_file),
        "--incumbent-cmd",
        "true",
        "--labels",
        str(label_file),
    ]
    try:
        with pytest.raises(SystemExit, match="align"):
            migrate.main()
    finally:
        sys.argv = argv


def test_it_refuses_to_run_with_nothing_to_compare_against(tmp_path):
    import sys

    request_file = tmp_path / "r.jsonl"
    request_file.write_text("{}")
    argv = sys.argv
    sys.argv = ["migrate.py", str(request_file)]
    try:
        with pytest.raises(SystemExit, match="nothing to compare"):
            migrate.main()
    finally:
        sys.argv = argv
