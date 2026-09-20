"""The CLI works on a fresh clone with no weights, which is the point of it."""

from __future__ import annotations

import json

import pytest

from trigon.cli import main


def test_ask_answers_a_request_from_a_file(tmp_path, capsys):
    request = tmp_path / "req.json"
    request.write_text(
        json.dumps(
            {
                "state": "the card transaction was refused",
                "questions": {
                    "intent": {
                        "type": "choice",
                        "instructions": "Route this ticket.",
                        "options": [
                            {"name": "card_declined", "criteria": "a card was refused"},
                            {"name": "lost_luggage", "criteria": "missing baggage"},
                        ],
                    }
                },
            }
        )
    )
    assert main(["ask", str(request)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["answers"]["intent"]["selected"] == "card_declined"


def test_eval_exits_non_zero_when_a_gate_fails(capsys):
    """CI can depend on this directly: a failed calibration gate blocks."""
    code = main(["eval", "calibration", "-n", "40"])
    out = capsys.readouterr().out
    assert code == 1
    assert "Release gates" in out
    assert "FAIL" in out


def test_eval_writes_both_report_formats(tmp_path, capsys):
    out = tmp_path / "reports" / "run.md"
    main(["eval", "jaggedness", "-n", "4", "--out", str(out)])
    capsys.readouterr()
    assert out.exists()
    payload = json.loads(out.with_suffix(".json").read_text())
    assert len(payload["results"]) == 9


def test_fit_saves_temperatures(tmp_path, capsys):
    out = tmp_path / "t.json"
    with pytest.warns(Warning):
        # The lexical floor's logits carry little signal, so the fit warns --
        # which is the behaviour under test as much as the file is.
        main(["fit", "--out", str(out), "-n", "60"])
    capsys.readouterr()
    assert set(json.loads(out.read_text())["primitive"]) == {"choice", "noul", "score"}


def test_spec_prints_the_contract(capsys):
    assert main(["spec"]) == 0
    assert "/v1/systemone" in json.loads(capsys.readouterr().out)["paths"]


def test_unknown_backend_fails_with_a_useful_message():
    with pytest.raises(SystemExit, match="lexical"):
        main(["ask", "-", "--backend", "nope"])


def test_eval_runs_the_cardinality_gate(capsys):
    main(["eval", "cardinality", "-n", "20", "--cardinality-max", "256"])
    out = capsys.readouterr().out
    assert "Cardinality recall gate" in out
    assert "Fields stated" in out


def test_eval_runs_the_workflow_suite(capsys):
    main(["eval", "workflow", "-n", "20"])
    out = capsys.readouterr().out
    assert "Workflows" in out
    assert "support_triage" in out and "moderation_queue" in out
    assert "Model calls / case" in out


def test_eval_all_includes_every_suite(capsys):
    main(["eval", "all", "-n", "20", "--cardinality-max", "256"])
    out = capsys.readouterr().out
    for section in ("## Suites", "Cardinality recall gate", "## Workflows", "Release gates"):
        assert section in out, section


def test_fit_writes_a_conformal_profile(tmp_path, capsys):
    import pytest as _pytest

    from trigon.calibration import ConformalPredictor

    profile = tmp_path / "profiles" / "default.json"
    with _pytest.warns(Warning):
        main(
            ["fit", "--out", str(tmp_path / "t.json"), "-n", "200", "--conformal-out", str(profile)]
        )
    capsys.readouterr()
    loaded = ConformalPredictor.load(profile)
    assert loaded.target_coverage == 0.9
    # One Choice question per case in the synthetic generator ("plan");
    # "size" is a Score and "at_risk" a Noul, and conformal sets are over
    # categorical labels.
    assert loaded.calibration_n == 200
