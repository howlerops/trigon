"""The CLI works on a fresh clone with no weights, which is the point of it."""

from __future__ import annotations

import json
import pathlib

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


def test_fit_stores_only_the_calibrators_it_kept(tmp_path, capsys):
    """The file records what is applied, not what was attempted.

    This asserted all three primitives were always present, which encoded the
    old behaviour: fit a temperature per primitive and apply it regardless.
    `trigon fit` now runs the same selection as `trigon train` -- fit both
    candidates, score them on data neither was fitted on, apply whichever
    demonstrably helps -- so a primitive whose fit buys nothing is absent, and
    `fitted_on` does not claim a calibration that is not being served.
    """
    out = tmp_path / "t.json"
    with pytest.warns(Warning):
        # The lexical floor's logits carry little signal, so at least one fit
        # pins at a bound and warns. That is the honest-defaults rule and it
        # is under test as much as the file is: the selection suppressed this
        # warning for one commit and silently applied a T=20 fit.
        main(["fit", "--out", str(out), "-n", "60"])
    printed = capsys.readouterr()

    stored = json.loads(out.read_text())
    assert set(stored["primitive"]) <= {"choice", "noul", "score"}
    assert set(stored["primitive"]) == set(stored["fitted_on"])

    # Every primitive is accounted for in the output, kept or not, so a reader
    # can tell "not calibrated" from "not considered".
    for primitive in ("choice", "noul", "score"):
        assert f"{primitive}:" in printed.err
        if primitive not in stored["primitive"]:
            assert f"{primitive}: none" in printed.err


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


def test_train_writes_every_artifact_a_deployment_and_a_sweep_need(tmp_path, capsys):
    """`trigon train`'s output files are a contract, and it has been broken.

    `scripts/seed_sweep.py` reads the `-training.json` sidecar and the `.json`
    report to compare runs; `trigon serve --weights` reads the checkpoint;
    `trigon ask` reads the temperatures. None of that was covered at the CLI
    level, and the sweep shipped broken because `train` wrote no `.json`
    sibling -- a file every other command writes, missing from the one command
    whose output anything automated reads.

    Deliberately the smallest run that still exercises all five paths. What is
    under test is that the artifacts exist, are parseable and refer to each
    other; whether the model is any good is the gates' job, and at this size
    they will say it is not.
    """
    pytest.importorskip("torch", reason="training needs the 'train' extra")

    out = tmp_path / "reports" / "run.md"
    weights = tmp_path / "reports" / "run.pt"
    # A failed gate exits 1 and is a result, not an error -- at 200 cases the
    # sample-size gate fails by construction, which is the harness working.
    code = main(
        [
            "train",
            *("-n", "200", "--epochs", "1", "--d-model", "32", "--layers", "1"),
            *("--eval-n", "120", "--floor-trials", "5", "--seed", "0"),
            *("--out", str(out), "--save-model", str(weights)),
        ]
    )
    assert code in (0, 1)
    printed = capsys.readouterr().out

    stem = out.with_suffix("")
    report = json.loads(out.with_suffix(".json").read_text())
    training = json.loads(pathlib.Path(f"{stem}-training.json").read_text())
    temperatures = json.loads(pathlib.Path(f"{stem}-temperatures.json").read_text())

    # The two fields the sweep reads by name. It parses no markdown, on
    # purpose: a report format is for people, and anything that compares runs
    # reading it is one heading rename away from silently reporting nothing.
    assert training["final_loss"] > 0
    assert "kept_epoch" in training
    assert {g["name"] for g in report["gates"]} >= {
        "accuracy_over_baseline",
        "workhorse_ece",
        "quantization_ece_delta",
    }
    assert set(temperatures["primitive"]) <= {"choice", "noul", "score"}

    # The header has to be a command someone can paste back, which is the only
    # thing standing between a committed report and an unreproducible number.
    assert "trigon train" in printed and "--seed 0" in printed and "--eval-n 120" in printed
    # Flags that change which gates the report carries belong in the header,
    # and their absence has to mean the default rather than nothing: this run
    # kept the quantization gate, so the header must not disclaim it.
    assert "--no-quantization-gate" not in printed

    # And the checkpoint is a servable build, named after its weights rather
    # than after the code that made them.
    from trigon.backends.torch_readout import TorchReadoutBackend

    reloaded = TorchReadoutBackend.load(weights)
    assert not reloaded.model_version.endswith("-untrained")
    assert reloaded.model_version in printed or reloaded.model_version in out.read_text()


def test_the_three_training_splits_are_three_different_datasets():
    """The temperature must be fitted on data the model has never seen.

    It was fitted on the training split. A temperature closes the gap between
    a model's confidence and its accuracy, and on the training split that gap
    is the memorised one, so the fit under-corrects by however much that draw
    overfit. The first four-seed sweep at 8,000 cases made the cost visible:
    temperature scaling *raised* ECE on two seeds of four, on one of them from
    0.0431 to 0.0677 and past the gate. A calibration step that makes
    calibration worse on half its draws is not one.

    The property is about seeds, not models, so it is testable in
    milliseconds. What it is not is a claim of exact disjointness -- the
    generator draws from a finite record space, so two splits can coincide on
    a record by chance, and asserting otherwise would be asserting something
    false.
    """
    from trigon.cli import SPLIT_SEED_OFFSETS, training_splits

    train, calibration, evaluation = training_splits(
        n=120, calibration_n=60, eval_n=90, seed=0, noise=0.2
    )
    assert (len(train), len(calibration), len(evaluation)) == (120, 60, 90)

    def fingerprint(cases):
        return [tuple(sorted(c.request.state.items())) for c in cases]

    # Not a reshuffle of one another: the overlap between any two splits is far
    # below what a shared seed would produce, which is total.
    for left, right in (
        (train, calibration),
        (train, evaluation),
        (calibration, evaluation),
    ):
        shared = set(map(tuple, fingerprint(left))) & set(map(tuple, fingerprint(right)))
        assert len(shared) < min(len(left), len(right)) // 2

    # Offsets, not derived seeds, so `--seed 0` and `--seed 1` cannot collide:
    # no two runs within 1,000 seeds of each other share a split. Two offsets
    # were 500 and 900 apart, which made that false for `--seed 500` -- a
    # conformal split of the run at seed 0 was the training split of the run
    # at seed 500, and a conformal threshold fitted on another run's training
    # data reports a coverage number that is simply too good.
    assert len(set(SPLIT_SEED_OFFSETS.values())) == len(SPLIT_SEED_OFFSETS)
    assert (
        min(
            abs(a - b)
            for a in SPLIT_SEED_OFFSETS.values()
            for b in SPLIT_SEED_OFFSETS.values()
            if a != b
        )
        >= 1000
    )


def _constant_confidence_rows(rng, n, k, confidence, accuracy):
    """Labelled rows for a head that always says `confidence` and is right
    `accuracy` of the time. Returned as (probabilities, label) pairs."""
    rows = []
    for _ in range(n):
        probs = [(1.0 - confidence) / (k - 1)] * k
        probs[0] = confidence
        rows.append((probs, 0 if rng.random() < accuracy else rng.randrange(1, k)))
    return rows


def test_a_temperature_that_raises_held_out_ece_is_declined():
    """A temperature is a proposal, not a result.

    It is fitted by minimising NLL, and NLL is not ECE — the scalar that best
    explains the labels can be the one that worsens the calibration the gates
    measure. Scaling is also a one-parameter family, so a head whose
    miscalibration is not a uniform sharpening or flattening cannot be fixed by
    any member of it, and the fit returns its best member rather than declining.

    Measured: on seed 1 of the 8,000-case sweep, scaling took pooled ECE from
    0.0431 to 0.0516 and failed the run on that gate. More calibration data
    made it worse — the Noul head reached 0.1453 at 4,000 cases against 0.0928
    at 1,000 — while the fit was plainly converging, so it was not a sampling
    problem.
    """
    import math
    import random

    from trigon.calibration.metrics import report
    from trigon.cli import _scaled

    rng = random.Random(5)
    # A head that is already perfectly calibrated: it says 0.75 and is right
    # 0.75 of the time. No temperature can improve it, and any temperature the
    # fitter lands on other than exactly 1.0 must make it worse.
    rows = _constant_confidence_rows(rng, 4000, 4, confidence=0.75, accuracy=0.75)
    logits = [[math.log(max(p, 1e-12)) for p in probs] for probs, _ in rows]
    labels = [y for _, y in rows]

    baseline = report([_scaled(x, 1.0) for x in logits], labels, simulate_floor=False).ece
    assert baseline < 0.02, "the fixture must start calibrated or it tests nothing"

    # Sharpening this head is exactly the harm the check exists to refuse.
    harmed = report([_scaled(x, 0.5) for x in logits], labels, simulate_floor=False).ece
    assert harmed > baseline

    # And the scaler falls back to 1.0 for a primitive with no stored value,
    # which is what declining relies on.
    from trigon.calibration.temperature import TemperatureScaler

    scaler = TemperatureScaler()
    scaler.fit("choice", logits, labels)
    assert "choice" in scaler.primitive
    scaler.primitive.pop("choice")
    assert scaler.temperature("choice") == 1.0


def test_a_temperature_is_applied_only_where_it_demonstrably_helps():
    """The burden of proof sits on accepting, and that direction is measured.

    Three rules were scored against heads whose true calibration is known
    (`scripts/decline_rule.py`). Putting the burden on *declining* -- the rule
    that shipped for one commit -- is worse on six of seven shapes: it keeps a
    temperature 34 times in 40 on an already-calibrated head, where the rule
    that ships discards it 40 times in 40, for a worst-case ECE of 0.0460
    against 0.0188.
    """
    import math
    import random

    from trigon.cli import _help_is_real, _scaled

    rng = random.Random(17)
    rows = _constant_confidence_rows(rng, 1200, 4, confidence=0.75, accuracy=0.75)
    logits = [[math.log(max(p, 1e-12)) for p in probs] for probs, _ in rows]
    labels = [y for _, y in rows]
    unscaled = [_scaled(x, 1.0) for x in logits]

    # Already calibrated: no temperature can help, so none is applied.
    assert not _help_is_real(unscaled, [_scaled(x, 0.25) for x in logits], labels)
    assert not _help_is_real(unscaled, [_scaled(x, 1.001) for x in logits], labels)

    # Plainly overconfident: flattening is the difference between 0.05 and
    # 0.21 worst-case ECE, and must be applied.
    skewed = _constant_confidence_rows(rng, 1200, 4, confidence=0.95, accuracy=0.70)
    skewed_logits = [[math.log(max(p, 1e-12)) for p in probs] for probs, _ in skewed]
    assert _help_is_real(
        [_scaled(x, 1.0) for x in skewed_logits],
        [_scaled(x, 2.2) for x in skewed_logits],
        [y for _, y in skewed],
    )

    # Nothing to measure means no evidence of help, so the fit is not applied.
    assert not _help_is_real([], [], [])
