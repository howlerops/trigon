"""The whole product, in one pass: train, calibrate, save, serve, call, abstain.

Every other test covers a seam. This one covers the path a user actually takes,
because a stack whose parts each work is not the same as a stack that works —
this repository has shipped two components that were green in every unit test
and could not run at all end to end (the torch backend over HTTP, and the
seed sweep against a report it could not read).

Deliberately tiny: the point is that the path connects, not that the model is
good. Model quality is `trigon eval`'s job and the gates', and they say plainly
that it is not.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
from fastapi.testclient import TestClient

from trigon.server.app import build_app
from trigon.server.config import ServerConfig

torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sdk" / "python"))

TICKET = {
    "state": {"plan": "enterprise", "seats": 480, "open_tickets": 7, "payment_failed": True},
    "questions": {
        "plan": {
            "type": "choice",
            "instructions": "Which plan is this account on?",
            "options": [{"name": n} for n in ("free", "standard", "pro", "enterprise")],
        },
        "at_risk": {"type": "noul", "instructions": "Is this account at risk?"},
    },
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> dict:
    """Train a tiny model, calibrate it, and write everything a deployment needs.

    This is `trigon train` through its library entry points rather than the
    CLI, so a failure points at the step that broke.
    """
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.calibration.conformal import ConformalMethod, fit_conformal
    from trigon.calibration.temperature import TemperatureScaler
    from trigon.engine import Engine
    from trigon.evals import run_cases, synthetic_outcome_cases
    from trigon.training import TrainingConfig, train

    out = tmp_path_factory.mktemp("e2e")
    backend = TorchReadoutBackend(ReadoutConfig(d_model=64, n_layers=1), seed=0)
    compiler = backend.make_compiler()

    report = train(
        backend,
        synthetic_outcome_cases(n=240, seed=0, noise=0.2),
        TrainingConfig(epochs=2, learning_rate=0.01, accumulate=8, seed=0),
        compiler=compiler,
    )

    # Fit the temperature on a split the model never trained on and the gates
    # never read, as `trigon train` does. This said `seed=0` -- the same 240
    # cases the model had just been fitted to -- which is the defect that made
    # temperature scaling raise ECE on half the seeds of an 8,000-case sweep.
    # The path under test has to be the path that ships.
    engine = Engine(backend, compiler=compiler)
    fit_cases = synthetic_outcome_cases(n=240, seed=2000, noise=0.2)
    scaler = TemperatureScaler()
    rows: dict[str, list] = {}
    for outcome in run_cases(engine, fit_cases):
        for question in outcome.questions.values():
            if question.expected is None or question.expected.hard_label is None:
                continue
            import math

            logits = [math.log(max(p, 1e-12)) for p in question.probabilities]
            rows.setdefault(question.primitive, []).append((logits, question.expected.hard_label))
    for primitive, pairs in rows.items():
        if primitive == "noul":
            scaler.fit_binary([x[1] - x[0] for x, _ in pairs], [y for _, y in pairs])
        else:
            scaler.fit(primitive, [x for x, _ in pairs], [y for _, y in pairs])

    # A conformal profile, on a third split.
    calib = run_cases(engine, synthetic_outcome_cases(n=300, seed=99, noise=0.2))
    probs, labels = [], []
    for outcome in calib:
        question = outcome.questions["plan"]
        if question.expected is not None and question.expected.hard_label is not None:
            probs.append(list(question.probabilities))
            labels.append(question.expected.hard_label)
    predictor = fit_conformal(probs, labels, alpha=0.1, method=ConformalMethod.LAC)

    weights = out / "model.pt"
    temperatures = out / "temperatures.json"
    profiles = out / "profiles"
    backend.save(weights)
    scaler.save(temperatures)
    profiles.mkdir()
    predictor.save(profiles / "accounts.json")

    return {
        "weights": str(weights),
        "temperatures": str(temperatures),
        "conformal_dir": str(profiles),
        "version": backend.model_version,
        "report": report,
    }


def test_training_produces_a_named_servable_build(trained):
    """Step 1: a run leaves behind something a deployment can load."""
    assert trained["report"].final_loss > 0
    # Named after its weights, not after the code that made them.
    assert "+" in trained["version"] and not trained["version"].endswith("-untrained")
    assert pathlib.Path(trained["weights"]).exists()
    assert json.loads(pathlib.Path(trained["temperatures"]).read_text())["primitive"]


@pytest.fixture(scope="module")
def deployment(trained) -> TestClient:
    """Step 2: the gateway serves exactly those artifacts."""
    return TestClient(
        build_app(
            ServerConfig(
                backend="torch",
                weights=trained["weights"],
                temperature_path=trained["temperatures"],
                conformal_dir=trained["conformal_dir"],
            )
        )
    )


def test_the_deployment_reports_itself_honestly(deployment, trained):
    health = deployment.get("/healthz").json()
    assert health["status"] == "ok"
    # Both flags true only because this deployment really has both artifacts.
    assert health["trained"] is True
    assert health["calibrated"] is True
    assert deployment.get("/v1/models").json()["data"][0]["model"] == trained["version"]


def test_the_served_model_is_the_trained_one(deployment, trained):
    """Step 3: the answers come from those weights, not a fresh init."""
    from trigon.backends.torch_readout import TorchReadoutBackend
    from trigon.engine import Engine
    from trigon.types import SystemOneRequest

    served = deployment.post("/v1/systemone", json=TICKET).json()
    assert served["model"] == trained["version"]

    backend = TorchReadoutBackend.load(trained["weights"])
    direct = Engine(backend).answer(SystemOneRequest.model_validate(TICKET))
    # Uncalibrated in-process vs calibrated over HTTP, so not equal — but the
    # same weights must rank the options the same way.
    assert served["answers"]["plan"]["selected"] == direct.answers["plan"].selected


def test_the_sdk_drives_the_whole_thing(deployment, trained):
    """Step 4: the generated client, against this deployment."""
    trigon_client = pytest.importorskip("trigon_client")

    sdk = trigon_client.TrigonClient("http://testserver")

    def call(method, path, body=None):
        response = deployment.request(method, path, json=body)
        if response.status_code >= 400:
            raise trigon_client.TrigonError(response.status_code, response.json())
        return response.json()

    sdk._call = call

    answer = sdk.systemone(
        state=TICKET["state"],
        questions={
            "plan": trigon_client.choice(
                "Which plan is this account on?", ["free", "standard", "pro", "enterprise"]
            ),
            "at_risk": trigon_client.noul("Is this account at risk?"),
        },
        options={"conformal_profile": "accounts"},
    )

    assert answer.model == trained["version"]
    plan = answer.answers["plan"]
    assert isinstance(plan, trigon_client.ChoiceAnswer)
    assert plan.selected in {"free", "standard", "pro", "enterprise"}
    assert sum(plan.probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= plan.confidence <= 1.0

    # Step 5: the conformal wrapper is applied, and its promise is the one the
    # profile was fitted for.
    assert plan.coverage_target == pytest.approx(0.9)
    assert plan.prediction_set is not None
    assert set(plan.prediction_set) <= set(plan.probabilities)
    assert plan.selected in plan.prediction_set or len(plan.prediction_set) == 0

    # A Noul has no confidence field anywhere along this path.
    assert not hasattr(answer.answers["at_risk"], "confidence")


def test_the_deployment_refuses_what_the_contract_forbids(deployment):
    """Step 6: the guarantees hold on the served path, not just in unit tests."""
    trigon_client = pytest.importorskip("trigon_client")

    sdk = trigon_client.TrigonClient("http://testserver")

    def call(method, path, body=None):
        response = deployment.request(method, path, json=body)
        if response.status_code >= 400:
            raise trigon_client.TrigonError(response.status_code, response.json())
        return response.json()

    sdk._call = call

    # A Choice with one option has no answer to give.
    with pytest.raises(trigon_client.TrigonError) as one_option:
        sdk.systemone(state="x", questions={"q": trigon_client.choice("pick", ["only"])})
    assert one_option.value.status == 422

    # A state that does not fit is a 413, not a 500 and not a truncation.
    with pytest.raises(trigon_client.TrigonError) as too_big:
        sdk.systemone(state="word " * 80000, questions={"q": trigon_client.noul("ok?")})
    assert too_big.value.status == 413


def test_added_questions_do_not_move_the_others_on_the_served_path(deployment):
    """Step 7: the architectural claim, over HTTP, on a trained model.

    `tests/test_independence.py` proves this in-process on an untrained one.
    Here it has to survive compilation, calibration and serialisation.
    """
    before = deployment.post("/v1/systemone", json=TICKET).json()
    crowded = {
        "state": TICKET["state"],
        "questions": {
            **TICKET["questions"],
            **{
                f"filler_{i}": {"type": "noul", "instructions": f"Is fact {i} present?"}
                for i in range(8)
            },
        },
    }
    after = deployment.post("/v1/systemone", json=crowded).json()
    for qid in TICKET["questions"]:
        assert before["answers"][qid] == after["answers"][qid], f"{qid} moved"
