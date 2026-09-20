"""The gateway contract, including its error codes and the escalation path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trigon.backends.base import BackendOutput, QuestionOutput
from trigon.backends.lexical import LexicalBackend
from trigon.engine import Engine, EngineConfig
from trigon.server.app import build_app
from trigon.server.config import ServerConfig
from trigon.server.routing import RoutingPolicy, TieredRouter


@pytest.fixture
def client() -> TestClient:
    return TestClient(build_app(ServerConfig(backend="lexical")))


ROUTING_BODY = {
    "state": "the card transaction was refused at the till",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "Route this ticket.",
            "options": [
                {"name": "card_declined", "criteria": "a card transaction was refused"},
                {"name": "lost_luggage", "criteria": "baggage missing after a flight"},
            ],
        },
        "urgent": {"type": "noul", "instructions": "Needs a human within the hour?"},
    },
}


def test_answers_every_declared_question(client):
    body = client.post("/v1/systemone", json=ROUTING_BODY).json()
    assert set(body["answers"]) == {"intent", "urgent"}
    assert body["answers"]["intent"]["selected"] == "card_declined"
    assert "confidence" not in body["answers"]["urgent"]


def test_response_names_a_pinned_version_not_an_alias(client):
    body = client.post("/v1/systemone", json=ROUTING_BODY).json()
    # A version with no digits would be an alias, and aliases move under users.
    assert any(ch.isdigit() for ch in body["model"])


def test_malformed_schema_is_rejected_before_the_model_runs(client):
    bad = {
        "state": "x",
        "questions": {
            "q": {"type": "choice", "instructions": "pick", "options": [{"name": "only"}]}
        },
    }
    assert client.post("/v1/systemone", json=bad).status_code == 422


def test_unknown_question_type_is_rejected(client):
    bad = {"state": "x", "questions": {"q": {"type": "vibes", "instructions": "hmm"}}}
    assert client.post("/v1/systemone", json=bad).status_code == 422


def test_oversized_state_returns_413_not_422(client):
    """The request is well-formed; it just does not fit."""
    body = {
        "state": "word " * 80000,  # ~100k tokens, over the 65,536-token state budget
        "questions": {"q": {"type": "noul", "instructions": "ok?"}},
    }
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "schema_too_large"


def test_healthz_admits_when_a_deployment_is_uncalibrated(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["calibrated"] is False


def test_models_lists_the_configured_tiers(client):
    tiers = client.get("/v1/models").json()["data"]
    assert [t["tier"] for t in tiers] == ["workhorse"]


class ConfidentBackend:
    model_version = "premium-1.0.0"

    def infer(self, compiled, request):
        return BackendOutput(
            outputs={
                q.question_id: QuestionOutput(
                    question_id=q.question_id,
                    kind=q.kind,
                    logits=(9.0,) if q.kind == "noul" else (9.0,) + (0.0,) * (q.cardinality - 1),
                )
                for q in compiled.schema.questions
            },
            model_version=self.model_version,
            tier="premium",
        )


def test_low_confidence_questions_escalate_to_the_premium_tier():
    router = TieredRouter(
        Engine(LexicalBackend()),
        Engine(ConfidentBackend(), config=EngineConfig(tier="premium")),
        RoutingPolicy(escalate_below_confidence=0.9, escalate_noul_margin=0.5),
    )
    client = TestClient(build_app(ServerConfig(backend="lexical"), router=router))
    body = client.post("/v1/systemone", json=ROUTING_BODY).json()
    assert body["tier"] == "escalated"
    # Both tiers are named, because this request took two passes.
    assert "premium-1.0.0" in body["model"] and "lexical-floor" in body["model"]
    assert body["answers"]["intent"]["confidence"] > 0.9


def test_confident_requests_do_not_escalate():
    router = TieredRouter(
        Engine(LexicalBackend()),
        Engine(ConfidentBackend(), config=EngineConfig(tier="premium")),
        RoutingPolicy(escalate_below_confidence=0.0, escalate_noul_margin=0.0),
    )
    client = TestClient(build_app(ServerConfig(backend="lexical"), router=router))
    body = client.post("/v1/systemone", json=ROUTING_BODY).json()
    assert body["tier"] == "workhorse"


def test_request_level_threshold_overrides_the_deployment_policy():
    router = TieredRouter(
        Engine(LexicalBackend()),
        Engine(ConfidentBackend(), config=EngineConfig(tier="premium")),
        RoutingPolicy(escalate_below_confidence=0.0, escalate_noul_margin=0.0),
    )
    client = TestClient(build_app(ServerConfig(backend="lexical"), router=router))
    body = client.post(
        "/v1/systemone",
        json={**ROUTING_BODY, "options": {"escalate_below_confidence": 0.99}},
    ).json()
    assert body["tier"] == "escalated"


# -- serving a trained model -------------------------------------------------
#
# These exist because the whole suite was green while the torch backend could
# not be served at all: the gateway compiled with the character heuristic while
# the backend built tensors from its own tokenizer, and every request died with
# a 500. Nothing above this line touched the torch path over HTTP.


def _torch_body() -> dict:
    return {
        "state": "the invoice never arrived and the customer wants a refund",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Route this ticket.",
                "options": [
                    {"name": "billing", "criteria": "an invoice, charge or refund"},
                    {"name": "shipping", "criteria": "a parcel or a delivery"},
                ],
            },
            "urgent": {"type": "noul", "instructions": "Needs a human within the hour?"},
        },
    }


def _ask(config: ServerConfig, body: dict) -> dict:
    """POST one request at a gateway built from ``config``."""
    return TestClient(build_app(config)).post("/v1/systemone", json=body).json()


def _health(config: ServerConfig) -> dict:
    return TestClient(build_app(config)).get("/healthz").json()


def test_torch_backend_is_actually_servable():
    """The compiler must use the backend's tokenizer, not the heuristic."""
    pytest.importorskip("torch", reason="the reference model needs the 'train' extra")
    client = TestClient(build_app(ServerConfig(backend="torch")))
    response = client.post("/v1/systemone", json=_torch_body())
    assert response.status_code == 200, response.text
    assert set(response.json()["answers"]) == {"intent", "urgent"}


def test_gateway_and_cli_answer_identically(tmp_path):
    """Two entry points, one Engine. If they drift, one of them is lying."""
    pytest.importorskip("torch", reason="the reference model needs the 'train' extra")
    from trigon.cli import _engine
    from trigon.types import SystemOneRequest

    body = _torch_body()
    over_http = _ask(ServerConfig(backend="torch"), body)
    direct = _engine("torch", None, None).answer(SystemOneRequest.model_validate(body))

    assert over_http["answers"]["intent"]["probabilities"] == pytest.approx(
        direct.answers["intent"].probabilities
    )
    assert over_http["answers"]["urgent"]["probability"] == pytest.approx(
        direct.answers["urgent"].probability
    )


def test_serving_weights_changes_the_answers(tmp_path):
    """A loaded checkpoint must reach the model, not be accepted and ignored."""
    torch = pytest.importorskip("torch", reason="the reference model needs the 'train' extra")
    from trigon.backends.torch_readout import TorchReadoutBackend

    trained = TorchReadoutBackend(seed=0)
    with torch.no_grad():
        for parameter in trained.model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    checkpoint = tmp_path / "run.pt"
    trained.save(checkpoint)

    body = _torch_body()
    bare = _ask(ServerConfig(backend="torch"), body)
    loaded = _ask(ServerConfig(backend="torch", weights=str(checkpoint)), body)

    assert (
        loaded["answers"]["intent"]["probabilities"] != bare["answers"]["intent"]["probabilities"]
    )


def test_healthz_admits_when_a_deployment_is_untrained(tmp_path):
    """Untrained is the worse failure and looks identical from outside."""
    pytest.importorskip("torch", reason="the reference model needs the 'train' extra")
    from trigon.backends.torch_readout import TorchReadoutBackend

    assert _health(ServerConfig(backend="torch"))["trained"] is False

    checkpoint = tmp_path / "run.pt"
    TorchReadoutBackend(seed=0).save(checkpoint)
    assert _health(ServerConfig(backend="torch", weights=str(checkpoint)))["trained"] is True

    # The lexical floor has no weights to be missing.
    assert _health(ServerConfig(backend="lexical"))["trained"] is True


def test_response_distinguishes_an_untrained_model_from_a_trained_one(tmp_path):
    """``model_version`` is the only build identifier that reaches the caller."""
    pytest.importorskip("torch", reason="the reference model needs the 'train' extra")
    from trigon.backends.torch_readout import TorchReadoutBackend

    body = _torch_body()
    bare = _ask(ServerConfig(backend="torch"), body)
    assert bare["model"].endswith("-untrained")

    checkpoint = tmp_path / "run.pt"
    TorchReadoutBackend(seed=0).save(checkpoint)
    served = _ask(ServerConfig(backend="torch", weights=str(checkpoint)), body)
    assert not served["model"].endswith("-untrained")
    assert served["model"] != bare["model"]


def test_lexical_backend_rejects_weights_rather_than_ignoring_them():
    with pytest.raises(ValueError, match="no weights"):
        build_app(ServerConfig(backend="lexical", weights="reports/reference-run.pt"))


def test_healthz_counts_the_premium_tier_as_well(tmp_path):
    """An untrained premium tier is the harder one to notice: it answers only
    what the workhorse could not settle, which is the traffic nobody watches."""
    pytest.importorskip("torch", reason="the reference model needs the 'train' extra")
    from trigon.backends.torch_readout import TorchReadoutBackend

    untrained_premium = ServerConfig(backend="lexical", premium_backend="torch")
    assert _health(untrained_premium)["trained"] is False

    checkpoint = tmp_path / "premium.pt"
    TorchReadoutBackend(seed=0).save(checkpoint)
    assert (
        _health(
            ServerConfig(
                backend="lexical", premium_backend="torch", premium_weights=str(checkpoint)
            )
        )["trained"]
        is True
    )
