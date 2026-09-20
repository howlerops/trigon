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
