"""The compatibility adapter, checked against the published shapes.

"Drop-in" is the strongest claim this project makes and the easiest one to
assert without earning. These tests earn it in the only two ways available
without running their service:

1. the **exact example bodies from their documentation** (docs.typesafe.ai,
   read 2026-09-21) are accepted and answered, and every field their example
   response carries comes back with the right name and the right type;
2. the compat path and the native path return the **same numbers** for the
   same question, so the adapter is a translation and not a second model.

The second is the one that catches a real regression. A translation layer that
quietly answers a different question -- by dropping a rubric, by reordering
levels, by folding criteria into the wrong slot -- still produces a
well-formed response, and a well-formed wrong answer is the failure mode this
whole project is against.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trigon.server.app import build_app
from trigon.server.compat import (
    CompatError,
    build_compat_app,
    to_native,
)
from trigon.server.config import ServerConfig
from trigon.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

# Their documented example, verbatim.
CHOICE_REQUEST = {
    "state": "My running shoes arrived in the wrong size. Can I swap them for a size 10?",
    "model": "jev-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "returns": "Exchanges, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
            },
        }
    },
}

SCORE_REQUEST = {
    "state": "The export button does nothing, but you can still copy the table by hand.",
    "model": "jev-latest",
    "questions": {
        "bug_severity": {
            "type": "score",
            "instructions": "How severe is this bug?",
            "criteria": [
                "Cosmetic; no impact to functionality",
                "Broken or degraded feature, but workaround exists",
                "Blocking issue; no workaround exists",
            ],
        }
    },
}

NOUL_REQUEST = {
    "state": "I have asked three times now. Can I please just talk to a real person?",
    "model": "jev-latest",
    "questions": {
        "is_human_escalation": {
            "type": "noul",
            "instructions": "Is the customer asking for a human agent?",
        },
        "is_repeat_contact": {
            "type": "noul",
            "instructions": "Has the customer contacted support about this before?",
            "criteria": {
                "true": "Mentions a prior attempt, ticket, or that they have asked before",
                "false": "No sign of any previous contact",
            },
        },
    },
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(build_compat_app(ServerConfig(backend="lexical")))


# -- the request mapping ----------------------------------------------------


def test_a_choice_criteria_map_becomes_options_in_the_order_written():
    """Map order is the option order, and the option order is the answer's."""
    native = to_native(CHOICE_REQUEST)
    question = native.questions["department"]
    assert isinstance(question, ChoiceQuestion)
    assert [option.name for option in question.options] == ["returns", "shipping", "billing"]
    assert question.options[0].criteria == "Exchanges, wrong or damaged items"


def test_a_score_criteria_array_becomes_index_named_levels():
    """Their `score` is a mean of level *indices*, so the names must be them."""
    native = to_native(SCORE_REQUEST)
    question = native.questions["bug_severity"]
    assert isinstance(question, ScoreQuestion)
    assert [level.name for level in question.levels] == ["0", "1", "2"]
    assert [level.value for level in question.levels] == [0.0, 1.0, 2.0]
    assert question.levels[2].criteria == "Blocking issue; no workaround exists"


def test_a_noul_keeps_its_criteria_instead_of_dropping_them():
    """Our Noul has nowhere to put them, so they go where the model reads.

    Dropping them would be silent: the answer would still be well-formed, and
    the caller's boundary description would simply never have been seen.
    """
    native = to_native(NOUL_REQUEST)
    plain = native.questions["is_human_escalation"]
    folded = native.questions["is_repeat_contact"]
    assert isinstance(folded, NoulQuestion)
    assert plain.instructions == "Is the customer asking for a human agent?"
    assert "Mentions a prior attempt" in folded.instructions
    assert "No sign of any previous contact" in folded.instructions
    assert folded.instructions.startswith("Has the customer contacted support about this before?")


def test_structured_instructions_and_criteria_are_serialised_not_summarised():
    """Their contract allows rubrics. A rubric that loses a clause is a bug
    that produces a plausible answer, which is the worst kind here."""
    payload = {
        "state": "x",
        "model": "jev-latest",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": {"field": {"name": "tone", "type": "string"}, "focus": "the close"},
                "criteria": {
                    "warm": {"what": "friendly", "not_for": "sarcasm", "examples": ["thanks!"]},
                    "cold": "curt",
                },
            }
        },
    }
    question = to_native(payload).questions["q"]
    assert "tone" in question.instructions and "the close" in question.instructions
    criteria = question.options[0].criteria
    for fragment in ("friendly", "sarcasm", "thanks!"):
        assert fragment in criteria, f"{fragment!r} was lost from the rubric"


@pytest.mark.parametrize(
    "payload, because",
    [
        ({"model": "m", "questions": {}}, "no state"),
        ({"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}, "no model"),
        ({"state": "x", "model": "m"}, "no questions"),
        ({"state": "x", "model": "m", "questions": {}}, "empty questions"),
        (
            {"state": "x", "model": "m", "questions": {"q": {"type": "rank", "instructions": "?"}}},
            "unknown type",
        ),
        (
            {
                "state": "x",
                "model": "m",
                "questions": {"q": {"type": "choice", "instructions": "?"}},
            },
            "a choice with no criteria",
        ),
        (
            {
                "state": "x",
                "model": "m",
                "questions": {"q": {"type": "score", "instructions": "?", "criteria": {}}},
            },
            "a score whose criteria is a map",
        ),
    ],
)
def test_requests_their_contract_would_reject_are_rejected(payload, because):
    with pytest.raises(CompatError):
        to_native(payload)


# -- the response mapping ---------------------------------------------------


def test_the_choice_response_carries_exactly_their_fields(client):
    body = client.post("/v1/systemone", json=CHOICE_REQUEST).json()
    assert set(body) == {"model", "answers", "usage"}
    answer = body["answers"]["department"]
    assert set(answer) == {"type", "choice", "confidence", "probabilities"}
    assert answer["type"] == "choice"
    assert answer["choice"] in {"returns", "shipping", "billing"}
    assert set(answer["probabilities"]) == {"returns", "shipping", "billing"}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert 0.0 <= answer["confidence"] <= 1.0


def test_the_score_response_carries_a_legend_rebuilt_from_the_request(client):
    body = client.post("/v1/systemone", json=SCORE_REQUEST).json()
    answer = body["answers"]["bug_severity"]
    assert set(answer) == {"type", "score", "confidence", "legend", "probabilities"}
    assert answer["legend"] == {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists",
    }
    assert set(answer["probabilities"]) == {"0", "1", "2"}
    # Their score is a probability-weighted mean of the level numbers.
    expected = sum(int(k) * v for k, v in answer["probabilities"].items())
    assert answer["score"] == pytest.approx(expected)
    assert 0.0 <= answer["score"] <= 2.0


def test_a_noul_answer_has_no_confidence_field(client):
    """Theirs does not carry one and neither does ours, for the same reason.

    An adapter that invented one -- `max(p, 1 - p)` is the obvious guess --
    would be adding a number no model produced to a response a caller trusts.
    """
    body = client.post("/v1/systemone", json=NOUL_REQUEST).json()
    for qid in ("is_human_escalation", "is_repeat_contact"):
        answer = body["answers"][qid]
        assert set(answer) == {"type", "noul"}
        assert 0.0 <= answer["noul"] <= 1.0


def test_usage_reports_prefill_as_input_and_zero_output(client):
    """Zero is the measurement. There is no decode half to charge for."""
    body = client.post("/v1/systemone", json=CHOICE_REQUEST).json()
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] == 0


def test_the_response_names_the_build_that_answered_not_the_one_requested(client):
    """`model: jev-latest` goes in; what comes back is what actually ran."""
    body = client.post("/v1/systemone", json=CHOICE_REQUEST).json()
    assert body["model"] != "jev-latest"
    assert body["model"]


# -- the property that makes it a translation rather than a second model ----


def test_the_compat_path_and_the_native_path_answer_identically():
    """The whole claim, in one assertion.

    Same stack, two front doors. If these ever disagree the adapter has
    started deciding something, and a caller comparing the two would be
    comparing two models while being told they are one.
    """
    native_app = build_app(ServerConfig(backend="lexical"))
    with TestClient(native_app) as http:
        for payload in (CHOICE_REQUEST, SCORE_REQUEST, NOUL_REQUEST):
            translated = to_native(payload)
            native = http.post(
                "/v1/systemone", json=translated.model_dump(mode="json", exclude_none=True)
            ).json()
            compat = http.post("/compat/v1/systemone", json=payload).json()
            for qid, answer in compat["answers"].items():
                mirror = native["answers"][qid]
                if answer["type"] == "noul":
                    assert answer["noul"] == pytest.approx(mirror["probability"])
                elif answer["type"] == "choice":
                    assert answer["choice"] == mirror["selected"]
                    assert answer["probabilities"] == pytest.approx(mirror["probabilities"])
                    assert answer["confidence"] == pytest.approx(mirror["confidence"])
                else:
                    assert answer["score"] == pytest.approx(mirror["score"])
                    assert answer["probabilities"] == pytest.approx(mirror["probabilities"])


# -- status codes, because a caller's retry logic is written against theirs --


def test_a_malformed_request_is_a_422_not_a_500(client):
    assert client.post("/v1/systemone", json={"state": "x"}).status_code == 422
    assert client.post("/v1/systemone", content=b"{not json").status_code == 422


def test_an_oversized_request_is_a_422_because_their_contract_has_no_413():
    """Ours is a 413 -- well-formed, does not fit. Theirs has no such code."""
    from trigon.limits import DEFAULT_BUDGET

    payload = {
        "state": "word " * (DEFAULT_BUDGET.state_tokens * 2),
        "model": "jev-latest",
        "questions": {"q": {"type": "noul", "instructions": "over budget?"}},
    }
    with TestClient(build_compat_app(ServerConfig(backend="lexical"))) as http:
        assert http.post("/v1/systemone", json=payload).status_code == 422
    with TestClient(build_app(ServerConfig(backend="lexical"))) as http:
        assert http.post("/compat/v1/systemone", json=payload).status_code == 422
        # And the native path still reports it the way our own contract does.
        # Pinned because a 422 is also what a *malformed* request gets: without
        # this the test would keep passing if the payload stopped being
        # oversized and started being invalid, which is a different code path
        # and not the one this test is about.
        native = http.post(
            "/v1/systemone", json=to_native(payload).model_dump(mode="json", exclude_none=True)
        )
        assert native.status_code == 413
        assert native.json()["error"]["type"] == "schema_too_large"


def test_anything_their_envelope_accepts_ours_accepts_too():
    """The superset claim, exercised rather than asserted.

    `COMPAT_BUDGET` reproduces their limits; a request at every one of them
    must compile under ours, or "drop-in" fails on the first large caller.
    """
    from trigon.limits import COMPAT_BUDGET

    payload = {
        "state": "word " * (COMPAT_BUDGET.state_tokens // 2),
        "model": "jev-latest",
        "questions": {
            f"q{i}": {"type": "noul", "instructions": "ok?"}
            for i in range(COMPAT_BUDGET.max_questions)
        },
    }
    with TestClient(build_compat_app(ServerConfig(backend="lexical"))) as http:
        assert http.post("/v1/systemone", json=payload).status_code == 200


# -- the other direction: calling their service -----------------------------


def test_our_request_translates_out_to_their_shape():
    """`scripts/migrate.py` points at the incumbent, so it must speak theirs."""
    from trigon.server.compat import to_compat_request

    native = to_native(SCORE_REQUEST)
    out = to_compat_request(native)
    assert out["model"] == "jev-latest"
    assert out["questions"]["bug_severity"]["type"] == "score"
    assert out["questions"]["bug_severity"]["criteria"] == SCORE_REQUEST["questions"][
        "bug_severity"
    ]["criteria"]


def test_an_option_without_a_description_sends_its_name_not_an_empty_string():
    """An empty description is a weaker prompt than no description, and it
    would make a migration comparison unfair to the incumbent."""
    from trigon.server.compat import to_compat_request
    from trigon.types import ChoiceQuestion, SystemOneRequest

    request = SystemOneRequest(
        state="x",
        questions={
            "q": ChoiceQuestion(
                instructions="Pick.", options=[{"name": "alpha"}, {"name": "beta"}]
            )
        },
    )
    assert to_compat_request(request)["questions"]["q"]["criteria"] == {
        "alpha": "alpha",
        "beta": "beta",
    }


def test_their_score_answer_comes_back_under_our_level_names():
    """Their levels are numbered; ours are named. Without this restoration a
    Score comparison is between two different label vocabularies and every
    case reads as a disagreement on a question they may agree about."""
    from trigon.server.compat import compat_answer_to_native
    from trigon.types import ScoreQuestion

    question = ScoreQuestion(
        instructions="How big?",
        levels=[{"name": n, "value": float(i)} for i, n in enumerate(["bronze", "silver", "gold"])],
    )
    theirs = {
        "type": "score",
        "score": 1.4,
        "confidence": 0.35,
        "legend": {"0": "small", "1": "medium", "2": "large"},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
    }
    ours = compat_answer_to_native(theirs, question)
    assert set(ours["probabilities"]) == {"bronze", "silver", "gold"}
    assert ours["selected"] == "gold"
    assert ours["probabilities"]["silver"] == pytest.approx(0.4)


def test_a_noul_comes_back_as_a_probability_with_no_confidence():
    from trigon.server.compat import compat_answer_to_native

    ours = compat_answer_to_native({"type": "noul", "noul": 0.93}, None)
    assert ours == {"type": "noul", "probability": 0.93}


def test_the_migration_harness_compares_a_score_instead_of_two_nones():
    """The regression that made this whole direction worth testing.

    `_selected` looked for a `level` key, which no answer in this contract
    carries, so both sides returned None, None equalled None, and the harness
    printed 100% agreement on a question it had never compared. A vacuous
    agreement number is worse than a missing one -- it is the one a reader
    acts on.
    """
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "migrate.py"
    spec = importlib.util.spec_from_file_location("migrate", path)
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)

    score_answer = {"type": "score", "score": 1.4, "probabilities": {"low": 0.1, "high": 0.9}}
    assert migrate._selected(score_answer) == "high"
    assert migrate._selected({"type": "choice", "selected": "returns"}) == "returns"
    assert migrate._selected({"type": "noul", "probability": 0.93}) == "yes"
    assert migrate._selected({"type": "noul", "probability": 0.07}) == "no"
    # And the empty case still reads as "no decision" rather than as a label.
    assert migrate._selected({"type": "score"}) is None
