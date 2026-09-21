"""Translate the incumbent's wire shape to ours, and our answers back.

The point of this module is that "drop-in" stops being a claim in a README and
becomes a thing a caller can test: change a base URL, keep the request bodies,
read the same response fields. Everything here is a pure mapping over JSON --
no model, no calibration, no policy -- so the one thing that can go wrong is a
field we translate wrongly, and that is what `tests/test_compat.py` is for.

**The shapes are not ours and were not guessed.** They are the published
contract (docs.typesafe.ai/api.md and the three primitive pages, read
2026-09-21), and where they differ from our own the difference is recorded in
`docs/compat.md` rather than papered over. Three differences are worth naming
here because they are the ones a caller can observe:

1. **Their questions carry `criteria`; ours carry `options` and `levels`.** A
   Choice's criteria is a *map* from option name to description and a Score's
   is an *ordered array* of level descriptions. Map order is the option order,
   array index is the level number, and both are load-bearing -- their `score`
   is a probability-weighted mean of level *indices*, so reordering is not a
   cosmetic change.

2. **A Noul may carry `criteria: {true, false}` and our Noul has nowhere to put
   it.** Dropping it would silently discard something the caller wrote to steer
   the answer, so it is folded into the instructions (`_fold_noul_criteria`).
   That is lossy in the sense that the text is no longer separable, and it is
   not lossy in the sense that the model sees every word of it.

3. **Any free-text slot may arrive as an object or an array.** Their contract
   allows structured instructions and structured criteria (rubrics with
   `what`/`not_for`/`examples`, nested taxonomies). Ours are strings. They are
   serialised, deterministically, by `_as_text` -- never summarised, never
   truncated, because a rubric that silently loses its `not_for` clause is the
   worst possible failure here: the answer still looks well-formed.

`output_tokens` is reported as 0 and that is not a stub. There is no decode
half; the readout slots are prefill. A caller's cost accounting should read
zero there, because zero is what it costs.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..schema.compiler import SchemaTooLarge
from ..types import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)
from .app import create_router
from .config import ServerConfig
from .routing import TieredRouter

__all__ = [
    "build_compat_app",
    "compat_router",
    "to_native",
    "from_native",
    "to_compat_request",
    "compat_answer_to_native",
]

# Their contract validates the body itself rather than a schema library, so a
# malformed request comes back 422 with a message. Ours does the same through
# pydantic; this is the code for the cases we check by hand.
UNPROCESSABLE = 422


class CompatError(ValueError):
    """A request their contract would reject. Becomes a 422, as theirs does."""


def _as_text(value: Any, *, field: str) -> str:
    """Flatten a slot their contract allows to be structured.

    Strings pass through untouched. Anything else is serialised as compact
    JSON with its keys in the order written, which keeps the schema prefix
    hash stable for an unchanged request and keeps every word the caller
    wrote in front of the model.
    """
    if isinstance(value, str):
        return value
    if value is None:
        raise CompatError(f"{field} is required")
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:  # pragma: no cover - json is total here
        raise CompatError(f"{field} is not serialisable: {exc}") from exc


def _fold_noul_criteria(instructions: str, criteria: Any) -> str:
    """Put a Noul's true/false criteria where the model will actually read it.

    Our NoulQuestion has instructions and nothing else, by design: a binary
    question's schema is the question. Their contract lets a caller attach
    boundary descriptions, and a caller who wrote them expects them to matter.
    """
    if criteria is None:
        return instructions
    if not isinstance(criteria, dict):
        raise CompatError("a noul's criteria must be an object with 'true' and/or 'false'")
    parts = [instructions]
    for key in ("true", "false"):
        if key in criteria:
            parts.append(f"Counts as {key}: {_as_text(criteria[key], field=f'criteria.{key}')}")
    return "\n".join(parts)


def _choice(instructions: str, criteria: Any) -> ChoiceQuestion:
    if not isinstance(criteria, dict) or not criteria:
        raise CompatError("a choice's criteria must be a non-empty object of option -> description")
    options = [
        {"name": name, "criteria": _as_text(text, field=f"criteria.{name}") if text else None}
        for name, text in criteria.items()
    ]
    return ChoiceQuestion(instructions=instructions, options=options)


def _score(instructions: str, criteria: Any) -> ScoreQuestion:
    if not isinstance(criteria, list) or not criteria:
        raise CompatError("a score's criteria must be a non-empty ordered array of levels")
    # Their level *names* are the array indices -- that is what comes back in
    # `legend` and in `probabilities`, and what `score` is a mean of. Naming
    # them anything else here would produce a response whose keys are not the
    # ones their SDK reads.
    levels = [
        {
            "name": str(index),
            "criteria": _as_text(text, field=f"criteria[{index}]") if text else None,
            "value": float(index),
        }
        for index, text in enumerate(criteria)
    ]
    return ScoreQuestion(instructions=instructions, levels=levels)


def to_native(payload: dict[str, Any]) -> SystemOneRequest:
    """Their request body -> ours. Raises CompatError on what they'd reject."""
    if not isinstance(payload, dict):
        raise CompatError("the request body must be an object")
    for required in ("state", "model", "questions"):
        if required not in payload:
            raise CompatError(f"missing required field {required!r}")
    questions_in = payload["questions"]
    if not isinstance(questions_in, dict) or not questions_in:
        raise CompatError("questions must be a non-empty object")

    questions: dict[str, Any] = {}
    for qid, spec in questions_in.items():
        if not isinstance(spec, dict):
            raise CompatError(f"question {qid!r} must be an object")
        kind = spec.get("type")
        instructions = _as_text(spec.get("instructions"), field=f"questions.{qid}.instructions")
        criteria = spec.get("criteria")
        if kind == "noul":
            questions[qid] = NoulQuestion(instructions=_fold_noul_criteria(instructions, criteria))
        elif kind == "choice":
            questions[qid] = _choice(instructions, criteria)
        elif kind == "score":
            questions[qid] = _score(instructions, criteria)
        else:
            raise CompatError(
                f"question {qid!r} has unknown type {kind!r}; expected noul, choice or score"
            )
    # `model` is accepted and not honoured: which build answered is reported in
    # the response, where it is a fact rather than a request. A deployment
    # serves the weights it was started with.
    return SystemOneRequest(state=payload["state"], questions=questions)


def _legend(question: Any) -> dict[str, str]:
    """Their `legend`: level name -> the description the caller declared.

    We do not carry one on the answer, because our ScoreAnswer is keyed by the
    level names the caller chose and a legend would restate the request. Theirs
    echoes it, so the adapter rebuilds it from the request rather than
    inventing text: a level declared without a description gets an empty
    string, which is what the caller sent.
    """
    if not isinstance(question, ScoreQuestion):
        return {}
    return {level.name: level.criteria or "" for level in question.levels}


def _answer_out(answer: Any, question: Any) -> dict[str, Any]:
    if isinstance(answer, NoulAnswer):
        # No confidence field: theirs does not carry one either, for the same
        # reason ours does not -- the probability already says it.
        return {"type": "noul", "noul": answer.probability}
    if isinstance(answer, ChoiceAnswer):
        return {
            "type": "choice",
            "choice": answer.selected,
            "confidence": answer.confidence,
            "probabilities": dict(answer.probabilities),
        }
    if isinstance(answer, ScoreAnswer):
        return {
            "type": "score",
            "score": answer.score,
            "confidence": answer.confidence,
            "legend": _legend(question),
            "probabilities": dict(answer.probabilities),
        }
    raise ValueError(f"no compat mapping for {type(answer).__name__}")


def from_native(response: SystemOneResponse, request: SystemOneRequest) -> dict[str, Any]:
    """Our response -> theirs.

    Takes the request as well, because their Score answer carries a `legend`
    and ours does not: the legend is an echo of the declared levels, so it is
    rebuilt from the request instead of being stored twice.

    `model` reports the build that actually answered rather than echoing what
    was asked for. A response that named a model it did not run would make the
    one identifier reaching the caller useless for exactly the case it exists
    for.
    """
    return {
        "model": response.model,
        "answers": {
            qid: _answer_out(answer, request.questions.get(qid))
            for qid, answer in response.answers.items()
        },
        "usage": {
            "input_tokens": response.usage.prefill_tokens,
            # There is no decode half. Zero is the measurement, not a stub.
            "output_tokens": 0,
        },
    }


# The route is validated by hand rather than by a pydantic model, and this is
# the schema that documents it. Their contract is deliberately looser than
# anything we would write for ourselves -- `state`, `instructions` and every
# `criteria` value may be a string, an object, an array or null -- and a
# pydantic model tight enough to be worth reading would reject requests their
# service accepts, which is the one failure a compatibility layer may not have.
# So: hand validation for behaviour, this literal for the published spec, and
# `tests/test_compat.py` holding the two together with their own example
# bodies.
_ANY = {"description": "string, object or array"}

COMPAT_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["state", "model", "questions"],
    "properties": {
        "state": {**_ANY, "description": "What the questions are about."},
        "model": {"type": "string", "description": "Accepted; the response names what ran."},
        "questions": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "object",
                "required": ["type", "instructions"],
                "properties": {
                    "type": {"type": "string", "enum": ["noul", "choice", "score"]},
                    "instructions": _ANY,
                    "criteria": {
                        "description": (
                            "choice: a map of option name to description, in answer order. "
                            "score: an ordered array of level descriptions, index-numbered. "
                            "noul: an optional object with 'true' and/or 'false'."
                        )
                    },
                },
            },
        },
    },
}

COMPAT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["model", "answers", "usage"],
    "properties": {
        "model": {"type": "string", "description": "The build that answered."},
        "answers": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["noul", "choice", "score"]},
                    "noul": {"type": "number", "description": "noul only. No confidence field."},
                    "choice": {"type": "string", "description": "choice only."},
                    "score": {
                        "type": "number",
                        "description": "score only: the probability-weighted mean of the levels.",
                    },
                    "legend": {"type": "object", "description": "score only: level -> criteria."},
                    "confidence": {"type": "number", "description": "choice and score only."},
                    "probabilities": {"type": "object", "description": "choice and score only."},
                },
            },
        },
        "usage": {
            "type": "object",
            "properties": {
                "input_tokens": {"type": "integer"},
                "output_tokens": {
                    "type": "integer",
                    "description": "Always 0: prefill-only, there is no decode half.",
                },
            },
        },
    },
}


def compat_router(router: TieredRouter) -> APIRouter:
    """Their endpoint, served by our stack."""
    api = APIRouter()

    @api.post(
        "/v1/systemone",
        tags=["compat"],
        summary="The incumbent's request and response shapes, answered by this model",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": COMPAT_REQUEST_SCHEMA}},
            },
            "responses": {
                "200": {
                    "description": "One typed answer per question.",
                    "content": {"application/json": {"schema": COMPAT_RESPONSE_SCHEMA}},
                },
                "422": {"description": "A request their contract would reject."},
            },
        },
    )
    async def systemone(request: Request) -> Any:
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=UNPROCESSABLE, detail=f"invalid JSON: {exc}") from exc
        try:
            native = to_native(payload)
        except CompatError as exc:
            raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc
        except ValueError as exc:
            # A pydantic rejection of a body we translated: still the caller's
            # request being invalid, and their contract has one code for that.
            raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc
        try:
            answered = router.answer(native)
        except SchemaTooLarge as exc:
            # Ours is a 413 because the request is well-formed and does not
            # fit. Theirs has no 413 -- an oversized request is a 422 there,
            # and a caller's retry logic is written against their codes.
            raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=UNPROCESSABLE, detail=str(exc)) from exc
        return JSONResponse(content=from_native(answered, native))

    return api


def build_compat_app(
    config: ServerConfig | None = None, router: TieredRouter | None = None
) -> FastAPI:
    """A whole app whose root is their base URL.

    Mounted at `/compat` on the native gateway as well, so one process serves
    both -- but a caller migrating changes a base URL and nothing else, and
    that only works if the path is theirs exactly.
    """
    config = config or ServerConfig.from_env()
    router = router or create_router(config)
    app = FastAPI(
        title="Trigon System One API (compatibility)",
        version=__version__,
        description=(
            "The incumbent's wire shapes, answered by this model. Differences "
            "that a caller can observe are documented in docs/compat.md."
        ),
    )
    app.state.router = router
    app.state.config = config
    app.include_router(compat_router(router))

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        return {"status": "ok", "version": __version__, "compat": True}

    return app


# -- the other direction: talking *to* their service -------------------------
#
# `scripts/migrate.py` compares this gateway against whatever a caller runs
# today, and the whole point of it is to be pointed at the incumbent. It sent
# our body shape to their endpoint, which their contract answers with a 422 --
# so the most persuasive artifact in the plan could not reach the system it
# was written to be compared against. These two functions are what fixed that.


def to_compat_request(request: SystemOneRequest, model: str = "jev-latest") -> dict[str, Any]:
    """Our request -> theirs, for calling their service.

    Lossy in one place and the loss is repaired downstream: our Score levels
    carry names, theirs are numbered by array position, so a Score declared
    over `bronze/silver/gold` goes out as a three-element array and comes back
    keyed `"0"/"1"/"2"`. `compat_answer_to_native` puts the names back, which
    is why these two are documented together and should be used together.
    """
    questions: dict[str, Any] = {}
    for qid, question in request.questions.items():
        if isinstance(question, NoulQuestion):
            questions[qid] = {"type": "noul", "instructions": question.instructions}
        elif isinstance(question, ChoiceQuestion):
            questions[qid] = {
                "type": "choice",
                "instructions": question.instructions,
                # Their criteria values are descriptions. An option declared
                # without one sends its own name rather than an empty string:
                # an empty description is a weaker prompt than no description,
                # and it would make the comparison unfair to their side.
                "criteria": {
                    option.name: option.criteria or option.name for option in question.options
                },
            }
        elif isinstance(question, ScoreQuestion):
            questions[qid] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": [level.criteria or level.name for level in question.levels],
            }
        else:  # pragma: no cover - the union is closed
            raise ValueError(f"no compat mapping for {type(question).__name__}")
    state = request.state
    return {
        "state": state if isinstance(state, str) else json.loads(json.dumps(state, default=str)),
        "model": model,
        "questions": questions,
    }


def compat_answer_to_native(answer: dict[str, Any], question: Any) -> dict[str, Any]:
    """One of their answers -> our field names, with Score level names restored.

    Without the restoration a Score comparison is between two different label
    vocabularies -- their `"0"/"1"/"2"` against our `bronze/silver/gold` -- and
    every case reads as a disagreement. That is the failure mode this function
    exists for: it does not crash, it just reports 0% agreement on a question
    the two systems may agree about completely.
    """
    kind = answer.get("type")
    if kind == "noul":
        return {"type": "noul", "probability": float(answer["noul"])}
    if kind == "choice":
        return {
            "type": "choice",
            "selected": answer.get("choice"),
            "probabilities": dict(answer.get("probabilities") or {}),
            "confidence": answer.get("confidence"),
        }
    if kind == "score":
        names = (
            [level.name for level in question.levels]
            if isinstance(question, ScoreQuestion)
            else None
        )
        probabilities = dict(answer.get("probabilities") or {})
        if names:
            probabilities = {
                names[int(index)]: value
                for index, value in probabilities.items()
                if str(index).isdigit() and int(index) < len(names)
            }
        selected = max(probabilities, key=lambda k: probabilities[k]) if probabilities else None
        return {
            "type": "score",
            "score": answer.get("score"),
            "selected": selected,
            "probabilities": probabilities,
            "confidence": answer.get("confidence"),
        }
    raise ValueError(f"unknown answer type {kind!r}")
