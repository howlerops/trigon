from __future__ import annotations

import pytest

from trigon.backends.lexical import LexicalBackend
from trigon.engine import Engine
from trigon.types import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest


@pytest.fixture
def engine() -> Engine:
    return Engine(LexicalBackend())


@pytest.fixture
def support_request() -> SystemOneRequest:
    return SystemOneRequest(
        state="The customer writes: my card payment was declined at the store.",
        questions={
            "intent": ChoiceQuestion(
                instructions="Route this ticket.",
                options=[
                    {"name": "card_declined", "criteria": "a card transaction was refused"},
                    {"name": "lost_luggage", "criteria": "baggage missing after a flight"},
                    {"name": "password_reset", "criteria": "cannot sign in"},
                ],
            ),
            "severity": ScoreQuestion(
                instructions="How severe is this for the customer?",
                levels=[
                    {"name": "low", "value": 1.0},
                    {"name": "medium", "value": 3.0},
                    {"name": "high", "value": 5.0},
                ],
            ),
            "urgent": NoulQuestion(instructions="Does this need a human within the hour?"),
        },
    )
