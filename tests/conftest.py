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


# float32 carries ~1.2e-07 of relative precision. A question that actually
# read another question's tokens would move an answer by a visible fraction;
# an ulp is the arithmetic.
ROUNDING = 1e-6


def assert_answer_unmoved(before: dict, after: dict, qid: str) -> None:
    """One question's answer, compared across sequences of different lengths.

    **Exact equality here is not portable, and three separate tests asserted
    it.** Adding questions lengthens the sequence; a different length can
    select a different GEMM kernel; a different reduction order rounds
    differently. All three read exactly 0.0 on the machine they were written
    on and between 1.4e-08 and 2.4e-08 on GitHub's runners.

    The mask is what guarantees independence — there is no path from one
    question's tokens to another's — and that is exact and provable from the
    layout. What is not portable is the arithmetic's reproduction of it across
    shapes. So this is the shared rule rather than three copies of a judgement
    call: probabilities within rounding, decisions identical at any magnitude.

    Comparisons at a *fixed* shape do not use this and must stay exact: the
    same kernel reduces the same way anywhere, so a difference there is a real
    one. `tests/test_prefix_cache.py`'s cache-staleness check and
    `tests/test_training.py`'s "quantizing must not mutate the original" are
    both of that kind.
    """
    assert before.keys() == after.keys(), f"{qid}: the answer's shape changed"
    for field, value in before.items():
        other = after[field]
        if isinstance(value, dict):
            assert value.keys() == other.keys(), f"{qid}.{field}: labels changed"
            for label, probability in value.items():
                moved = abs(probability - other[label])
                assert moved < ROUNDING, (
                    f"{qid}.{field}[{label}] moved by {moved:.3e}, which is not rounding"
                )
        elif isinstance(value, float):
            moved = abs(value - other)
            assert moved < ROUNDING, f"{qid}.{field} moved by {moved:.3e}, which is not rounding"
        else:
            assert value == other, f"{qid}.{field} changed from {value!r} to {other!r}"
