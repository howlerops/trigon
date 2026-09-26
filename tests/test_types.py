"""The contract's validation rules. These are the structural guarantee's first
line: a request that could produce an ambiguous answer is rejected outright."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trigon.types import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion


def test_choice_needs_at_least_two_options():
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="pick", options=[{"name": "only"}])


def test_choice_rejects_duplicate_option_names():
    with pytest.raises(ValidationError, match="duplicate option name"):
        ChoiceQuestion(instructions="pick", options=[{"name": "a"}, {"name": "a"}])


def test_score_rejects_unordered_values():
    with pytest.raises(ValidationError, match="must ascend"):
        ScoreQuestion(
            instructions="rate",
            levels=[{"name": "hi", "value": 5.0}, {"name": "lo", "value": 1.0}],
        )


def test_score_rejects_partially_valued_levels():
    with pytest.raises(ValidationError, match="all carry a value"):
        ScoreQuestion(instructions="rate", levels=[{"name": "a", "value": 1.0}, {"name": "b"}])


def test_score_anchors_fall_back_to_index():
    question = ScoreQuestion(
        instructions="rate", levels=[{"name": "a"}, {"name": "b"}, {"name": "c"}]
    )
    assert question.anchors == [0.0, 1.0, 2.0]


def test_questions_are_frozen_and_reject_unknown_fields():
    with pytest.raises(ValidationError):
        NoulQuestion(instructions="ok?", temperature=0.5)


def test_request_rejects_an_empty_question_map():
    with pytest.raises(ValidationError):
        DecisionRequest(state="s", questions={})


def test_state_accepts_text_json_and_lists():
    for state in ("text", {"a": 1}, ["one", "two"]):
        request = DecisionRequest(state=state, questions={"q": NoulQuestion(instructions="ok?")})
        assert request.state == state
