"""The shipped workflows, and the property that makes them worth shipping."""

from __future__ import annotations

import pytest

from trigon.backends.lexical import LexicalBackend
from trigon.engine import Engine
from trigon.evals import all_workflows, moderation_queue, run_workflow, support_triage


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(LexicalBackend())


def test_every_workflow_runs_and_is_scored_against_outcomes(engine):
    for workflow, cases in all_workflows(n=40):
        result = run_workflow(engine, workflow, cases)
        assert result.n_cases == len(cases)
        # Scored against what the generator recorded, not against another
        # model's opinion -- so a shared error is not invisible.
        assert result.outcome_accuracy is not None
        assert 0.0 <= result.outcome_accuracy <= 1.0


def test_conditional_steps_actually_skip(engine):
    """A graph that asks the refund question only of billing tickets does less
    work than one that asks everything of everyone, and the cost axis has to
    show it."""
    workflow, cases = support_triage(n=60)
    result = run_workflow(engine, workflow, cases)
    assert 1.0 < result.mean_model_calls < 2.0


def test_moderation_gates_before_classifying(engine):
    workflow, cases = moderation_queue(n=60)
    result = run_workflow(engine, workflow, cases)
    assert 1.0 <= result.mean_model_calls < 2.0
    # Most items are benign and should cost one question, not three.
    assert result.mean_model_calls < 1.6


def test_the_lexical_floor_does_not_ace_the_routing_step():
    """The repo's own rule. An earlier draft quoted the option criteria in the
    ticket text and the floor scored 0.832; a later one made the surface signal
    anti-correlated and it scored 0.000. Both measure the wording, not the
    routing."""
    workflow, cases = support_triage(n=200, seed=3)
    result = run_workflow(Engine(LexicalBackend()), workflow, cases)
    assert 0.15 < result.outcome_accuracy < 0.6


def test_outcomes_only_cover_questions_the_graph_asked():
    """A skipped step has no outcome to score, so the accuracy denominator
    must not count it."""
    _, cases = support_triage(n=80)
    for case in cases:
        assert "team" in case.outcome
        if case.outcome["team"] != "billing":
            assert "refund" not in case.outcome


def test_workflows_are_deterministic_for_a_seed():
    a = support_triage(n=20, seed=5)[1]
    b = support_triage(n=20, seed=5)[1]
    assert [c.state for c in a] == [c.state for c in b]
    assert [c.outcome for c in a] == [c.outcome for c in b]
