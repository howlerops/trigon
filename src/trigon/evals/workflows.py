"""Concrete workflows with resolvable outcomes.

The build plan says to reuse the competitor's four published workflows and add
our own. Theirs are not ours to redistribute, and more importantly they are
scored against a frontier ensemble's answers -- vendor-graded, so a shared
error is invisible. These are the "ground-truth variant" the plan asks for
instead: every decision here has an outcome the generator computed, so a wrong
answer is wrong against reality rather than against another model's opinion.

Each workflow has conditional steps, because that is the shape real pipelines
have and it is what makes the cost axis meaningful: a graph that asks the
refund question only of billing tickets does less work than one that asks
everything of everyone, and the harness reports the difference as
``mean_model_calls``.

The states are generated, and the signal is placed in text a reader could
follow. Distractors are deliberate: a workflow whose first step is a keyword
lookup measures nothing.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from .workflow import Workflow, WorkflowCase, WorkflowStep

__all__ = ["all_workflows", "moderation_queue", "support_triage"]

_TEAMS = [
    ("billing", "a charge, invoice, refund or payment that failed"),
    ("shipping", "delivery, tracking, or a package that did not arrive"),
    ("account", "sign-in, password, or profile settings"),
    ("hardware", "a physical device that is broken or faulty"),
]

# Deliberately paraphrased away from the option criteria. An earlier draft
# used the criteria's own words ("a charge... refund... payment that failed")
# and the lexical floor scored 0.832 on routing, which measures string overlap
# rather than comprehension. The repo's own rule: run the floor against a
# benchmark before believing it.
_COMPLAINT = {
    "billing": (
        "I paid for one order but two amounts left my account and the second payment was rejected"
    ),
    "shipping": ("the parcel is still not here and delivery has not moved in a week"),
    "account": "I cannot sign in and the reset link never reaches my inbox",
    "hardware": "the device turned up with a cracked screen and will not switch on",
}

# Sentences that name another team's vocabulary without being about it. Without
# these the routing step is a keyword lookup.
# One sentence per team, each using that team's criteria vocabulary verbatim
# while ruling the team out. A ticket draws one at random from the teams it is
# NOT about, so a word-matching router is pulled toward a different wrong
# answer each time rather than the same one.
#
# Getting this wrong in both directions was instructive. With no distractors
# the lexical floor scored 0.832 on routing -- string overlap, not
# comprehension. With a fixed distractor per team it scored 0.000, because the
# pull was deterministic: that measures the distractor, not the router.
_DISTRACTOR = {
    "billing": "No refund is in dispute here.",
    "shipping": "Tracking says the package arrived.",
    "account": "My password itself works fine.",
    "hardware": "Nothing is physically broken.",
}


def _ticket_state(rng: random.Random) -> tuple[dict, str, bool, bool]:
    """A ticket plus the decisions its content determines."""
    team = rng.choice([name for name, _ in _TEAMS])
    wants_refund = team == "billing" and rng.random() < 0.6
    severe = rng.random() < 0.35
    message = _COMPLAINT[team]
    if wants_refund:
        # Not "money back", which is the Noul's own wording.
        message += ", and I expect that amount reimbursed"
    if severe:
        message += ". This is the third time this month and I am considering leaving."
    other = rng.choice([name for name, _ in _TEAMS if name != team])
    state = {
        "channel": rng.choice(["email", "chat", "phone"]),
        "message": f"{message}. {_DISTRACTOR[other]}",
        "plan": rng.choice(["free", "standard", "pro", "enterprise"]),
        "open_tickets": rng.randint(0, 9),
    }
    return state, team, wants_refund, severe


def support_triage(n: int = 120, seed: int = 0) -> tuple[Workflow, list[WorkflowCase]]:
    """Route, then ask the follow-up only the routed team needs.

    Step 2 runs only for billing tickets, so a run's ``mean_model_calls`` lands
    between 1 and 2 and reflects the graph rather than the question count.
    """
    routing = ChoiceQuestion(
        instructions="Which team should handle this ticket? Judge what the customer is asking for.",
        options=[{"name": name, "criteria": criteria} for name, criteria in _TEAMS],
    )

    def route(state: object, answers: dict) -> dict[str, Question]:
        return {"team": routing}

    def refund(state: object, answers: dict) -> dict[str, Question] | None:
        if answers.get("team") != "billing":
            return None
        return {
            "refund": NoulQuestion(
                instructions="Is the customer asking for money back?",
            )
        }

    workflow = Workflow(
        name="support_triage",
        steps=(
            WorkflowStep(name="route", build=route),
            WorkflowStep(name="refund", build=refund),
        ),
    )

    rng = random.Random(f"support_triage:{seed}")
    cases = []
    for i in range(n):
        state, team, wants_refund, _ = _ticket_state(rng)
        outcome = {"team": team}
        if team == "billing":
            outcome["refund"] = "yes" if wants_refund else "no"
        cases.append(WorkflowCase(case_id=f"support/{i}", state=state, outcome=outcome))
    return workflow, cases


def moderation_queue(n: int = 120, seed: int = 0) -> tuple[Workflow, list[WorkflowCase]]:
    """Gate first, then classify only what the gate admitted.

    The shape a guardrail actually has: most items are fine and should cost one
    question, not three.
    """
    _CATEGORIES = [
        ("harassment", "targeted abuse or threats against a person"),
        ("spam", "unsolicited advertising or repeated promotional content"),
        ("self_harm", "content describing self-harm or suicide"),
    ]

    def gate(state: object, answers: dict) -> dict[str, Question]:
        return {
            "violates": NoulQuestion(
                instructions=(
                    "Does this content violate the policy on harassment, spam or self-harm?"
                )
            )
        }

    def classify(state: object, answers: dict) -> dict[str, Question] | None:
        if answers.get("violates") != "yes":
            return None
        return {
            "category": ChoiceQuestion(
                instructions="Which policy does this content violate?",
                options=[{"name": n, "criteria": c} for n, c in _CATEGORIES],
            ),
            "severity": ScoreQuestion(
                instructions="How severe is the violation?",
                levels=[
                    {"name": "minor", "value": 1.0},
                    {"name": "moderate", "value": 2.0},
                    {"name": "severe", "value": 3.0},
                ],
            ),
        }

    bodies = {
        "harassment": "You are worthless and everyone here would be better off if you left.",
        "spam": "BUY NOW! Limited offer, click this link, best prices, act fast, click now!",
        "self_harm": "I have been thinking about hurting myself and I do not want to be here.",
    }
    benign = [
        "Does anyone know how to export this to CSV? The docs are not clear.",
        "Thanks for the help yesterday, the fix worked perfectly.",
        "I disagree with the roadmap but I understand the reasoning behind it.",
    ]

    workflow = Workflow(
        name="moderation_queue",
        steps=(
            WorkflowStep(name="gate", build=gate),
            WorkflowStep(name="classify", build=classify),
        ),
    )

    rng = random.Random(f"moderation:{seed}")
    cases = []
    for i in range(n):
        violating = rng.random() < 0.4
        if violating:
            category = rng.choice([c for c, _ in _CATEGORIES])
            state = {"body": bodies[category], "author_age_days": rng.randint(1, 900)}
            outcome = {"violates": "yes", "category": category}
        else:
            state = {"body": rng.choice(benign), "author_age_days": rng.randint(1, 900)}
            outcome = {"violates": "no"}
        cases.append(WorkflowCase(case_id=f"moderation/{i}", state=state, outcome=outcome))
    return workflow, cases


def all_workflows(n: int = 120, seed: int = 0) -> Sequence[tuple[Workflow, list[WorkflowCase]]]:
    """Every shipped workflow, with its cases."""
    return [support_triage(n, seed), moderation_queue(n, seed)]
