"""Realistic schemas, so cost and behaviour can be quoted per use case.

A price is a property of a *schema*, not of a model: what a decision costs is
decided by how many tokens the state runs to, how big the schema is, how many
questions ride along, and how much of that is a cacheable prefix. All four are
exact from `SchemaCompiler` — which is why `scripts/price.py` can quote a cost
per thousand decisions without a GPU in the room, and why the only assumptions
left in that quote are the two it names.

These also exist because the synthetic outcome corpus describes SaaS accounts
and nothing else. A contract that only ever demonstrates one domain has not
demonstrated that it is a contract. The domains here are the ones
`docs/data.md` cleared to **green** — GoEmotions (Apache-2.0), Civil Comments
(CC0-1.0), measuring_hate_speech and HelpSteer2 (CC BY 4.0) — so a use case
defined here can be trained and published, not merely evaluated. Classic SST-5
is deliberately absent: its card carries no licence, which the audit reads as
red.

**The states here are illustrative, not data.** They are written by hand to be
the right *shape* and length for pricing; the labelled corpora those domains
come from are a phase-2 deliverable and are not vendored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, State

__all__ = ["UseCase", "all_use_cases", "use_case"]


@dataclass(frozen=True)
class UseCase:
    """One schema, one representative state, and where its labels would come from."""

    name: str
    #: What a caller is actually deciding. One line, because a use case whose
    #: purpose takes a paragraph is two use cases.
    purpose: str
    state: State
    questions: dict[str, Question]
    #: The green-tier corpus this would be trained and evaluated on, from
    #: `docs/data.md`. Naming it is the difference between a use case and a
    #: demo: it says what would make the numbers real.
    corpus: str
    #: Requests per month at a plausible production volume, used by the pricing
    #: model as a stated input rather than a hidden one.
    monthly_volume: int = 1_000_000
    notes: str = field(default="")


_SENTIMENT = UseCase(
    name="sentiment",
    purpose="Read a product review: polarity, strength, and whether it needs a reply.",
    state=(
        "Ordered the 512GB model on the 3rd and it arrived two days early, which was "
        "a nice surprise. Battery life is genuinely better than my old one — I get "
        "through a full day of meetings with ~30% left. The screen is lovely. That "
        "said, the fingerprint sensor fails maybe one time in four and I have had to "
        "type my PIN more than I would like. Support told me to update the firmware, "
        "which I did, and it is no better. Would still buy it again but the sensor is "
        "annoying enough that I am docking a star."
    ),
    questions={
        "polarity": ChoiceQuestion(
            instructions="What is the overall sentiment of this review?",
            options=[
                {"name": "positive", "criteria": "Recommends it, or praise outweighs complaint"},
                {"name": "negative", "criteria": "Advises against it, or complaint dominates"},
                {"name": "mixed", "criteria": "Substantive praise and substantive complaint"},
                {"name": "neutral", "criteria": "Descriptive, with no evaluative content"},
            ],
        ),
        "stars": ScoreQuestion(
            instructions="How many stars would this reviewer give?",
            levels=[
                {"name": "one", "value": 1.0},
                {"name": "two", "value": 2.0},
                {"name": "three", "value": 3.0},
                {"name": "four", "value": 4.0},
                {"name": "five", "value": 5.0},
            ],
        ),
        "needs_reply": NoulQuestion(
            instructions="Does this review raise an issue support should respond to?",
        ),
    },
    corpus="GoEmotions (Apache-2.0) for polarity; HelpSteer2 (CC BY 4.0) for the ordinal head",
    monthly_volume=5_000_000,
    notes=(
        "Three heads over one read of the text, which is the shape the economics "
        "turn on: a prompted baseline pays for the review three times."
    ),
)

_MODERATION = UseCase(
    name="moderation",
    purpose="Decide whether a comment can stay up, and how urgently a human is needed.",
    state=(
        "honestly if you think that policy is going to work you have not been paying "
        "attention. the people pushing it know exactly what they are doing and they "
        "do not care who it hurts. absolute clowns, every one of them."
    ),
    questions={
        "action": ChoiceQuestion(
            instructions="What should happen to this comment?",
            options=[
                {"name": "allow", "criteria": "Within policy, however rude"},
                {"name": "limit", "criteria": "Stays up, not amplified"},
                {"name": "remove", "criteria": "Breaks policy"},
                {"name": "escalate", "criteria": "Possible legal or safety exposure"},
            ],
        ),
        "severity": ScoreQuestion(
            instructions="How severe is the policy breach, if any?",
            levels=[
                {"name": "none", "value": 0.0},
                {"name": "minor", "value": 1.0},
                {"name": "serious", "value": 2.0},
                {"name": "severe", "value": 3.0},
            ],
        ),
        "targets_a_person": NoulQuestion(
            instructions="Is this directed at a specific identifiable person?",
        ),
    },
    corpus="Civil Comments (CC0-1.0); measuring_hate_speech (CC BY 4.0) for the severity head",
    monthly_volume=50_000_000,
    notes=(
        "The volume is what makes this the interesting cell: at fifty million "
        "decisions a month, a tenth of a cent each is fifty thousand dollars."
    ),
)

_SUPPORT = UseCase(
    name="support_triage",
    purpose="Route an inbound ticket and decide whether it can wait.",
    state={
        "channel": "email",
        "plan": "enterprise",
        "seats": 480,
        "open_tickets": 7,
        "payment_failed": True,
        "subject": "charged twice for October",
        "body": (
            "We were billed on the 1st and again on the 4th for the same "
            "subscription. Finance has flagged it and I need the duplicate "
            "reversed before month end close."
        ),
    },
    questions={
        "team": ChoiceQuestion(
            instructions="Which team should own this ticket?",
            options=[
                {"name": "billing", "criteria": "Charges, invoices, refunds"},
                {"name": "technical", "criteria": "Errors, outages, integrations"},
                {"name": "account", "criteria": "Seats, permissions, renewals"},
                {"name": "other", "criteria": "Anything else"},
            ],
        ),
        "urgency": ScoreQuestion(
            instructions="How urgent is this?",
            levels=[
                {"name": "whenever", "value": 0.0},
                {"name": "this_week", "value": 1.0},
                {"name": "today", "value": 2.0},
                {"name": "now", "value": 3.0},
            ],
        ),
        "wants_refund": NoulQuestion(instructions="Is the customer asking for money back?"),
    },
    corpus="Banking77 (CC BY 4.0) and CLINC150 (CC BY 3.0) for the routing head",
    monthly_volume=2_000_000,
    notes="Structured state plus free text, which is the mixed case most callers have.",
)

_USE_CASES = {u.name: u for u in (_SENTIMENT, _MODERATION, _SUPPORT)}


def use_case(name: str) -> UseCase:
    """One use case by name, or a listing of what there is."""
    try:
        return _USE_CASES[name]
    except KeyError:
        raise KeyError(f"unknown use case {name!r}; try one of {sorted(_USE_CASES)}") from None


def all_use_cases() -> list[UseCase]:
    """Every use case, in a stable order."""
    return [_USE_CASES[name] for name in sorted(_USE_CASES)]
