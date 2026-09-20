"""Verifiable synthetic data: generated state with code-checkable answers.

This is the cheapest of the five data streams in the build plan and the one
that needs no license audit, no annotators and no teacher calls. State is a
generated JSON record; questions are predicates over that record that the
generator can evaluate exactly. It is therefore both training data with real
ground-truth outcomes and an eval set the calibration suite can run on the day
the repo is cloned.

Its limits are worth stating plainly, because a calibration number measured
only here would be dishonest: the state is structured and clean, the
predicates are decidable, and nothing is subjective. It exercises the
calibration machinery end to end. It does not stand in for the public labelled
corpora or the annotator-distribution corpora, which is why
``docs/data.md`` treats it as one stream of five rather than a shortcut.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from ..types import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from .harness import Case, Expectation

__all__ = ["synthetic_outcome_cases"]

_PLANS = ["free", "standard", "pro", "enterprise"]
_REGIONS = ["emea", "amer", "apac"]
_TIERS = [("bronze", 0), ("silver", 1), ("gold", 2), ("platinum", 3)]


def _record(rng: random.Random) -> dict:
    return {
        "plan": rng.choice(_PLANS),
        "region": rng.choice(_REGIONS),
        "seats": rng.randint(1, 500),
        "open_tickets": rng.randint(0, 12),
        "days_since_signup": rng.randint(1, 2000),
        "payment_failed": rng.random() < 0.3,
    }


def synthetic_outcome_cases(n: int = 200, seed: int = 0, *, noise: float = 0.0) -> list[Case]:
    """Generate ``n`` cases whose answers are computed, not annotated.

    ``noise`` flips the ground truth with that probability. A zero-noise set is
    perfectly predictable, which makes it useless for calibration: a model can
    be right every time and any confidence below 1.0 reads as miscalibrated.
    Non-zero noise creates genuine aleatoric uncertainty, so the correct
    behaviour is a probability near ``1 - noise`` and the suite can tell a
    calibrated model from a confident one.
    """
    rng = random.Random(f"synthetic:{seed}")
    return list(_generate(rng, n, noise))


def _generate(rng: random.Random, n: int, noise: float) -> Iterator[Case]:
    for i in range(n):
        record = _record(rng)
        plan_index = _PLANS.index(record["plan"])
        at_risk = record["payment_failed"] and record["open_tickets"] >= 3
        tier_index = min(record["seats"] // 128, len(_TIERS) - 1)

        if noise:
            if rng.random() < noise:
                plan_index = rng.randrange(len(_PLANS))
            if rng.random() < noise:
                at_risk = not at_risk
            if rng.random() < noise:
                tier_index = rng.randrange(len(_TIERS))

        yield Case(
            case_id=f"synthetic/{i}",
            request=SystemOneRequest(
                state=record,
                questions={
                    "plan": ChoiceQuestion(
                        instructions="Which plan is this account on?",
                        options=[{"name": p} for p in _PLANS],
                    ),
                    "at_risk": NoulQuestion(
                        instructions=(
                            "Is this account at risk? An account is at risk when a "
                            "payment has failed and it has three or more open tickets."
                        )
                    ),
                    "size": ScoreQuestion(
                        instructions="How large is this account by seat count?",
                        levels=[{"name": name, "value": float(value)} for name, value in _TIERS],
                    ),
                },
            ),
            expected={
                "plan": Expectation(label=plan_index),
                "at_risk": Expectation(probability=1.0 if at_risk else 0.0),
                "size": Expectation(label=tier_index),
            },
            domain="accounts",
            tags=("synthetic", "verifiable"),
        )
