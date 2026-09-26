#!/usr/bin/env python
"""Support triage, end to end: ask, read the confidence, escalate, abstain.

    python examples/triage_cookbook.py

Runs on a fresh clone with no weights and no GPU, because the lexical floor is
a real backend. Add ``--weights reports/run.pt`` after a ``trigon train`` to
see the same code against a trained model.

The point of the example is not the routing -- it is what you do with a
*distribution* that you cannot do with a string. Four things, in order:

1. one pass answers every question, and adding questions moves none of the
   others;
2. confidence is a number you can threshold, so a queue can route the easy
   cases and hold the rest;
3. a Noul has no confidence field, because for a binary question the
   probability already is one;
4. a conformal wrapper turns "here is my best guess" into "here is a set that
   contains the answer 90% of the time", which is a different promise.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.engine import Engine  # noqa: E402
from trigon.types import DecisionRequest  # noqa: E402

TICKETS = [
    "My card was declined at the till twice and I still got charged once.",
    "Where is the parcel I ordered on Tuesday? Tracking has not moved in days.",
    "I cannot log in since yesterday, the reset email never arrives.",
    "hi",
]

ROUTES = [
    {"name": "billing", "criteria": "a charge, refund, invoice or card payment"},
    {"name": "shipping", "criteria": "a parcel, delivery, tracking or address"},
    {"name": "account", "criteria": "login, password, profile or access trouble"},
    {"name": "other", "criteria": "anything the categories above do not cover"},
]

QUESTIONS = {
    # A Choice is a distribution over exactly these four names. Nothing else
    # is representable -- there is no sampler that could emit a fifth.
    "route": {"type": "choice", "instructions": "Which queue handles this?", "options": ROUTES},
    # A Score is a distribution over ordered levels, and the reported number
    # is its expectation. It cannot land outside the scale you declared.
    "severity": {
        "type": "score",
        "instructions": "How badly is this affecting the customer?",
        "levels": [
            {"name": "minor", "value": 1.0, "criteria": "an inconvenience"},
            {"name": "moderate", "value": 2.0, "criteria": "blocked on something"},
            {"name": "severe", "value": 3.0, "criteria": "out of pocket or locked out"},
        ],
    },
    # A Noul is a yes/no judgement, independent of every other question here.
    "money_involved": {
        "type": "noul",
        "instructions": "Is the customer out of pocket right now?",
    },
}

#: Below this, the queue should not act on the model's pick by itself.
ESCALATE_BELOW = 0.35


def build_engine(weights: str | None) -> Engine:
    if weights:
        from trigon.backends.torch_readout import TorchReadoutBackend

        backend = TorchReadoutBackend.load(weights)
        # The engine takes the tokenizer from the backend, so the compiled
        # token counts match the tensors. Nothing to wire up.
        return Engine(backend)

    from trigon.backends.lexical import LexicalBackend

    return Engine(LexicalBackend())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None, help="a checkpoint from 'trigon train'")
    args = parser.parse_args()

    engine = build_engine(args.weights)
    print(f"model: {engine.backend.model_version}\n")

    escalated = 0
    for ticket in TICKETS:
        response = engine.answer(DecisionRequest(state=ticket, questions=QUESTIONS))
        route = response.answers["route"]
        severity = response.answers["severity"]
        money = response.answers["money_involved"]

        print(f"“{ticket}”")

        # 2. Confidence is a number, so this is a policy rather than a vibe.
        if route.confidence < ESCALATE_BELOW:
            escalated += 1
            runners_up = sorted(route.probabilities.items(), key=lambda kv: -kv[1])[:2]
            spread = ", ".join(f"{name} {p:.2f}" for name, p in runners_up)
            print(f"   route     → HOLD FOR A HUMAN (confidence {route.confidence:.2f}: {spread})")
        else:
            print(f"   route     → {route.selected} (confidence {route.confidence:.2f})")

        print(f"   severity  → {severity.score:.2f} of 3.00 (confidence {severity.confidence:.2f})")

        # 3. No confidence field on a Noul. Distance from 0.5 is the signal,
        #    and 0.5 exactly is the model saying it does not know.
        verdict = "yes" if money.probability > 0.5 else "no"
        print(f"   out of pocket → {verdict} (P={money.probability:.2f})")
        print(f"   cost      → {response.usage.prefill_tokens} tokens, one pass\n")

    print(f"{escalated} of {len(TICKETS)} tickets held for a human.")
    if escalated == len(TICKETS):
        # Not a disappointing result -- the intended one. The lexical floor is
        # a baseline, not a model, and a confidence-gated queue should refuse
        # to act on it. That is the whole argument for gating on a number
        # rather than on an argmax: a weak model routes nothing automatically
        # instead of routing everything wrongly and quietly.
        print(
            "   ...which is the point: this backend is the lexical floor, and a\n"
            "   queue gated on confidence declines to act on it rather than\n"
            "   acting wrongly and quietly. Pass --weights to see the contrast."
        )
    print()

    # 1. Adding a question moves none of the others -- exactly, not approximately.
    first = engine.answer(DecisionRequest(state=TICKETS[0], questions=QUESTIONS))
    with_extra = engine.answer(
        DecisionRequest(
            state=TICKETS[0],
            questions={
                **QUESTIONS,
                "is_repeat": {"type": "noul", "instructions": "Has this been reported before?"},
            },
        )
    )
    unchanged = all(
        first.answers[qid].model_dump() == with_extra.answers[qid].model_dump() for qid in QUESTIONS
    )
    print(f"adding a fifth question moved the other answers: {not unchanged}")

    # 4. A conformal wrapper, if this deployment has one fitted. It gives a set
    #    with a coverage guarantee that holds whether or not the model is
    #    calibrated -- a different and stronger promise than a probability.
    profile = pathlib.Path("reports/conformal/accounts.json")
    if profile.exists():
        from trigon.calibration.conformal import ConformalPredictor

        engine.conformal = {"accounts": ConformalPredictor.load(profile)}
        covered = engine.answer(
            DecisionRequest(
                state=TICKETS[0],
                questions=QUESTIONS,
                options={"conformal_profile": "accounts"},
            )
        )
        answer = covered.answers["route"]
        print(
            f"conformal set at {answer.coverage_target:.0%} coverage: "
            f"{answer.prediction_set} (of {len(ROUTES)} routes)"
        )
    else:
        print("conformal: no fitted profile — run `trigon fit --conformal-out ...`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
