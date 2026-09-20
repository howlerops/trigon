"""Tier routing: answer cheaply, escalate only what was not settled.

The workhorse serves 90%+ of traffic; the premium tier exists for the
questions the workhorse was not confident about. Two things make this honest
rather than a cost trick:

* escalation is decided per *question*, not per request, and the premium pass
  only carries the unsettled questions -- which is affordable precisely
  because questions are independent by construction;
* the signal is calibrated confidence. Escalating on an uncalibrated
  confidence would escalate the wrong requests, which is another reason
  confidence is derived after temperature scaling rather than before.

Noul has no confidence field, so its escalation signal is distance from 0.5:
a Noul at 0.5 is the model saying it does not know.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..engine import Engine
from ..types import Answer, NoulAnswer, SystemOneRequest, SystemOneResponse, Timing

__all__ = ["RoutingPolicy", "TieredRouter"]


@dataclass(frozen=True)
class RoutingPolicy:
    """When to spend premium-tier money."""

    # Choice and Score escalate below this confidence.
    escalate_below_confidence: float = 0.35
    # Noul escalates when |p - 0.5| is below this.
    escalate_noul_margin: float = 0.15
    # Escalation is off unless a premium engine is configured.
    enabled: bool = True

    def should_escalate(self, answer: Answer) -> bool:
        if not self.enabled:
            return False
        if isinstance(answer, NoulAnswer):
            return abs(answer.probability - 0.5) < self.escalate_noul_margin
        return answer.confidence < self.escalate_below_confidence


class TieredRouter:
    """Workhorse first, premium for what it could not settle."""

    def __init__(
        self,
        workhorse: Engine,
        premium: Engine | None = None,
        policy: RoutingPolicy | None = None,
    ) -> None:
        self.workhorse = workhorse
        self.premium = premium
        self.policy = policy or RoutingPolicy()

    def answer(self, request: SystemOneRequest) -> SystemOneResponse:
        started = time.perf_counter()
        response = self.workhorse.answer(request)
        if self.premium is None:
            return response

        threshold = request.options.escalate_below_confidence
        policy = (
            self.policy
            if threshold is None
            else RoutingPolicy(
                escalate_below_confidence=threshold,
                escalate_noul_margin=self.policy.escalate_noul_margin,
                enabled=self.policy.enabled,
            )
        )
        unsettled = [
            qid for qid, answer in response.answers.items() if policy.should_escalate(answer)
        ]
        if not unsettled:
            return response

        escalated = self.premium.answer(
            request.model_copy(
                update={"questions": {qid: request.questions[qid] for qid in unsettled}}
            )
        )
        answers = dict(response.answers)
        answers.update(escalated.answers)

        usage = response.usage.model_copy(
            update={
                "prefill_tokens": response.usage.prefill_tokens + escalated.usage.prefill_tokens
            }
        )
        return response.model_copy(
            update={
                "answers": answers,
                "usage": usage,
                # Name both tiers: a caller comparing cost or latency across
                # requests needs to know this one took two passes.
                "model": f"{response.model}+{escalated.model}",
                "tier": "escalated",
                "timing": Timing(
                    total_ms=(time.perf_counter() - started) * 1000.0,
                    model_ms=response.timing.model_ms + escalated.timing.model_ms,
                    compile_ms=response.timing.compile_ms + escalated.timing.compile_ms,
                ),
            }
        )
