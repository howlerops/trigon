"""Request budgets, and the thresholds that engage the large-cardinality stage.

These numbers answer open question 1 of the build plan ("per-request
option/token budget before the retrieval stage kicks in"). The derivation and
the experiment that validates them are in ``docs/decisions.md``; this module is
the single place the numbers live so the gateway, the schema compiler and the
eval harness cannot disagree about them.

The envelope is matched to the contract we are drop-in compatible with, which
has **two** limits rather than one: a total request budget, and a tighter
budget on state plus the *longest single question*. That second limit is the
one that actually governs a high-cardinality Choice, and building to a single
flat context number gets the retrieval trigger wrong -- an option set can fit
the total budget comfortably and still be inadmissible on its own.

Verified against docs.typesafe.ai and the OpenRouter model listing,
2026-09-20. Re-check at kickoff; the numbers are cited in docs/decisions.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# Total tokens in one request: state plus every question.
CONTEXT_BUDGET_TOKENS = 65_536

# State plus the longest single question. The binding constraint for large
# option sets, and the reason the retrieval trigger is per-question.
SINGLE_QUESTION_ENVELOPE_TOKENS = 32_768

# How the total is carved up. State gets a generous slice because it dominates
# tokens in document-heavy workloads; schema gets the largest because it is the
# cacheable half and the one large option sets grow into.
STATE_BUDGET_TOKENS = 16_384
SCHEMA_BUDGET_TOKENS = 40_960
READOUT_BUDGET_TOKENS = 4_096
# The remainder is headroom for template scaffolding and for tokenizer
# estimation error, which is one-sided on purpose.

# The most any single question may compile to, derived rather than chosen:
# whatever the per-question envelope leaves once state has taken its budget.
MAX_QUESTION_TOKENS = SINGLE_QUESTION_ENVELOPE_TOKENS - STATE_BUDGET_TOKENS

MAX_QUESTIONS_PER_REQUEST = 64
MAX_LEVELS_PER_SCORE = 32

# The large-cardinality stage engages when EITHER trigger fires. The count
# trigger protects the readout head; the token trigger protects the
# per-question envelope. Which fires first depends on how verbose the options
# are: bare names trip the count trigger, options carrying criteria trip the
# token trigger at roughly 800 options.
RETRIEVAL_OPTION_COUNT_TRIGGER = 1_024
RETRIEVAL_QUESTION_TOKEN_TRIGGER = MAX_QUESTION_TOKENS

# Options surviving the ANN prefilter and scored by the model. Chosen to sit
# just above the 255-option cap of the contract we mirror, so the
# post-retrieval path is never narrower than the competitor's native path.
RETRIEVAL_SHORTLIST_SIZE = 256

# Release gate on the prefilter: if recall@shortlist of the true option drops
# below this on the cardinality eval sets, raise the shortlist rather than
# shipping a recall ceiling the model cannot recover from.
RETRIEVAL_MIN_RECALL_AT_K = 0.99

# A hard ceiling that no schema may exceed even with retrieval enabled, so a
# pathological request fails fast at the gateway instead of OOM-ing a replica.
MAX_OPTIONS_PER_QUESTION = 100_000


@dataclass(frozen=True)
class Budget:
    """Resolved budgets for one deployment tier."""

    context_tokens: int = CONTEXT_BUDGET_TOKENS
    single_question_envelope: int = SINGLE_QUESTION_ENVELOPE_TOKENS
    state_tokens: int = STATE_BUDGET_TOKENS
    schema_tokens: int = SCHEMA_BUDGET_TOKENS
    readout_tokens: int = READOUT_BUDGET_TOKENS
    max_questions: int = MAX_QUESTIONS_PER_REQUEST
    retrieval_option_trigger: int = RETRIEVAL_OPTION_COUNT_TRIGGER
    shortlist_size: int = RETRIEVAL_SHORTLIST_SIZE

    def __post_init__(self) -> None:
        reserved = self.state_tokens + self.schema_tokens + self.readout_tokens
        if reserved > self.context_tokens:
            raise ValueError(
                f"budget over-subscribed: {reserved} reserved tokens "
                f"exceed the {self.context_tokens}-token context"
            )
        if self.state_tokens >= self.single_question_envelope:
            raise ValueError(
                f"state budget {self.state_tokens} leaves no room for a question "
                f"inside the {self.single_question_envelope}-token per-question envelope"
            )

    @property
    def max_question_tokens(self) -> int:
        """The largest a single compiled question may be.

        Derived from the per-question envelope rather than configured, so it
        cannot drift away from the contract it mirrors.
        """
        return self.single_question_envelope - self.state_tokens

    def needs_retrieval(self, option_count: int, question_tokens: int) -> bool:
        """Whether a question of this size must go through the ANN prefilter."""
        return (
            option_count > self.retrieval_option_trigger
            or question_tokens > self.max_question_tokens
        )

    def fits_envelope(self, state_tokens: int, question_tokens: int) -> bool:
        """Whether state plus this one question fits the per-question envelope."""
        return state_tokens + question_tokens <= self.single_question_envelope


DEFAULT_BUDGET = Budget()


@dataclass(frozen=True)
class LatencyTarget:
    """Serving SLO, published rather than claimed -- see docs/evals.md."""

    p50_ms: float = 150.0
    p99_ms: float = 500.0
    state_tokens: int = 2_048
    tier: str = "workhorse"


DEFAULT_LATENCY_TARGET = LatencyTarget()

# Release gates from the build plan's eval section.
CALIBRATION_GATES: dict[str, float] = {
    "workhorse_max_ece": 0.05,
    "premium_max_ece": 0.03,
    "max_quantization_ece_delta": 0.01,
}

# The smallest eval set a published ECE may be computed on.
#
# ECE estimators are biased upward at small n, and the bias is the same size as
# the gate. An independent re-analysis of published Jev benchmarks found ECE
# 0.0505-0.0712 reported at n=60 as evidence of good calibration, and pointed
# out that at that sample size the same figures are also what serious
# miscalibration looks like. Simulating it here
# (``trigon.calibration.metrics.noise_floor``) makes the scale concrete: on
# 4-way predictions a PERFECTLY calibrated model scores a mean ECE of about
# 0.12 at n=60, about 0.030 at n=1,000, and about 0.015 at n=4,000.
#
# That second figure is a finding about our own release gate, not only about
# someone else's numbers. At n=1,000 a calibrated model's 95th-percentile ECE
# is roughly 0.049 -- the whole 0.05 gate -- so the gate cannot distinguish a
# calibrated model from a miscalibrated one there. The minimum below is set
# where the floor sits at about a third of the gate, and
# ``check_gates`` additionally refuses to certify a run whose own simulated
# floor is too close to its limit, so the rule adapts to the data rather than
# trusting this constant.
MIN_CALIBRATION_SAMPLES = 5_000

# A gate is only a test if a calibrated model would clear it with room. Require
# the simulated floor's 95th percentile to sit at or below this fraction of the
# limit; otherwise the run is reported as untestable rather than as a pass.
MAX_FLOOR_FRACTION_OF_GATE = 0.5
