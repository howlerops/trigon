"""Request budgets, and the thresholds that engage the large-cardinality stage.

These numbers answer open question 1 of the build plan ("per-request
option/token budget before the retrieval stage kicks in"). The derivation and
the experiment that validates them are in ``docs/decisions.md``; this module is
the single place the numbers live so the gateway, the schema compiler and the
eval harness cannot disagree about them.
"""

from __future__ import annotations

from dataclasses import dataclass

# Total context the workhorse is trained and served at, matching the target in
# the build plan.
CONTEXT_BUDGET_TOKENS = 32_768

# How the context is carved up. State gets the largest slice because in
# document-heavy workloads state dominates tokens; schema gets the next largest
# because it is the cacheable half.
STATE_BUDGET_TOKENS = 12_288
SCHEMA_BUDGET_TOKENS = 12_288
READOUT_BUDGET_TOKENS = 2_048
# The remainder is headroom for the template scaffolding and for tokenizer
# estimation error, which is one-sided on purpose.

MAX_QUESTIONS_PER_REQUEST = 64
MAX_LEVELS_PER_SCORE = 32

# The large-cardinality stage engages when EITHER trigger fires. The count
# trigger protects the readout head; the token trigger protects the context,
# and it is the one that actually fires whenever options carry criteria.
RETRIEVAL_OPTION_COUNT_TRIGGER = 1_024
RETRIEVAL_SCHEMA_TOKEN_TRIGGER = SCHEMA_BUDGET_TOKENS

# Options surviving the ANN prefilter and scored by the model. Chosen to sit
# just above Jev's 255-option cap so the post-retrieval path is never narrower
# than the competitor's native path.
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
    state_tokens: int = STATE_BUDGET_TOKENS
    schema_tokens: int = SCHEMA_BUDGET_TOKENS
    readout_tokens: int = READOUT_BUDGET_TOKENS
    max_questions: int = MAX_QUESTIONS_PER_REQUEST
    retrieval_option_trigger: int = RETRIEVAL_OPTION_COUNT_TRIGGER
    retrieval_token_trigger: int = RETRIEVAL_SCHEMA_TOKEN_TRIGGER
    shortlist_size: int = RETRIEVAL_SHORTLIST_SIZE

    def __post_init__(self) -> None:
        reserved = self.state_tokens + self.schema_tokens + self.readout_tokens
        if reserved > self.context_tokens:
            raise ValueError(
                f"budget over-subscribed: {reserved} reserved tokens "
                f"exceed the {self.context_tokens}-token context"
            )

    def needs_retrieval(self, option_count: int, schema_tokens: int) -> bool:
        """Whether a question of this size must go through the ANN prefilter."""
        return (
            option_count > self.retrieval_option_trigger
            or schema_tokens > self.retrieval_token_trigger
        )


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
