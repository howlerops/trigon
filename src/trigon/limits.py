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

# ---------------------------------------------------------------------------
# Capacity
#
# The isolation rules make attention cost asymmetric, and the budgets are built
# on that asymmetry rather than on one flat context number. Measured with
# ``CompiledRequest.attention_pairs`` (see docs/architecture.md):
#
#   layout                              tokens   saving vs dense   costs like
#   32k state, 4 questions x 20 opts     11,150             18.6%      10,057
#   8k state, 20 questions x 50 opts     16,652             93.5%       4,256
#   8k state, 64 questions x 50 opts     47,804             98.1%       6,650
#
# Schema is block-diagonal -- a question's block attends only to itself -- so
# its cost is the SUM of per-question squares, not the square of the sum.
# Readouts are linear. State attends only to itself, which makes it the one
# term quadratic in the whole request.
#
# So: questions and option sets are cheap to grow and state is not. These
# budgets are sized accordingly, and they are deliberately far above the
# contract we mirror. Drop-in compatibility is preserved because accepting a
# superset never rejects a request the competitor would have taken; use
# ``COMPAT_BUDGET`` to reproduce their limits exactly for benchmarking.

# Total tokens in one request: state plus every question.
CONTEXT_BUDGET_TOKENS = 524_288

# State plus the longest single question. The binding constraint for large
# option sets, and the reason the retrieval trigger is per-question.
SINGLE_QUESTION_ENVELOPE_TOKENS = 131_072

# How the total is carved up.
#
# State is the expensive axis: 65,536 state tokens alone cost 4.3e9 attention
# pairs, which dominates every other term. It is raised deliberately and
# bounded deliberately.
STATE_BUDGET_TOKENS = 65_536
# Schema is nearly free by comparison, so it gets the large share.
SCHEMA_BUDGET_TOKENS = 393_216
READOUT_BUDGET_TOKENS = 32_768
# The remainder is headroom for template scaffolding and for tokenizer
# estimation error, which is one-sided on purpose.
#
# A full-size request under these budgets costs roughly what a dense model
# would spend on 82k tokens. That is the honest form of the long-context
# claim: not free, but about six times cheaper than the length suggests.

# The most any single question may compile to, derived rather than chosen:
# whatever the per-question envelope leaves once state has taken its budget.
MAX_QUESTION_TOKENS = SINGLE_QUESTION_ENVELOPE_TOKENS - STATE_BUDGET_TOKENS

# One readout slot per question under dot-product scoring, so questions are
# close to free. The competitor's own cookbook batches 13.
MAX_QUESTIONS_PER_REQUEST = 1_024
MAX_LEVELS_PER_SCORE = 64

# The large-cardinality stage engages when EITHER trigger fires. The count
# trigger protects the readout head; the token trigger protects the
# per-question envelope.
RETRIEVAL_OPTION_COUNT_TRIGGER = 1_024
RETRIEVAL_QUESTION_TOKEN_TRIGGER = MAX_QUESTION_TOKENS

# Options surviving the ANN prefilter and scored by the model.
#
# Raised from 256 to 2,048 because the per-question budget now allows it, and
# because 256 demonstrably was not enough: at 10,000 options with a query
# stating two of four fields, recall@256 was 0.9400 against a 0.99 gate, while
# recall@1536 was 1.0000. At the old 16,384-token per-question budget a
# 1,536-option shortlist only fitted with the option criteria stripped out; at
# 65,536 it fits with the criteria intact (43,775 tokens, 33% headroom).
# See docs/decisions.md section 1.
RETRIEVAL_SHORTLIST_SIZE = 2_048

# Release gate on the prefilter: if recall@shortlist of the true option drops
# below this on the cardinality eval sets, raise the shortlist rather than
# shipping a recall ceiling the model cannot recover from.
RETRIEVAL_MIN_RECALL_AT_K = 0.99

# A hard ceiling that no schema may exceed even with retrieval enabled, so a
# pathological request fails fast at the gateway instead of OOM-ing a replica.
MAX_OPTIONS_PER_QUESTION = 100_000

# Above this many options in one answer, returning every probability makes the
# response itself the bottleneck: 100,000 options is roughly 2 MB of JSON.
# Callers can ask for the top-k instead; see RequestOptions.top_probabilities.
FULL_DISTRIBUTION_ADVISORY_LIMIT = 4_096


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

#: The contract we are drop-in compatible with, reproduced exactly. Use it to
#: benchmark like for like, and to check that a request built for their limits
#: is still valid here -- which it always is, because ours are a superset.
COMPAT_BUDGET = Budget(
    context_tokens=65_536,
    single_question_envelope=32_768,
    state_tokens=16_384,
    schema_tokens=40_960,
    readout_tokens=4_096,
    max_questions=64,
    shortlist_size=256,
)


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


# How far a model must beat the marginal predictor -- the one that ignores the
# state and reports each question's label frequencies -- before a run may
# certify it.
#
# This gate exists because the first trained reference model passed every
# calibration gate with 46% accuracy: ECE 0.0111 against a 0.05 limit, bins
# aligned, floor cleared. That is not a fluke and it is not a bug in the
# metrics. A model that reports the true marginal distribution is calibrated
# BY CONSTRUCTION -- it just does not use the input. Gating on ECE alone
# therefore certifies the one model guaranteed to be useless, which makes a
# calibration-only release gate worse than no gate, because it looks like
# evidence.
#
# The margin is a floor against that degenerate case, not an accuracy target.
# The real accuracy bar is the workflow suite.
MIN_ACCURACY_OVER_BASELINE = 0.05
