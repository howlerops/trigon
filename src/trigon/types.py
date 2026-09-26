"""The wire contract: three primitives in, three answer shapes out.

Type safety here is structural, not validated-after-the-fact. A Choice answer
can only ever be a softmax over the option set the request declared, because
that is the only vector the readout head produces -- there is no sampler that
could emit an option nobody asked for. These models describe that guarantee at
the API boundary; ``trigon.schema.compiler`` enforces it at the tensor
boundary.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .limits import (
    MAX_LEVELS_PER_SCORE,
    MAX_OPTIONS_PER_QUESTION,
    MAX_QUESTIONS_PER_REQUEST,
)

__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "EvidenceMethod",
    "EvidenceSpan",
    "LevelSpec",
    "NoulAnswer",
    "NoulQuestion",
    "OptionSpec",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "State",
    "DecisionRequest",
    "DecisionResponse",
    "Usage",
]

# State is whatever the caller already has: a blob of text, a JSON record, or a
# list of documents. It is never interpreted as instructions -- see the
# injection-robustness section of docs/training.md.
State = str | dict[str, Any] | list[Any]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OptionSpec(_Strict):
    """One option of a Choice, with the criteria that select it."""

    name: str = Field(
        min_length=1, max_length=256, description="The label this option is returned under."
    )
    criteria: str | None = Field(
        default=None,
        max_length=8192,
        description="What selects this option. Compiled into the option's schema block.",
    )


class LevelSpec(_Strict):
    """One level of a Score. Levels are ordered low to high as given."""

    name: str = Field(
        min_length=1, max_length=256, description="The label this level is returned under."
    )
    criteria: str | None = Field(
        default=None, max_length=8192, description="What places the state at this level."
    )
    value: float | None = Field(
        default=None,
        description=(
            "Anchors the level on your own scale (1-5 stars, 0-4 Likert), so the reported "
            "score is on that scale. Omit on every level and they anchor at their index "
            "instead. Values must ascend; all levels carry one or none do."
        ),
    )


class _QuestionBase(_Strict):
    instructions: str = Field(
        min_length=1,
        max_length=32768,
        description="What this question is asking. Part of the cacheable schema prefix.",
    )


class ChoiceQuestion(_QuestionBase):
    """Pick one option from a defined set. Relative: it settles *which*."""

    type: Literal["choice"] = "choice"
    options: list[OptionSpec] = Field(
        min_length=2,
        max_length=MAX_OPTIONS_PER_QUESTION,
        description=(
            "The options to choose between, in the order probabilities are returned. "
            "Names must be unique. Two is the minimum: a single-option Choice has no "
            "answer to give. Above the large-cardinality trigger the set is narrowed by "
            "a prefilter and the dropped options come back at probability 0."
        ),
    )

    @field_validator("options")
    @classmethod
    def _unique_option_names(cls, options: list[OptionSpec]) -> list[OptionSpec]:
        _reject_duplicates([o.name for o in options], "option")
        return options

    @property
    def names(self) -> list[str]:
        return [o.name for o in self.options]


class ScoreQuestion(_QuestionBase):
    """Place the state on an ordered set of descriptive levels.

    Not a regression head: the model produces a distribution over the declared
    levels and the reported score is its expectation.
    """

    type: Literal["score"] = "score"
    levels: list[LevelSpec] = Field(
        min_length=2,
        max_length=MAX_LEVELS_PER_SCORE,
        description=(
            "The levels, ordered low to high as given. Names must be unique. The answer "
            "is a distribution over these, and the reported score is its expectation."
        ),
    )

    @field_validator("levels")
    @classmethod
    def _unique_level_names(cls, levels: list[LevelSpec]) -> list[LevelSpec]:
        _reject_duplicates([lv.name for lv in levels], "level")
        return levels

    @model_validator(mode="after")
    def _values_are_ordered(self) -> ScoreQuestion:
        values = [lv.value for lv in self.levels]
        given = [v for v in values if v is not None]
        if given and len(given) != len(values):
            raise ValueError("score levels must all carry a value or none of them may")
        if given != sorted(given):
            raise ValueError(f"score level values must ascend, got {given}")
        return self

    @property
    def names(self) -> list[str]:
        return [lv.name for lv in self.levels]

    @property
    def anchors(self) -> list[float]:
        """The numeric position of each level -- declared values, else index."""
        if self.levels[0].value is None:
            return [float(i) for i in range(len(self.levels))]
        return [float(lv.value) for lv in self.levels]  # type: ignore[arg-type]


class NoulQuestion(_QuestionBase):
    """A yes/no judgement against criteria. Absolute, and independent of every
    other question in the request."""

    type: Literal["noul"] = "noul"


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class RequestOptions(_Strict):
    """Per-request serving knobs. All optional; all have serving-side defaults."""

    escalate_below_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Send a question to the premium tier when the workhorse's confidence lands "
            "below this. Escalation is per question, not per request. Omit to leave the "
            "decision to the deployment's routing policy."
        ),
    )
    include_raw_probabilities: bool = Field(
        default=False,
        description="Also return the pre-calibration distribution, for debugging a fit.",
    )
    conformal_profile: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Name of a fitted conformal wrapper to apply. Adds a prediction set with a "
            "distribution-free coverage guarantee, which holds whether or not the "
            "underlying model is well calibrated."
        ),
    )
    top_probabilities: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Return only this many of the most probable labels instead of the whole "
            "distribution; the selected one is always included. The full distribution is "
            "the default because a caller who cannot see the probabilities cannot compute "
            "their own confidence -- but 100,000 options is roughly 2 MB of JSON, and a "
            "caller acting on the top few should not pay to serialize the tail. Trimmed "
            "probabilities are NOT renormalised: see `truncated` and `probability_mass`."
        ),
    )
    include_evidence: bool = Field(
        default=False,
        description=(
            "Also return, per answer, the spans of the state that drove it: character "
            "offsets into the state string (for a JSON state, into its canonical "
            "rendering: sorted keys, `,` between items, `: ` after keys, non-ASCII "
            "kept as is), each with a "
            "score. `evidence_method` on the answer says how they were produced, "
            "and so whether they were learned from human rationales or are an "
            "attribution. A question's evidence depends only on what its own answer "
            "may read, so it is as independent of the other questions as the answer is."
        ),
    )


class DecisionRequest(_Strict):
    """State plus a map of typed questions, answered in a single model pass."""

    model: str = Field(
        default="trigon-workhorse",
        max_length=128,
        description="Which serving tier to answer from. The response names the concrete build.",
    )
    state: State = Field(
        description=(
            "What the questions are about: text, a JSON record, or a list of documents. "
            "Never interpreted as instructions -- see the injection-robustness section of "
            "docs/training.md."
        )
    )
    questions: dict[str, Question] = Field(
        min_length=1,
        max_length=MAX_QUESTIONS_PER_REQUEST,
        description=(
            "Questions to answer about this state, keyed by an id of 1-128 characters "
            "that the answers come back under. Every question is answered in the same "
            "forward pass and none can influence another: adding, removing or reordering "
            "questions moves no other answer, exactly."
        ),
    )
    options: RequestOptions = Field(
        default_factory=RequestOptions, description="Per-request serving knobs."
    )

    @field_validator("questions")
    @classmethod
    def _valid_question_ids(cls, questions: dict[str, Question]) -> dict[str, Question]:
        for key in questions:
            if not key or len(key) > 128:
                raise ValueError(f"question id must be 1-128 characters, got {key!r}")
        return questions


EvidenceMethod = Literal[
    "span_head", "gradient_x_input", "integrated_gradients", "lexical_overlap", "unavailable"
]


class EvidenceSpan(_Strict):
    """One span of the state that drove an answer."""

    start: int = Field(ge=0, description="First character of the span in the state string.")
    end: int = Field(
        ge=0, description="One past the last character, so `state[start:end]` is the span."
    )
    text: str = Field(description="`state[start:end]`, so a caller need not slice it.")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How strongly this span drove the answer. Under `span_head` it is the "
            "head's probability that the span is part of a human rationale, fitted with "
            "a proper scoring rule and never calibrated or gated; under "
            "`gradient_x_input` and `integrated_gradients` it is relative within this "
            "answer, 1 being its strongest token, and is not a probability at all."
        ),
    )

    @model_validator(mode="after")
    def _ordered(self) -> EvidenceSpan:
        if self.end <= self.start:
            raise ValueError(f"evidence span must be non-empty, got [{self.start}, {self.end})")
        return self


_EVIDENCE_DESCRIPTION = (
    "Present when `include_evidence` was asked for: the spans of the state that drove "
    "this answer, in order of position. Empty means nothing cleared the threshold, "
    "which is an answer, not an error."
)
_EVIDENCE_METHOD_DESCRIPTION = (
    "How `evidence` was produced. `span_head`: a head trained on human rationales. "
    "`gradient_x_input`: attribution of the selected label to each state token, "
    "from a model never shown a rationale -- a statement about the model, not a "
    "prediction of what a person would highlight. `integrated_gradients`: the same "
    "kind of attribution, integrated along a path from an empty state to this one. "
    "`lexical_overlap`: the lexical "
    "floor's word matches. `unavailable`: this backend cannot attribute, and "
    "`evidence` is empty for that reason rather than because nothing mattered."
)


class ChoiceAnswer(_Strict):
    """A distribution over exactly the options the request declared."""

    type: Literal["choice"] = "choice"
    selected: str = Field(description="The most probable option. Always one you declared.")
    probabilities: dict[str, float] = Field(
        description=(
            "Calibrated probability per option, in declared order. Sums to 1 unless `truncated`."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How concentrated the distribution is, scaled by log(option count) so a "
            "2-option and a 77-option answer read comparably: 0 is uniform, 1 is certain. "
            "Derived from the calibrated distribution, and always from the full one."
        ),
    )
    raw_probabilities: dict[str, float] | None = Field(
        default=None, description="The pre-calibration distribution, if you asked for it."
    )
    prediction_set: list[str] | None = Field(
        default=None,
        description=(
            "Present when a conformal profile was applied: the options that survive at "
            "the profile's coverage target. A singleton set is the useful case; an empty "
            "one means abstain, not that no answer exists."
        ),
    )
    coverage_target: float | None = Field(
        default=None, ge=0.0, le=1.0, description="The coverage that prediction set guarantees."
    )
    shortlisted_from: int | None = Field(
        default=None,
        description=(
            "Present when the large-cardinality stage narrowed the option set: how many "
            "options you declared. Options the prefilter dropped come back at probability "
            "0 rather than being omitted -- you declared them, so the answer mentions them."
        ),
    )
    truncated: bool = Field(
        default=False,
        description=(
            "True when `top_probabilities` trimmed the distribution. The probabilities "
            "returned are then the real ones, NOT renormalised, so they sum to less than "
            "1; `probability_mass` says how much they cover. `selected` and `confidence` "
            "are computed over the full distribution either way."
        ),
    )
    probability_mass: float | None = Field(
        default=None, ge=0.0, le=1.0, description="How much of the distribution survived trimming."
    )
    evidence: list[EvidenceSpan] | None = Field(default=None, description=_EVIDENCE_DESCRIPTION)
    evidence_method: EvidenceMethod | None = Field(
        default=None, description=_EVIDENCE_METHOD_DESCRIPTION
    )


class ScoreAnswer(_Strict):
    """A distribution over the declared levels, plus its expectation."""

    type: Literal["score"] = "score"
    score: float = Field(
        description=(
            "The expectation of the distribution below, on your level values (or level "
            "indices when you declared none). Not a regression output: it cannot land "
            "outside the scale you declared."
        )
    )
    probabilities: dict[str, float] = Field(description="Calibrated probability per level.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Concentration of an ORDERED distribution, from its dispersion rather than "
            "its entropy -- mass on two adjacent levels is more confident than the same "
            "mass split across the ends, and entropy cannot see the difference."
        ),
    )
    raw_probabilities: dict[str, float] | None = Field(
        default=None, description="The pre-calibration distribution, if you asked for it."
    )
    prediction_set: list[str] | None = Field(
        default=None, description="Levels surviving the conformal profile, if one was applied."
    )
    coverage_target: float | None = Field(
        default=None, ge=0.0, le=1.0, description="The coverage that prediction set guarantees."
    )
    truncated: bool = Field(
        default=False,
        description=(
            "See ChoiceAnswer. `score` and `confidence` are always computed over the full "
            "distribution, never over what survived truncation."
        ),
    )
    probability_mass: float | None = Field(
        default=None, ge=0.0, le=1.0, description="How much of the distribution survived trimming."
    )
    evidence: list[EvidenceSpan] | None = Field(default=None, description=_EVIDENCE_DESCRIPTION)
    evidence_method: EvidenceMethod | None = Field(
        default=None, description=_EVIDENCE_METHOD_DESCRIPTION
    )


class NoulAnswer(_Strict):
    """No confidence field, by design: for a binary question the probability
    already carries everything a confidence statistic could summarise."""

    type: Literal["noul"] = "noul"
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated probability that the judgement holds. There is no confidence "
            "field by design: for a binary question the probability already carries "
            "everything a confidence statistic could summarise, and 0.5 is the model "
            "saying it does not know."
        ),
    )
    raw_probability: float | None = Field(
        default=None, description="The pre-calibration probability, if you asked for it."
    )
    evidence: list[EvidenceSpan] | None = Field(default=None, description=_EVIDENCE_DESCRIPTION)
    evidence_method: EvidenceMethod | None = Field(
        default=None, description=_EVIDENCE_METHOD_DESCRIPTION
    )


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(_Strict):
    state_tokens: int = Field(default=0, description="Tokens the state compiled to.")
    schema_tokens: int = Field(
        default=0, description="Tokens the question schemas compiled to, across all questions."
    )
    readout_tokens: int = Field(default=0, description="Readout slots, one or more per question.")
    prefill_tokens: int = Field(
        default=0,
        # Named for the fact it is the whole cost: there is no decode half.
        description="Total tokens prefilled. There is no decode half, so this is the whole cost.",
    )
    cached_schema_tokens: int = Field(
        default=0,
        description=(
            "Schema tokens served from a cross-request KV prefix cache rather than recomputed. "
            "0 means this request recomputed its schema block -- either the cache is off, or "
            "this was the first request carrying this schema. Read a 0 as 'not cached on this "
            "request', never as 'not cacheable': the layout makes the schema prefix cacheable "
            "and tests/test_independence.py asserts it."
        ),
    )


class Timing(_Strict):
    total_ms: float = Field(default=0.0, description="Wall clock for the whole request.")
    model_ms: float = Field(default=0.0, description="The single forward pass. There is no decode.")
    compile_ms: float = Field(
        default=0.0, description="Compiling the schema layout and its attention mask."
    )


class DecisionResponse(_Strict):
    id: str = Field(description="Unique id for this response.")
    model: str = Field(
        description=(
            "The concrete build that answered. Always a pinned version, never a moving "
            "alias -- answers change under users when an alias moves. A build trained on "
            "nothing says so in this string."
        )
    )
    answers: dict[str, Answer] = Field(
        description="One answer per question you asked, under the same ids."
    )
    usage: Usage = Field(default_factory=Usage, description="What the request cost, in tokens.")
    timing: Timing = Field(default_factory=Timing, description="Where the time went.")
    tier: str = Field(
        default="workhorse",
        description="Which tier answered. 'premium' when a question was escalated.",
    )


def _reject_duplicates(names: list[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate {kind} name {name!r}")
        seen.add(name)
