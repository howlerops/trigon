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
    "LevelSpec",
    "NoulAnswer",
    "NoulQuestion",
    "OptionSpec",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "State",
    "SystemOneRequest",
    "SystemOneResponse",
    "Usage",
]

# State is whatever the caller already has: a blob of text, a JSON record, or a
# list of documents. It is never interpreted as instructions -- see the
# injection-robustness section of docs/training.md.
State = str | dict[str, Any] | list[Any]

_NAME = Field(min_length=1, max_length=256)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OptionSpec(_Strict):
    """One option of a Choice, with the criteria that select it."""

    name: str = _NAME
    criteria: str | None = Field(default=None, max_length=8192)


class LevelSpec(_Strict):
    """One level of a Score. Levels are ordered low to high as given."""

    name: str = _NAME
    criteria: str | None = Field(default=None, max_length=8192)
    # Anchors the level on the caller's own scale (1-5 stars, 0-4 Likert). When
    # omitted, levels are anchored at their index, so the reported score is the
    # expected level index.
    value: float | None = None


class _QuestionBase(_Strict):
    instructions: str = Field(min_length=1, max_length=32768)


class ChoiceQuestion(_QuestionBase):
    """Pick one option from a defined set. Relative: it settles *which*."""

    type: Literal["choice"] = "choice"
    options: list[OptionSpec] = Field(min_length=2, max_length=MAX_OPTIONS_PER_QUESTION)

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
    levels: list[LevelSpec] = Field(min_length=2, max_length=MAX_LEVELS_PER_SCORE)

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

    # Escalate to the premium tier when workhorse confidence lands below this.
    # ``None`` leaves the decision to the deployment's routing policy.
    escalate_below_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    # Return the pre-calibration distribution alongside the calibrated one.
    include_raw_probabilities: bool = False
    # Name of a fitted conformal wrapper to apply, if the deployment has one.
    conformal_profile: str | None = Field(default=None, max_length=128)
    # Return only the most probable options rather than the whole
    # distribution. The full distribution is the default because returning it
    # is the product's whole point -- a caller who cannot see the probabilities
    # cannot compute their own confidence. But at tens of thousands of options
    # the response itself becomes the bottleneck (100,000 options is roughly
    # 2 MB of JSON), and a caller who only acts on the top few should not pay
    # to serialize the tail. The selected option is always included.
    top_probabilities: int | None = Field(default=None, ge=1)


class SystemOneRequest(_Strict):
    """State plus a map of typed questions, answered in a single model pass."""

    model: str = Field(default="trigon-workhorse", max_length=128)
    state: State
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_REQUEST)
    options: RequestOptions = Field(default_factory=RequestOptions)

    @field_validator("questions")
    @classmethod
    def _valid_question_ids(cls, questions: dict[str, Question]) -> dict[str, Question]:
        for key in questions:
            if not key or len(key) > 128:
                raise ValueError(f"question id must be 1-128 characters, got {key!r}")
        return questions


class ChoiceAnswer(_Strict):
    type: Literal["choice"] = "choice"
    selected: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    raw_probabilities: dict[str, float] | None = None
    # Present when a conformal profile was applied: the options that survive at
    # the profile's coverage target. A singleton set is the useful case.
    prediction_set: list[str] | None = None
    coverage_target: float | None = Field(default=None, ge=0.0, le=1.0)
    # Present when the large-cardinality stage narrowed the option set; options
    # outside the shortlist carry probability 0 rather than being omitted.
    shortlisted_from: int | None = None
    # True when ``top_probabilities`` trimmed the distribution. The reported
    # probabilities are then the real ones, not renormalised, so they sum to
    # less than 1 -- and ``probability_mass`` says how much they cover.
    truncated: bool = False
    probability_mass: float | None = Field(default=None, ge=0.0, le=1.0)


class ScoreAnswer(_Strict):
    type: Literal["score"] = "score"
    score: float
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    raw_probabilities: dict[str, float] | None = None
    prediction_set: list[str] | None = None
    coverage_target: float | None = Field(default=None, ge=0.0, le=1.0)
    # See ChoiceAnswer: the score and the confidence are always computed over
    # the full distribution, never over what survived truncation.
    truncated: bool = False
    probability_mass: float | None = Field(default=None, ge=0.0, le=1.0)


class NoulAnswer(_Strict):
    """No confidence field, by design: for a binary question the probability
    already carries everything a confidence statistic could summarise."""

    type: Literal["noul"] = "noul"
    probability: float = Field(ge=0.0, le=1.0)
    raw_probability: float | None = None


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(_Strict):
    state_tokens: int = 0
    schema_tokens: int = 0
    readout_tokens: int = 0
    # Named for the fact it is the whole cost: there is no decode half.
    prefill_tokens: int = 0
    cached_schema_tokens: int = 0


class Timing(_Strict):
    total_ms: float = 0.0
    model_ms: float = 0.0
    compile_ms: float = 0.0


class SystemOneResponse(_Strict):
    id: str
    # Always a pinned version, never a moving alias: answers change under users
    # when an alias moves.
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    timing: Timing = Field(default_factory=Timing)
    # Populated when the request was escalated to the premium tier.
    tier: str = "workhorse"


def _reject_duplicates(names: list[str], kind: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate {kind} name {name!r}")
        seen.add(name)
