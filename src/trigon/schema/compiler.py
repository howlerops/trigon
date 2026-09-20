"""Compile typed questions into a prefill layout, a block mask and a cache key.

Layout is schema-first on purpose::

    [ schema: q1 block | q2 block | ... ] [ state ] [ readout slots ]

Schema tokens depend only on the question map, so their KV is a prefix that is
identical across every request carrying that schema -- that is the cross-request
cache the whole serving story rests on. State comes next because it varies per
request. Readout slots trail everything so they can see both, and so the causal
fallback layout (``bidirectional=False``) still works without a rewrite.

Three isolation rules, all enforced by the mask rather than by convention:

* a question's readout may not attend to any other question's readout, which is
  what makes Noul answers independent of each other;
* with ``isolate_question_schemas`` (the default), a question's schema block may
  not attend to another question's schema block either. That costs a little
  cross-question context and buys per-question schema cache reuse: the compiled
  KV for "route to one of these 77 intents" is reusable in any request that asks
  it, not only in requests that ask exactly the same set of questions;
* with ``state_attends_to_schema`` off (the default), state encodes itself and
  nothing else.

That third rule is the one that actually delivers the no-context-rot property,
and it is easy to get wrong. If state may attend to the schema, then adding a
second question changes the state's hidden states, which changes the *first*
question's readout -- adding a question silently moves an answer the caller did
not ask about. With the rule on, a readout sees only its own schema block and a
state encoding that no other question can perturb, so per-question
independence is exact and testable (``tests/test_independence.py``), and the
state KV becomes reusable across different question sets over the same
document.

The cost is real and worth naming: state is encoded question-agnostically, so
all cross-referencing between state and schema happens in the readout slots
rather than throughout the stack. Turning the flag on recovers that capacity
and gives up the guarantee. Which side wins is the phase-1 ablation; the
default is the one whose published claim we can actually stand behind.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from ..limits import DEFAULT_BUDGET, Budget
from ..types import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    State,
    SystemOneRequest,
)
from .tokens import CharHeuristicEstimator, TokenEstimator

__all__ = [
    "AttentionPlan",
    "CompiledQuestion",
    "CompiledRequest",
    "CompiledSchema",
    "OptionScoring",
    "SchemaCompiler",
    "Segment",
    "SegmentKind",
    "SchemaTooLarge",
    "compile_request",
    "compile_schema",
    "materialize_mask",
]

# Bump when the layout or the serialization changes: the hash is a cache key,
# and a stale hit would serve KV built under different rules.
LAYOUT_VERSION = 1


class SchemaTooLarge(ValueError):
    """A request cannot fit the context even with the retrieval stage engaged."""


class SegmentKind(str, Enum):
    SCHEMA_QUESTION = "schema_question"
    SCHEMA_OPTION = "schema_option"
    SCHEMA_LEVEL = "schema_level"
    STATE = "state"
    READOUT = "readout"


class OptionScoring(str, Enum):
    """How Choice option logits are produced.

    ``READOUT_PER_OPTION`` gives each option a trailing readout slot: the most
    expressive, and the cost is linear in options. ``DOT_PRODUCT`` scores one
    question readout state against pooled option embeddings: one slot no matter
    how many options, which is what makes high-cardinality sets affordable.
    ``AUTO`` picks per question. This is the phase-1 ablation.
    """

    AUTO = "auto"
    READOUT_PER_OPTION = "readout_per_option"
    DOT_PRODUCT = "dot_product"


# Above this many options, a slot per option stops paying for itself.
DOT_PRODUCT_CROSSOVER = 64


@dataclass(frozen=True)
class Segment:
    """One contiguous span of the prefill sequence."""

    kind: SegmentKind
    text: str
    tokens: int
    # Which question this belongs to; ``None`` only for shared state.
    question_id: str | None = None
    # Index of the option or level this segment encodes, when it encodes one.
    member_index: int | None = None

    @property
    def group(self) -> str:
        """Isolation group. The mask is defined over groups, not tokens."""
        if self.kind is SegmentKind.STATE:
            return "state"
        if self.kind is SegmentKind.READOUT:
            return f"readout:{self.question_id}"
        return f"schema:{self.question_id}"


@dataclass(frozen=True)
class CompiledQuestion:
    """One question's compiled form: its cache key, its slots, its budget."""

    question_id: str
    kind: str
    # Stable across requests, so per-question schema KV can be cached by it.
    schema_hash: str
    labels: tuple[str, ...]
    schema_tokens: int
    readout_slots: int
    option_scoring: OptionScoring
    needs_retrieval: bool
    # Set when the retrieval stage narrowed the option set for this request.
    shortlisted_from: int | None = None

    @property
    def cardinality(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class AttentionPlan:
    """Which groups each group may attend to."""

    groups: tuple[str, ...]
    visibility: dict[str, frozenset[str]]
    bidirectional: bool = True

    def can_attend(self, query_group: str, key_group: str) -> bool:
        return key_group in self.visibility.get(query_group, frozenset())


@dataclass(frozen=True)
class CompiledSchema:
    """The request-independent half: everything derivable from the questions."""

    schema_hash: str
    questions: tuple[CompiledQuestion, ...]
    segments: tuple[Segment, ...]
    total_tokens: int
    layout_version: int = LAYOUT_VERSION

    def question(self, question_id: str) -> CompiledQuestion:
        for q in self.questions:
            if q.question_id == question_id:
                return q
        raise KeyError(question_id)


@dataclass(frozen=True)
class CompiledRequest:
    """Schema plus state plus readouts: the full sequence handed to a backend."""

    schema: CompiledSchema
    segments: tuple[Segment, ...]
    attention: AttentionPlan
    state_tokens: int
    readout_tokens: int
    budget: Budget = field(default=DEFAULT_BUDGET)

    @property
    def schema_tokens(self) -> int:
        return self.schema.total_tokens

    @property
    def total_tokens(self) -> int:
        return self.schema_tokens + self.state_tokens + self.readout_tokens


class SchemaCompiler:
    """Turns a request into a layout. Stateless apart from its configuration."""

    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        budget: Budget = DEFAULT_BUDGET,
        *,
        option_scoring: OptionScoring = OptionScoring.AUTO,
        isolate_question_schemas: bool = True,
        state_attends_to_schema: bool = False,
        bidirectional: bool = True,
    ) -> None:
        self.estimator = estimator or CharHeuristicEstimator()
        self.budget = budget
        self.option_scoring = option_scoring
        self.isolate_question_schemas = isolate_question_schemas
        self.state_attends_to_schema = state_attends_to_schema
        self.bidirectional = bidirectional

    # -- schema half -----------------------------------------------------

    def compile_schema(self, questions: dict[str, Question]) -> CompiledSchema:
        if len(questions) > self.budget.max_questions:
            raise SchemaTooLarge(
                f"{len(questions)} questions exceeds the per-request limit of "
                f"{self.budget.max_questions}"
            )
        compiled: list[CompiledQuestion] = []
        segments: list[Segment] = []
        for qid in sorted(questions):  # sorted so the hash ignores map order
            cq, segs = self._compile_question(qid, questions[qid])
            compiled.append(cq)
            segments.extend(segs)

        total = sum(s.tokens for s in segments)
        if total > self.budget.schema_tokens:
            raise SchemaTooLarge(
                f"compiled schema is {total} tokens, over the "
                f"{self.budget.schema_tokens}-token schema budget; shorten criteria "
                f"or enable the retrieval stage for the large questions"
            )
        return CompiledSchema(
            schema_hash=_hash({"v": LAYOUT_VERSION, "q": [c.schema_hash for c in compiled]}),
            questions=tuple(compiled),
            segments=tuple(segments),
            total_tokens=total,
        )

    def plan_question(self, qid: str, question: Question) -> CompiledQuestion:
        """Size one question without compiling the whole request.

        The engine needs this before compiling, because a question that trips
        the retrieval trigger has to be narrowed *first* -- otherwise the
        compiler rejects the request it was about to make fit.
        """
        return self._compile_question(qid, question)[0]

    def _compile_question(
        self, qid: str, question: Question
    ) -> tuple[CompiledQuestion, list[Segment]]:
        # Count the text that is actually emitted, header included: a backend
        # with an exact tokenizer asserts span lengths against these numbers.
        header = f"[{question.type}:{qid}] {question.instructions}"
        segments = [
            Segment(
                kind=SegmentKind.SCHEMA_QUESTION,
                text=header,
                tokens=self.estimator.count(header),
                question_id=qid,
            )
        ]
        if isinstance(question, ChoiceQuestion):
            members = question.options
            member_kind = SegmentKind.SCHEMA_OPTION
        elif isinstance(question, ScoreQuestion):
            members = question.levels
            member_kind = SegmentKind.SCHEMA_LEVEL
        else:
            members = []
            member_kind = SegmentKind.SCHEMA_OPTION

        for i, member in enumerate(members):
            text = member.name if member.criteria is None else f"{member.name}: {member.criteria}"
            segments.append(
                Segment(
                    kind=member_kind,
                    text=text,
                    tokens=self.estimator.count(text),
                    question_id=qid,
                    member_index=i,
                )
            )

        tokens = sum(s.tokens for s in segments)
        labels = _labels(question)
        scoring = self._resolve_scoring(question, len(labels))
        needs_retrieval = isinstance(question, ChoiceQuestion) and self.budget.needs_retrieval(
            len(labels), tokens
        )
        if not needs_retrieval and tokens > self.budget.max_question_tokens:
            # A non-Choice question cannot be narrowed, so an oversized one is
            # a rejection rather than a retrieval case.
            raise SchemaTooLarge(
                f"question {qid!r} compiles to {tokens} tokens, over the "
                f"{self.budget.max_question_tokens}-token per-question budget"
            )
        return (
            CompiledQuestion(
                question_id=qid,
                kind=question.type,
                schema_hash=_hash(
                    {"v": LAYOUT_VERSION, "id": qid, "q": question.model_dump(mode="json")}
                ),
                labels=tuple(labels),
                schema_tokens=tokens,
                readout_slots=_readout_slots(question, scoring, len(labels)),
                option_scoring=scoring,
                needs_retrieval=needs_retrieval,
            ),
            segments,
        )

    def _resolve_scoring(self, question: Question, cardinality: int) -> OptionScoring:
        if not isinstance(question, ChoiceQuestion):
            return OptionScoring.DOT_PRODUCT
        if self.option_scoring is not OptionScoring.AUTO:
            return self.option_scoring
        if cardinality > DOT_PRODUCT_CROSSOVER:
            return OptionScoring.DOT_PRODUCT
        return OptionScoring.READOUT_PER_OPTION

    # -- full request ----------------------------------------------------

    def compile_request(self, request: SystemOneRequest) -> CompiledRequest:
        schema = self.compile_schema(request.questions)
        state_text = render_state(request.state)
        state_tokens = self.estimator.count(state_text)
        if state_tokens > self.budget.state_tokens:
            raise SchemaTooLarge(
                f"state is ~{state_tokens} tokens, over the "
                f"{self.budget.state_tokens}-token state budget"
            )
        state_segment = Segment(
            kind=SegmentKind.STATE, text=state_text, tokens=state_tokens, question_id=None
        )

        # The contract we mirror bounds state plus the LONGEST SINGLE question,
        # separately from the total. A request can sit well inside the total
        # budget and still be inadmissible on this one.
        longest = max(schema.questions, key=lambda q: q.schema_tokens)
        if not self.budget.fits_envelope(state_tokens, longest.schema_tokens):
            raise SchemaTooLarge(
                f"state (~{state_tokens} tokens) plus the longest question "
                f"{longest.question_id!r} (~{longest.schema_tokens} tokens) exceeds the "
                f"{self.budget.single_question_envelope}-token per-question envelope"
            )

        readouts: list[Segment] = []
        for cq in schema.questions:
            for slot in range(cq.readout_slots):
                readouts.append(
                    Segment(
                        kind=SegmentKind.READOUT,
                        text="",
                        tokens=1,
                        question_id=cq.question_id,
                        member_index=slot if cq.readout_slots > 1 else None,
                    )
                )
        readout_tokens = len(readouts)
        if readout_tokens > self.budget.readout_tokens:
            raise SchemaTooLarge(
                f"{readout_tokens} readout slots exceed the "
                f"{self.budget.readout_tokens}-slot budget; use dot-product option "
                f"scoring for the high-cardinality questions"
            )

        segments = (*schema.segments, state_segment, *readouts)
        total = sum(s.tokens for s in segments)
        if total > self.budget.context_tokens:
            raise SchemaTooLarge(
                f"compiled request is ~{total} tokens, over the "
                f"{self.budget.context_tokens}-token context"
            )
        return CompiledRequest(
            schema=schema,
            segments=segments,
            attention=self._attention_plan(schema),
            state_tokens=state_tokens,
            readout_tokens=readout_tokens,
            budget=self.budget,
        )

    def _attention_plan(self, schema: CompiledSchema) -> AttentionPlan:
        schema_groups = [f"schema:{q.question_id}" for q in schema.questions]
        groups = [*schema_groups, "state", *[f"readout:{q.question_id}" for q in schema.questions]]
        visibility: dict[str, frozenset[str]] = {}

        for q in schema.questions:
            sg = f"schema:{q.question_id}"
            # Schema never sees state: that is what keeps its KV cacheable
            # across every request carrying the same schema.
            visibility[sg] = frozenset([sg] if self.isolate_question_schemas else schema_groups)

        # Off by default: see the module docstring. On, the request still works
        # but per-question independence stops being exact.
        visibility["state"] = frozenset(
            [*schema_groups, "state"] if self.state_attends_to_schema else ["state"]
        )

        for q in schema.questions:
            # A readout sees its own question's schema and the state, and
            # nothing else -- no other question's schema, no other readout.
            visibility[f"readout:{q.question_id}"] = frozenset(
                [f"schema:{q.question_id}", "state", f"readout:{q.question_id}"]
            )
        return AttentionPlan(
            groups=tuple(groups), visibility=visibility, bidirectional=self.bidirectional
        )


# Masks are quadratic in sequence length and identical across every request
# with the same shape -- which, for a schema served at volume, is most of them.
# Building one costs more than the forward pass does at spike sizes, so the
# result is memoized on the shape rather than on the request.
_MASK_CACHE: dict[object, list[list[bool]]] = {}
_MASK_CACHE_LIMIT = 256


def mask_shape_key(compiled: CompiledRequest) -> tuple:
    """Everything the mask depends on, and nothing else.

    Not the schema hash: two different schemas with the same segment shape and
    the same visibility produce the same mask, and the same schema with a
    different state length does not.
    """
    plan = compiled.attention
    return (
        tuple((seg.group, seg.tokens) for seg in compiled.segments),
        plan.bidirectional,
        tuple(sorted((g, tuple(sorted(v))) for g, v in plan.visibility.items())),
    )


def materialize_mask(compiled: CompiledRequest) -> list[list[bool]]:
    """Expand the group-level plan to a token-level boolean mask.

    ``mask[i][j]`` is True when token ``i`` may attend to token ``j``. Used by
    the torch backend and, more importantly, by the tests that prove the
    isolation rules hold at token granularity.

    The returned mask is shared between callers with the same shape. Treat it
    as read-only; copy it before mutating.
    """
    key = mask_shape_key(compiled)
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    owners: list[str] = []
    for seg in compiled.segments:
        owners.extend([seg.group] * seg.tokens)

    n = len(owners)
    plan = compiled.attention
    mask = [[False] * n for _ in range(n)]
    # Group-level lookups are hoisted out of the inner loop: the visibility
    # answer is the same for every token pair in a pair of groups.
    groups = sorted(set(owners))
    allowed = {(a, b): plan.can_attend(a, b) for a in groups for b in groups}
    for i, qg in enumerate(owners):
        row = mask[i]
        for j, kg in enumerate(owners):
            if not allowed[(qg, kg)]:
                continue
            # Within a group the causal fallback still applies when the model
            # was not converted to prefix-LM attention.
            row[j] = plan.bidirectional or j <= i

    if len(_MASK_CACHE) >= _MASK_CACHE_LIMIT:
        _MASK_CACHE.clear()
    _MASK_CACHE[key] = mask
    return mask


def render_state(state: State) -> str:
    """Canonical text form of the state.

    JSON state is serialized with sorted keys so that two callers sending the
    same record produce the same tokens, and so state hashes are stable.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ": "))


def _labels(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return question.names
    if isinstance(question, ScoreQuestion):
        return question.names
    if isinstance(question, NoulQuestion):
        return ["yes"]
    raise TypeError(f"unsupported question type {type(question).__name__}")


def _readout_slots(question: Question, scoring: OptionScoring, cardinality: int) -> int:
    if isinstance(question, ChoiceQuestion) and scoring is OptionScoring.READOUT_PER_OPTION:
        return cardinality
    # Score and Noul always read out from a single slot: their logits come from
    # a fixed-width head, not from per-member states.
    return 1


def _hash(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


_DEFAULT = SchemaCompiler()


def compile_schema(questions: dict[str, Question]) -> CompiledSchema:
    """Compile with the default compiler configuration."""
    return _DEFAULT.compile_schema(questions)


def compile_request(request: SystemOneRequest) -> CompiledRequest:
    """Compile with the default compiler configuration."""
    return _DEFAULT.compile_request(request)
