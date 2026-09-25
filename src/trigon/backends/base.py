"""The backend boundary: compiled request in, per-question logits out.

Everything above this line -- schema compilation, calibration, confidence,
the wire contract -- is shared by every backend. Everything below it is a way
of producing logits: a trained readout head, a prompted LLM baseline, or the
lexical floor. Keeping the boundary at *logits* rather than at probabilities is
deliberate: temperature scaling has to happen on logits, and a backend that can
only return probabilities cannot be calibrated post hoc.

The contract every backend must honour:

* one entry per question in the compiled request, keyed by question id;
* Choice and Score return one logit per declared option or level, in declared
  order; Noul returns exactly one;
* no backend may invent, drop or reorder a label. This is where structural type
  safety is actually enforced -- ``validate_output`` below is called on every
  path, so a broken backend fails loudly instead of serving a schema violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..schema import CompiledRequest, SegmentKind
from ..schema.tokens import TokenEstimator
from ..types import SystemOneRequest

__all__ = ["Backend", "BackendOutput", "QuestionOutput", "estimator_of", "validate_output"]


@dataclass(frozen=True)
class QuestionOutput:
    """Raw head output for one question."""

    question_id: str
    kind: str
    logits: tuple[float, ...]
    # Filled when the large-cardinality stage scored a shortlist: maps position
    # in ``logits`` to position in the caller's declared option list.
    shortlist_indices: tuple[int, ...] | None = None
    #: Filled only when the request asked for evidence and this backend can
    #: attribute: ``(start, end, score)`` per state token, character offsets
    #: into the rendered state, score in [0, 1]. Tokens, never spans -- the
    #: engine merges them under one rule for every backend (`trigon.evidence`).
    evidence: tuple[tuple[int, int, float], ...] | None = None
    #: How ``evidence`` was produced; one of the contract's `EvidenceMethod`.
    evidence_method: str | None = None


@dataclass(frozen=True)
class BackendOutput:
    """Everything one forward pass produced."""

    outputs: dict[str, QuestionOutput]
    model_version: str
    tier: str = "workhorse"
    model_ms: float = 0.0
    cached_schema_tokens: int = 0
    # Free-form, surfaced in eval reports but never in the wire response.
    diagnostics: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """A source of logits."""

    @property
    def model_version(self) -> str:
        """Pinned version string. Never an alias -- answers change under users
        when an alias moves, so the response always names a concrete build."""
        ...

    def infer(self, compiled: CompiledRequest, request: SystemOneRequest) -> BackendOutput: ...


def estimator_of(backend: object) -> TokenEstimator | None:
    """The backend's own tokenizer, if it has an exact one.

    A backend whose tensors are built from a real tokenizer must be compiled
    with that same tokenizer, or the compiled token counts and the tensors
    disagree and inference fails. Rather than asking every call site to
    remember ``backend.make_compiler()``, ``Engine`` reads this and wires the
    compiler itself -- so the gateway, the CLI and the eval harness cannot
    drift apart on it. Backends without an exact tokenizer return ``None`` and
    get the character heuristic.
    """
    return getattr(backend, "estimator", None)


def validate_output(output: BackendOutput, compiled: CompiledRequest) -> None:
    """Assert the structural guarantee. Raises ``ValueError`` on violation."""
    expected = {q.question_id: q for q in compiled.schema.questions}
    state_length = sum(len(s.text) for s in compiled.segments if s.kind is SegmentKind.STATE)
    missing = expected.keys() - output.outputs.keys()
    if missing:
        raise ValueError(f"backend returned no logits for {sorted(missing)}")
    extra = output.outputs.keys() - expected.keys()
    if extra:
        raise ValueError(f"backend returned logits for undeclared questions {sorted(extra)}")

    for qid, compiled_q in expected.items():
        got = output.outputs[qid]
        if got.kind != compiled_q.kind:
            raise ValueError(
                f"question {qid!r}: backend returned a {got.kind} head for a "
                f"{compiled_q.kind} question"
            )
        want = 1 if compiled_q.kind == "noul" else compiled_q.cardinality
        if got.shortlist_indices is not None:
            if len(got.shortlist_indices) != len(got.logits):
                raise ValueError(
                    f"question {qid!r}: {len(got.logits)} logits against "
                    f"{len(got.shortlist_indices)} shortlist indices"
                )
            if any(not 0 <= i < want for i in got.shortlist_indices):
                raise ValueError(
                    f"question {qid!r}: shortlist points outside the declared {want}-option set"
                )
        elif len(got.logits) != want:
            raise ValueError(f"question {qid!r}: expected {want} logits, got {len(got.logits)}")
        if any(_is_not_finite(x) for x in got.logits):
            raise ValueError(f"question {qid!r}: non-finite logit")
        if got.evidence is not None:
            _validate_evidence(qid, got.evidence, state_length)


def _validate_evidence(qid: str, evidence, state_length: int) -> None:
    """Offsets inside the state, scores in [0, 1]: a span pointing past the
    state would slice the wrong text and nothing downstream would notice."""
    for start, end, score in evidence:
        if not 0 <= start < end <= state_length:
            raise ValueError(
                f"question {qid!r}: evidence [{start}, {end}) lies outside the "
                f"{state_length}-character state"
            )
        if _is_not_finite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"question {qid!r}: evidence score {score} is not in [0, 1]")


def _is_not_finite(x: float) -> bool:
    return x != x or x in (float("inf"), float("-inf"))
