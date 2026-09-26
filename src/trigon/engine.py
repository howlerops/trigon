"""The serving pipeline: compile, shortlist, infer, calibrate, answer.

One place owns the order of operations, so the gateway and the eval harness
cannot drift apart -- and so the phase-4 SDKs have one pipeline to mirror
rather than three:

1. narrow any question that trips the large-cardinality trigger;
2. compile the schema-first layout and its block mask;
3. run one forward pass and check the structural guarantee;
4. apply the fitted temperature, then derive confidence from the result;
5. optionally wrap in a conformal prediction set;
6. optionally attach evidence, merged from the backend's per-token scores
   under the one rule in ``trigon.evidence``.

Step 4 is in that order on purpose. Confidence is a statistic of the
*calibrated* distribution -- deriving it from raw logits would produce a
number that looks like a probability and is not one, which is the failure mode
the whole project exists to avoid.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from .backends.base import Backend, QuestionOutput, estimator_of, validate_output
from .calibration.conformal import ConformalPredictor
from .calibration.isotonic import IsotonicCalibrator
from .calibration.temperature import TemperatureScaler
from .confidence import ConfidenceMethod, choice_confidence, score_confidence
from .evidence import EVIDENCE_THRESHOLD, merge_spans
from .limits import DEFAULT_BUDGET, RETRIEVAL_SHORTLIST_SIZE, Budget
from .numeric import expectation, softmax
from .retrieval import LexicalShortlister, Shortlister
from .schema import SchemaCompiler, render_state
from .types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    EvidenceSpan,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Timing,
    Usage,
)

__all__ = ["Engine", "EngineConfig"]


@dataclass
class EngineConfig:
    """Serving-side knobs that are deployment decisions, not request options."""

    choice_confidence: ConfidenceMethod = ConfidenceMethod.ENTROPY
    score_confidence: ConfidenceMethod = ConfidenceMethod.DISPERSION
    # Selects the per-domain temperature and conformal profile when a
    # deployment has fitted them.
    domain: str | None = None
    shortlist_size: int = RETRIEVAL_SHORTLIST_SIZE
    budget: Budget = field(default=DEFAULT_BUDGET)
    tier: str = "workhorse"
    #: Where a token's evidence score becomes a span. `trigon.evidence` says
    #: what the default means for each method.
    evidence_threshold: float = EVIDENCE_THRESHOLD


class Engine:
    """Turns a request into an answer using one backend."""

    def __init__(
        self,
        backend: Backend,
        *,
        compiler: SchemaCompiler | None = None,
        scaler: TemperatureScaler | None = None,
        isotonic: IsotonicCalibrator | None = None,
        conformal: dict[str, ConformalPredictor] | None = None,
        shortlister: Shortlister | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.backend = backend
        # The backend's own tokenizer wins when it has one: compiled token
        # counts have to match the tensors the backend builds, and a mismatch
        # is a 500 at serve time rather than anything a caller can fix.
        self.compiler = compiler or SchemaCompiler(
            budget=self.config.budget, estimator=estimator_of(backend)
        )
        self.scaler = scaler or TemperatureScaler()
        # Two calibrators, selected per primitive at fit time, because neither
        # wins everywhere: a temperature takes a uniformly overconfident head
        # from ECE 0.2135 to 0.0247 and an isotonic map cannot beat that, while
        # a *tilted* head -- overconfident where it is confident, under where
        # it is not -- has no correct temperature at all and isotonic scores
        # 0.0227 against 0.1232. `docs/decisions.md`, "Neither calibrator wins".
        #
        # Both are applied when both are present, temperature first: the
        # temperature reshapes the logits and the isotonic map then corrects
        # the top-1 confidence, which is the order they were fitted in. In
        # practice `trigon train` stores at most one per primitive.
        self.isotonic = isotonic or IsotonicCalibrator()
        self.conformal = conformal or {}
        self.shortlister = shortlister or LexicalShortlister()

    # -- public ----------------------------------------------------------

    def answer(self, request: DecisionRequest) -> DecisionResponse:
        started = time.perf_counter()
        served, shortlists = self._apply_retrieval(request)

        compile_started = time.perf_counter()
        compiled = self.compiler.compile_request(served)
        compile_ms = (time.perf_counter() - compile_started) * 1000.0

        output = self.backend.infer(compiled, served)
        validate_output(output, compiled)

        answers: dict[str, Answer] = {}
        for qid, question in request.questions.items():
            answers[qid] = self._answer_one(
                qid, question, output.outputs[qid], shortlists.get(qid), request
            )

        return DecisionResponse(
            id=f"so_{uuid.uuid4().hex[:24]}",
            model=output.model_version,
            answers=answers,
            usage=Usage(
                state_tokens=compiled.state_tokens,
                schema_tokens=compiled.schema_tokens,
                readout_tokens=compiled.readout_tokens,
                prefill_tokens=compiled.total_tokens,
                cached_schema_tokens=output.cached_schema_tokens,
            ),
            timing=Timing(
                total_ms=(time.perf_counter() - started) * 1000.0,
                model_ms=output.model_ms,
                compile_ms=compile_ms,
            ),
            tier=output.tier or self.config.tier,
        )

    def answer_many(
        self, requests: Sequence[DecisionRequest], batch_size: int = 16
    ) -> list[DecisionResponse]:
        """Answer several requests, coalescing the forward passes.

        `docs/next.md` B.2. A prefill-only model makes this the easy case:
        every request is exactly one pass, so a batch is a pad and a stack --
        no decode loop, no ragged generation, no per-step scheduling.

        **An answer must not depend on what else was in the batch.** That is
        the block mask's guarantee extended across requests, and
        `tests/test_batching.py` asserts it against this method's own
        one-at-a-time path. A batcher that quietly mixes two callers' states
        produces well-formed answers to questions nobody asked, which is the
        failure this whole project is built against.

        With the schema KV cache on, a batch reads it as a single request
        does: the torch backends group the batch by schema and compute only
        the state and readout tokens, against one cached prefix per schema.
        Each response's `usage.cached_schema_tokens` is its own.

        Backends without a batched path fall through to `answer`, so this is
        always safe to call; it is faster only where the backend implements
        one. Requests are batched in arrival order rather than sorted by
        length: sorting would pad less and reorder results, and a caller
        reading `responses[i]` as the answer to `requests[i]` is a contract
        worth more than the padding.
        """
        infer_many = getattr(self.backend, "infer_many", None)
        if infer_many is None or len(requests) < 2:
            return [self.answer(request) for request in requests]

        responses: list[DecisionResponse] = []
        for start in range(0, len(requests), max(1, batch_size)):
            window = requests[start : start + max(1, batch_size)]
            prepared = []
            for request in window:
                served, shortlists = self._apply_retrieval(request)
                compile_started = time.perf_counter()
                compiled = self.compiler.compile_request(served)
                prepared.append(
                    (
                        request,
                        served,
                        shortlists,
                        compiled,
                        (time.perf_counter() - compile_started) * 1000.0,
                    )
                )

            started = time.perf_counter()
            outputs = infer_many([(compiled, served) for _, served, _, compiled, _ in prepared])
            elapsed = (time.perf_counter() - started) * 1000.0

            for (request, _, shortlists, compiled, compile_ms), output in zip(
                prepared, outputs, strict=True
            ):
                validate_output(output, compiled)
                answers = {
                    qid: self._answer_one(
                        qid, question, output.outputs[qid], shortlists.get(qid), request
                    )
                    for qid, question in request.questions.items()
                }
                responses.append(
                    DecisionResponse(
                        id=f"so_{uuid.uuid4().hex[:24]}",
                        model=output.model_version,
                        answers=answers,
                        usage=Usage(
                            state_tokens=compiled.state_tokens,
                            schema_tokens=compiled.schema_tokens,
                            readout_tokens=compiled.readout_tokens,
                            prefill_tokens=compiled.total_tokens,
                            cached_schema_tokens=output.cached_schema_tokens,
                        ),
                        timing=Timing(
                            # The batch's wall clock shared out, plus this
                            # request's own compile. Reporting each request the
                            # whole batch's time would make a batch of eight
                            # look eight times more expensive than the same
                            # work done one at a time.
                            total_ms=elapsed / len(prepared) + compile_ms,
                            model_ms=output.model_ms,
                            compile_ms=compile_ms,
                        ),
                        tier=output.tier or self.config.tier,
                    )
                )
        return responses

    # -- stages ----------------------------------------------------------

    def _apply_retrieval(
        self, request: DecisionRequest
    ) -> tuple[DecisionRequest, dict[str, list[int]]]:
        """Narrow oversized Choice questions before the schema is compiled."""
        shortlists: dict[str, list[int]] = {}
        narrowed: dict[str, object] = {}
        state_text = render_state(request.state)

        for qid, question in request.questions.items():
            if not isinstance(question, ChoiceQuestion):
                continue
            planned = self.compiler.plan_question(qid, question)
            if not planned.needs_retrieval:
                continue
            keep = sorted(
                self.shortlister.shortlist(question, state_text, self.config.shortlist_size)
            )
            if len(keep) >= len(question.options):
                continue
            shortlists[qid] = keep
            narrowed[qid] = question.model_copy(
                update={"options": [question.options[i] for i in keep]}
            )

        if not narrowed:
            return request, shortlists
        questions = dict(request.questions)
        questions.update(narrowed)  # type: ignore[arg-type]
        return request.model_copy(update={"questions": questions}), shortlists

    def _answer_one(
        self,
        qid: str,
        question: ChoiceQuestion | ScoreQuestion | NoulQuestion,
        output: QuestionOutput,
        shortlist: list[int] | None,
        request: DecisionRequest,
    ) -> Answer:
        answer = self._distribution(qid, question, output, shortlist, request)
        if not request.options.include_evidence:
            return answer
        return answer.model_copy(update=self._evidence(output, request))

    def _evidence(self, output: QuestionOutput, request: DecisionRequest) -> dict:
        """The answer's evidence fields, merged from the backend's token scores.

        A backend that cannot attribute says nothing, and the answer says so:
        ``unavailable`` with an empty list, rather than an empty list a caller
        would read as "nothing in the state mattered".
        """
        if output.evidence is None:
            return {"evidence": [], "evidence_method": "unavailable"}
        text = render_state(request.state)
        spans = merge_spans(text, output.evidence, self.config.evidence_threshold)
        return {
            "evidence": [
                EvidenceSpan(start=start, end=end, text=text[start:end], score=score)
                for start, end, score in spans
            ],
            "evidence_method": output.evidence_method or "unavailable",
        }

    def _distribution(
        self,
        qid: str,
        question: ChoiceQuestion | ScoreQuestion | NoulQuestion,
        output: QuestionOutput,
        shortlist: list[int] | None,
        request: DecisionRequest,
    ) -> Answer:
        opts = request.options
        domain = self.config.domain

        if isinstance(question, NoulQuestion):
            logit = output.logits[0]
            # A Noul is one probability, so the isotonic map is applied to it
            # directly rather than through a distribution. `apply` would see a
            # two-element vector and rescale the complement, which is the same
            # arithmetic spelled less clearly.
            return NoulAnswer(
                probability=self.isotonic.confidence(
                    "noul", self.scaler.apply_binary(logit, domain)
                ),
                raw_probability=_sigmoid_raw(logit) if opts.include_raw_probabilities else None,
            )

        primitive = "choice" if isinstance(question, ChoiceQuestion) else "score"
        scaled = self.isotonic.apply(
            primitive, self.scaler.apply(list(output.logits), primitive, domain)
        )
        names = question.names
        probs = _expand(scaled, shortlist, len(names))
        raw = (
            _expand(softmax(list(output.logits)), shortlist, len(names))
            if opts.include_raw_probabilities
            else None
        )
        # Confidence and the conformal set are always computed on the FULL
        # distribution, before any trimming: a confidence derived from a
        # truncated vector would read high simply because the tail was dropped.
        confidence = choice_confidence(probs, self.config.choice_confidence)
        prediction_set, coverage_target = self._conformal(probs, names, opts.conformal_profile)
        top = max(range(len(probs)), key=lambda i: probs[i])

        by_name, truncated, mass = _present(names, probs, opts.top_probabilities, keep=top)
        raw_by_name = (
            _present(names, raw, opts.top_probabilities, keep=top)[0] if raw is not None else None
        )

        if isinstance(question, ChoiceQuestion):
            return ChoiceAnswer(
                selected=names[top],
                probabilities=by_name,
                confidence=confidence,
                raw_probabilities=raw_by_name,
                prediction_set=prediction_set,
                coverage_target=coverage_target,
                shortlisted_from=len(names) if shortlist is not None else None,
                truncated=truncated,
                probability_mass=mass if truncated else None,
            )

        anchors = question.anchors
        return ScoreAnswer(
            # The score is the expectation over the whole distribution, never
            # over what survived truncation.
            score=expectation(probs, anchors),
            probabilities=by_name,
            confidence=score_confidence(probs, anchors, self.config.score_confidence),
            raw_probabilities=raw_by_name,
            prediction_set=prediction_set,
            coverage_target=coverage_target,
            truncated=truncated,
            probability_mass=mass if truncated else None,
        )

    def _conformal(
        self, probs: list[float], names: list[str], profile: str | None
    ) -> tuple[list[str] | None, float | None]:
        if profile is None:
            return None, None
        predictor = self.conformal.get(profile)
        if predictor is None:
            raise KeyError(
                f"no conformal profile named {profile!r}; fitted profiles are "
                f"{sorted(self.conformal) or 'none'}"
            )
        result = predictor.predict(probs, names)
        return list(result.labels), predictor.target_coverage


def _present(
    names: list[str], probs: list[float], top_k: int | None, keep: int
) -> tuple[dict[str, float], bool, float]:
    """The probabilities to put on the wire.

    Returns the real probabilities, never renormalised: a truncated map that
    summed to 1 would misrepresent how much of the distribution the caller is
    seeing. ``probability_mass`` reports the coverage instead.
    """
    if top_k is None or top_k >= len(names):
        return dict(zip(names, probs, strict=True)), False, 1.0

    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
    if keep not in order:
        # The selected option is always present, even if the caller asked for
        # fewer entries than its rank.
        order[-1] = keep
    order.sort()
    kept = {names[i]: probs[i] for i in order}
    return kept, True, sum(kept.values())


def _expand(scaled: list[float], shortlist: list[int] | None, full: int) -> list[float]:
    """Put shortlist probabilities back in the caller's declared positions.

    Options the prefilter dropped get probability 0 rather than disappearing:
    the caller declared them, so the answer has to mention them.
    """
    if shortlist is None:
        return scaled
    probs = [0.0] * full
    for value, index in zip(scaled, shortlist, strict=True):
        probs[index] = value
    return probs


def _sigmoid_raw(logit: float) -> float:
    from .numeric import sigmoid

    return sigmoid(logit)
