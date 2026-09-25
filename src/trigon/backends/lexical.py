"""The lexical floor: a dependency-free baseline that is not a model.

Its job is to be the bottom of every Pareto plot and to keep the whole stack --
gateway, evals, load tests -- runnable with no weights and no GPU. It
scores each option by weighted token overlap between the option's name and
criteria and the state, which puts it meaningfully above chance on keyword-ish
tasks like intent routing and at chance on anything requiring inference. That
is the point: a learned backend that cannot beat this on a suite has not
learned that suite.

It is deterministic given (state, schema), so it also serves as the fixture
backend for tests that care about plumbing rather than accuracy.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter

from ..schema import CompiledRequest, render_state
from ..text import STOPWORDS, stem
from ..text import tokenize as _tokenize
from ..types import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from .base import BackendOutput, QuestionOutput

__all__ = ["LexicalBackend"]

_WORD = re.compile(r"[A-Za-z0-9]+")


class LexicalBackend:
    """Token-overlap scoring. Honest about being a floor, not a model."""

    def __init__(self, *, temperature: float = 2.0, seed: int = 0) -> None:
        # Higher temperature than 1.0 because raw overlap counts are spiky and
        # would otherwise produce absurdly confident distributions.
        self.temperature = temperature
        self.seed = seed

    @property
    def model_version(self) -> str:
        return "lexical-floor-0.1.0"

    def infer(self, compiled: CompiledRequest, request: SystemOneRequest) -> BackendOutput:
        started = time.perf_counter()
        state_text = render_state(request.state)
        state_counts = Counter(_tokenize(state_text))
        outputs: dict[str, QuestionOutput] = {}
        want_evidence = request.options.include_evidence

        for compiled_q in compiled.schema.questions:
            question = request.questions[compiled_q.question_id]
            if isinstance(question, ChoiceQuestion):
                members = [f"{o.name} {o.criteria or ''}" for o in question.options]
            elif isinstance(question, ScoreQuestion):
                members = [f"{lv.name} {lv.criteria or ''}" for lv in question.levels]
            elif isinstance(question, NoulQuestion):
                members = [question.instructions]
            else:  # pragma: no cover - the union is closed
                raise TypeError(f"unsupported question {type(question).__name__}")

            scores = [self._score(state_counts, m) for m in members]
            if isinstance(question, NoulQuestion):
                # Centre the single logit so an unrelated state reads ~0.5
                # rather than drifting to a confident yes or no.
                logits = (scores[0] - self._pseudo_prior(compiled_q.schema_hash),)
            else:
                logits = tuple(s / self.temperature for s in scores)
            outputs[compiled_q.question_id] = QuestionOutput(
                question_id=compiled_q.question_id,
                kind=compiled_q.kind,
                logits=logits,
                evidence=(
                    _overlap_evidence(state_text, [question.instructions, *members])
                    if want_evidence
                    else None
                ),
                evidence_method="lexical_overlap" if want_evidence else None,
            )

        return BackendOutput(
            outputs=outputs,
            model_version=self.model_version,
            model_ms=(time.perf_counter() - started) * 1000.0,
            diagnostics={"state_tokens_seen": sum(state_counts.values())},
        )

    def _score(self, state_counts: Counter, member: str) -> float:
        """IDF-ish overlap: rare words in the option that appear in the state."""
        tokens = _tokenize(member)
        if not tokens:
            return 0.0
        total = 0.0
        for token in set(tokens):
            hits = state_counts.get(token, 0)
            if hits:
                # Longer tokens are likelier to be content words.
                total += math.log1p(hits) * math.log1p(len(token))
        return total / math.sqrt(len(set(tokens)))

    def _pseudo_prior(self, schema_hash: str) -> float:
        """A small, stable per-question offset in [0, 0.5).

        Without it every unmatched Noul returns exactly 0.5 and the calibration
        suite reports a suspiciously perfect-looking constant predictor.
        """
        digest = hashlib.sha256(f"{self.seed}:{schema_hash}".encode()).digest()
        return (digest[0] / 255.0) * 0.5

    def __repr__(self) -> str:
        return f"LexicalBackend(temperature={self.temperature})"


def _overlap_evidence(state_text: str, question_texts: list[str]) -> tuple:
    """The floor's evidence: every state word the question's own text contains.

    The same matching the floor answers with -- lowercased, stemmed, stopwords
    dropped -- so its evidence is exactly what its answer read. It is the
    bottom of the evidence table the same way the floor is the bottom of every
    Pareto plot: a model whose highlights are no better than the words it was
    asked about has not learned to highlight.
    """
    wanted = {token for text in question_texts for token in _tokenize(text)}
    out = []
    for match in _WORD.finditer(state_text):
        raw = match.group().lower()
        if len(raw) < 2 or raw in STOPWORDS:
            continue
        if stem(raw) in wanted:
            out.append((match.start(), match.end(), 1.0))
    return tuple(out)
