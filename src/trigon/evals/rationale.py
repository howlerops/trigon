"""Plausibility: does the model highlight what a person highlighted?

The evidence a model returns can be judged two ways. *Faithfulness* asks
whether the highlighted spans are what the model actually used; *plausibility*
asks whether they are what a person would have marked. This module measures
the second, with the two metrics the rationale literature standardised on
(DeYoung et al., 2020, "ERASER"; Mathew et al., 2021, "HateXplain"):

* **token F1** -- precision and recall of highlighted words against the human
  rationale's words, F1 per case, averaged over cases;
* **IOU F1** -- a predicted span *matches* a human span when their overlap is
  at least half their union, in words; precision and recall over spans, F1
  per case, averaged over cases.

Both are scored on **words of the state** -- runs of non-whitespace -- not on
model tokens, so a byte-level BPE and a word-level floor are measured on the
same units. A word counts as highlighted when any evidence span touches it.

**A plausibility number means nothing without its floors**, the same rule as
every benchmark here (`CLAUDE.md`, "When adding a benchmark"). Three are
computed beside every model and printed in the same table:

* ``lexical floor`` -- `LexicalBackend`'s own evidence, the words of the state
  the question's text contains. The floor every suite runs.
* ``rationale lexicon`` -- the words people highlight most often, learned from
  the training split: every word highlighted in at least half its training
  occurrences. On hate speech this is, very nearly, a slur list, and it is
  the floor a highlighter has to clear before its plausibility says anything
  about reading rather than about vocabulary.
* ``every word`` -- highlight everything. Recall 1 by construction; its token
  F1 is set by how much of a post people highlight, and a model below it is
  worse than not choosing at all.

Pure Python, like the calibration metrics, so it imports without torch.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..engine import Engine
from ..schema import render_state
from .harness import Case

__all__ = [
    "IOU_MATCH",
    "PlausibilityResult",
    "RationaleLexicon",
    "highlighted_words",
    "iou_f1",
    "plausibility",
    "rationale_cases",
    "render_plausibility",
    "run_rationale_suite",
    "token_prf",
    "words",
]

#: A predicted span matches a human one at this intersection-over-union,
#: in words. ERASER's value, kept so the number means what it means there.
IOU_MATCH = 0.5

_WORD = re.compile(r"\S+")
_STRIP = re.compile(r"^\W+|\W+$")


def words(text: str) -> list[tuple[int, int]]:
    """Character spans of the words of ``text``: maximal runs of non-whitespace."""
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def highlighted_words(
    units: Sequence[tuple[int, int]], spans: Iterable[tuple[int, int]]
) -> list[bool]:
    """Which words any span touches."""
    spans = sorted(spans)
    return [any(s < end and start < e for s, e in spans) for start, end in units]


def token_prf(predicted: Sequence[bool], gold: Sequence[bool]) -> tuple[float, float, float]:
    """Precision, recall and F1 of highlighted words. 0 where undefined."""
    hits = sum(1 for p, g in zip(predicted, gold, strict=True) if p and g)
    chosen, wanted = sum(predicted), sum(gold)
    precision = hits / chosen if chosen else 0.0
    recall = hits / wanted if wanted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union else 0.0


def iou_f1(predicted: Sequence[bool], gold: Sequence[bool], match: float = IOU_MATCH) -> float:
    """ERASER's IOU F1 for one case, over contiguous runs of highlighted words."""
    pred_runs, gold_runs = _runs(predicted), _runs(gold)
    if not pred_runs or not gold_runs:
        return 0.0
    precision = sum(any(_iou(p, g) >= match for g in gold_runs) for p in pred_runs) / len(pred_runs)
    recall = sum(any(_iou(p, g) >= match for p in pred_runs) for g in gold_runs) / len(gold_runs)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@dataclass(frozen=True)
class PlausibilityResult:
    """One highlighter's plausibility over a set of human rationales."""

    name: str
    method: str
    n: int
    token_f1: float
    token_precision: float
    token_recall: float
    iou_f1: float
    #: Share of words highlighted, beside the human share, so a high recall
    #: bought by highlighting everything is visible in the same row.
    highlighted_fraction: float
    gold_fraction: float
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "n": self.n,
            "token_f1": self.token_f1,
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "iou_f1": self.iou_f1,
            "highlighted_fraction": self.highlighted_fraction,
            "gold_fraction": self.gold_fraction,
            **self.extra,
        }


def plausibility(
    name: str, method: str, pairs: Sequence[tuple[Sequence[bool], Sequence[bool]]]
) -> PlausibilityResult:
    """Macro-averaged plausibility over ``(predicted, gold)`` word masks."""
    if not pairs:
        raise ValueError("no rationales to score against")
    rows = [token_prf(p, g) for p, g in pairs]
    return PlausibilityResult(
        name=name,
        method=method,
        n=len(pairs),
        token_precision=statistics.fmean(r[0] for r in rows),
        token_recall=statistics.fmean(r[1] for r in rows),
        token_f1=statistics.fmean(r[2] for r in rows),
        iou_f1=statistics.fmean(iou_f1(p, g) for p, g in pairs),
        highlighted_fraction=statistics.fmean(sum(p) / len(p) for p, _ in pairs),
        gold_fraction=statistics.fmean(sum(g) / len(g) for _, g in pairs),
    )


def rationale_cases(cases: Iterable[Case]) -> list[tuple[Case, str]]:
    """``(case, question id)`` for every question whose human rationale is non-empty.

    An empty rationale -- a person looked and marked nothing -- has no words
    to recover, so precision and recall are both undefined and the case is
    left out rather than scored 0 for everyone alike.
    """
    out = []
    for case in cases:
        for qid, expected in case.expected.items():
            if expected.rationale:
                out.append((case, qid))
    return out


def _normal(word: str) -> str:
    return _STRIP.sub("", word.lower())


class RationaleLexicon:
    """The words people highlight, learned from a training split.

    A word is in the lexicon when it was highlighted in at least ``threshold``
    of its training occurrences and occurred at least ``min_count`` times.
    Lowercased, surrounding punctuation stripped. Deliberately nothing
    smarter: it is a floor, and its job is to show how much of a plausibility
    score is vocabulary.
    """

    def __init__(self, *, threshold: float = 0.5, min_count: int = 3) -> None:
        self.threshold = threshold
        self.min_count = min_count
        self.vocabulary: frozenset[str] = frozenset()
        #: Share of its training occurrences each word was highlighted in, for
        #: words seen at least ``min_count`` times. The vocabulary is the words
        #: at or above ``threshold``; the faithfulness suite ranks by the rate.
        self.rates: dict[str, float] = {}

    def fit(self, cases: Iterable[Case]) -> RationaleLexicon:
        seen: dict[str, int] = {}
        marked: dict[str, int] = {}
        for case, qid in rationale_cases(cases):
            text = render_state(case.request.state)
            units = words(text)
            gold = highlighted_words(units, case.expected[qid].rationale or ())
            for (start, end), hit in zip(units, gold, strict=True):
                word = _normal(text[start:end])
                if not word:
                    continue
                seen[word] = seen.get(word, 0) + 1
                marked[word] = marked.get(word, 0) + int(hit)
        self.rates = {w: marked.get(w, 0) / n for w, n in seen.items() if n >= self.min_count}
        self.vocabulary = frozenset(
            w
            for w, n in seen.items()
            if n >= self.min_count and marked.get(w, 0) >= self.threshold * n
        )
        return self

    def rate(self, word: str) -> float:
        """The training highlight rate of ``word``, normalised; 0 if unseen or rare."""
        return self.rates.get(_normal(word), 0.0)

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(s, e) for s, e in words(text) if _normal(text[s:e]) in self.vocabulary]


def _with_evidence(case: Case):
    return case.request.model_copy(
        update={"options": case.request.options.model_copy(update={"include_evidence": True})}
    )


def score_engine(name: str, engine: Engine, cases: Sequence[Case]) -> PlausibilityResult:
    """Ask ``engine`` for evidence on every case and score it against the people."""
    pairs, methods = [], set()
    for case, qid in rationale_cases(cases):
        response = engine.answer(_with_evidence(case))
        answer = response.answers[qid]
        methods.add(answer.evidence_method or "unavailable")
        text = render_state(case.request.state)
        units = words(text)
        predicted = highlighted_words(units, [(s.start, s.end) for s in answer.evidence or ()])
        pairs.append((predicted, highlighted_words(units, case.expected[qid].rationale or ())))
    return plausibility(name, "+".join(sorted(methods)), pairs)


def score_spans(name: str, method: str, cases: Sequence[Case], spans_of) -> PlausibilityResult:
    """Score a highlighter that is a function from state text to spans."""
    pairs = []
    for case, qid in rationale_cases(cases):
        text = render_state(case.request.state)
        units = words(text)
        pairs.append(
            (
                highlighted_words(units, spans_of(text)),
                highlighted_words(units, case.expected[qid].rationale or ()),
            )
        )
    return plausibility(name, method, pairs)


def run_rationale_suite(
    engines: dict[str, Engine], evaluation: Sequence[Case], training: Sequence[Case]
) -> list[PlausibilityResult]:
    """Every engine given, then the three floors, on the same human rationales.

    ``training`` fits the rationale lexicon and nothing else; it must be
    disjoint from ``evaluation`` or the lexicon floor is an oracle.
    """
    from ..backends.lexical import LexicalBackend

    rows = [score_engine(name, engine, evaluation) for name, engine in engines.items()]
    rows.append(score_engine("lexical floor", Engine(LexicalBackend()), evaluation))
    lexicon = RationaleLexicon().fit(training)
    rows.append(score_spans("rationale lexicon", "fitted word list", evaluation, lexicon.spans))
    rows.append(score_spans("every word", "all words", evaluation, words))
    return rows


def render_plausibility(rows: Sequence[PlausibilityResult]) -> str:
    """The plausibility table, model rows and floors together."""
    lines = [
        "| Highlighter | Method | Rationales | Token F1 | Precision | Recall | IOU F1 "
        "| Highlighted | Human |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | `{r.method}` | {r.n:,} | {r.token_f1:.4f} | "
            f"{r.token_precision:.4f} | {r.token_recall:.4f} | {r.iou_f1:.4f} | "
            f"{r.highlighted_fraction:.3f} | {r.gold_fraction:.3f} |"
        )
    return "\n".join(lines)
