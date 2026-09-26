"""Faithfulness: did the model use what it highlighted?

Plausibility (`trigon.evals.rationale`) asks whether a person would agree
with a highlight. It cannot say whether the model looked there: a highlighter
that returns a slur list is plausible on hate speech whatever the model read.
This module measures the other property, with ERASER's two metrics (DeYoung
et al., 2020), per case, for the label the model selected on the full state:

* **comprehensiveness** -- ``p(y | state) - p(y | state without the top
  words)``. Higher means the highlighted words carried the answer: take them
  away and it falls.
* **sufficiency** -- ``p(y | state) - p(y | only the top words)``. Lower means
  the highlighted words alone were enough to keep it.

**Binned by top-k% of words, and averaged over the bins** (ERASER's AOPC,
k in 1, 5, 10, 20 and 50%). A method that highlights a third of the post and
one that highlights a tenth are not comparable at their own thresholds; at a
fixed share of the post they are. So the thresholded spans are never used:
each method's *scores* rank the words, and each bin removes (or keeps) the
same number of words for every method. ``ceil(k * words)`` of them, at least
one, and never more than the post has.

**The unit is the word of the state, not the model token.** The backend's
evidence arrives as per-token ``(start, end, score)`` triples; a word scores
the highest of the tokens that touch it -- the rule `trigon.evidence` uses to
score a span -- and it is words that are ranked and deleted. Three reasons.
Plausibility is scored on words, so the two tables rank the same units.
The controls have no tokens: the rationale lexicon scores words. And deleting
a subword fragment leaves a string the tokenizer segments into different
tokens, so "remove token 7" is not an operation the re-encode can honour
anyway. Ties -- the positive part of a gradient is zero for most words -- are
broken by one seeded draw per case, shared by every method, so a tie is
resolved the same way for all of them and never by position in the post.

**"Remove" is a real deletion and a real re-ask.** The words are cut out of
the rendered state along with the whitespace after each (`delete_words`),
and the shorter string goes back through the engine as a new request:
tokenised, positioned, and prefix-cached exactly as a caller's would be. The
alternative ERASER also admits -- masking token embeddings in place -- keeps
positions the caller's string no longer has, and would measure a model that
never serves.

**Read the model's own distribution, not the calibrated one.** Every probe
asks for ``include_raw_probabilities`` and scores the softmax of the logits.
The isotonic map is piecewise constant, so a move inside one of its plateaus
reads as zero after calibration, and a temperature rescales every move by a
factor that differs per checkpoint. The question is what the model used, and
the calibrator does not use anything.

**A faithfulness number means nothing without its controls**, the rule every
benchmark here follows (`CLAUDE.md`, "When adding a benchmark"). Two are
scored the same way, on the same cases, through the same engine:

* ``random`` -- a seeded uniform score per word. Its comprehensiveness is
  what deleting k% of any words costs; a method that does not beat it has
  found nothing the model needed.
* ``rationale lexicon`` -- the words people highlight, ranked by their
  training highlight rate (`RationaleLexicon.rate`), so the lexicon's own
  words come first. A model that reads slurs is faithful to a slur list, and
  a highlighter has to beat this to show it found more than vocabulary.

Each row reports its difference from both controls, paired per case, with a
95% normal interval, because the absolute numbers are bounded by how
confident the model was to start with and mean little across checkpoints.

Pure Python, like the calibration metrics; the model is only ever reached
through an `Engine`.
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..engine import Engine
from ..schema import render_state
from ..types import DecisionRequest
from .harness import Case
from .rationale import RationaleLexicon, rationale_cases, words

__all__ = [
    "CONTROLS",
    "FAITHFULNESS_BINS",
    "FaithfulnessResult",
    "Prober",
    "Scorer",
    "aopc",
    "delete_words",
    "engine_scorer",
    "faithfulness",
    "lexicon_scorer",
    "random_scorer",
    "render_faithfulness",
    "run_faithfulness_suite",
    "top_words",
    "word_scores",
]

#: ERASER's bins: the share of the state's words removed or kept.
FAITHFULNESS_BINS = (0.01, 0.05, 0.10, 0.20, 0.50)

#: A scorer maps ``(case, question id, rendered state)`` to one score per word
#: of the state (`rationale.words`) and the name of the method that made them.
Scorer = Callable[[Case, str, str], tuple[list[float], str]]


def word_scores(text: str, token_scores: Sequence[tuple[int, int, float]]) -> list[float]:
    """Per-token ``(start, end, score)`` to one score per word of ``text``.

    A word scores the highest of the tokens that touch it, after each token is
    trimmed of whitespace -- a byte-level token's leading space touches the
    word before it and must not score it. A word no token touches scores 0.
    """
    units = words(text)
    out = [0.0] * len(units)
    starts = [s for s, _ in units]
    for start, end, score in token_scores:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            continue
        # Words are sorted and disjoint; walk from the last one starting before `end`.
        i = _bisect(starts, end) - 1
        while i >= 0 and units[i][1] > start:
            out[i] = max(out[i], float(score))
            i -= 1
    return out


def _bisect(values: Sequence[int], x: int) -> int:
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def bin_count(fraction: float, n: int) -> int:
    """How many of ``n`` words a bin of ``fraction`` takes: the ceiling, at least one.

    The ceiling is taken with a small tolerance, because ``0.1 * 30`` is
    ``3.0000000000000004`` in floating point and a bare ``ceil`` would make it 4.
    """
    if n <= 0:
        return 0
    return min(n, max(1, math.ceil(fraction * n - 1e-9)))


def top_words(scores: Sequence[float], count: int, tiebreak: Sequence[float]) -> set[int]:
    """Indices of the ``count`` highest scores; ties by ``tiebreak``, not position."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], tiebreak[i]))
    return set(order[:count])


def delete_words(text: str, units: Sequence[tuple[int, int]], drop: set[int]) -> str:
    """``text`` with the words at ``drop`` cut out, and the whitespace after each.

    Kept words keep the whitespace that followed them in the original, so a
    post joined by single spaces stays joined by single spaces, and a newline
    between two kept words survives. Text before the first word and after the
    last is kept as it was.
    """
    if not units:
        return text
    kept = [i for i in range(len(units)) if i not in drop]
    pieces = [text[: units[0][0]]]
    for position, i in enumerate(kept):
        start, end = units[i]
        pieces.append(text[start:end])
        if position + 1 < len(kept):
            following = units[i + 1][0] if i + 1 < len(units) else len(text)
            pieces.append(text[end:following])
    pieces.append(text[units[-1][1] :])
    return "".join(pieces)


def aopc(values: Sequence[float]) -> float:
    """ERASER's area over the perturbation curve: the mean over the bins."""
    return statistics.fmean(values)


class Prober:
    """Re-asks the engine for modified states, and remembers every answer.

    The cache is keyed on ``(case, question, text)``, so two methods that
    choose the same words at some k -- very common at 1% of a short post --
    pay for one forward, not two. Probes go through ``Engine.answer_many`` in
    batches of ``batch_size`` (1 is `Engine.answer`, one at a time), which is
    the serving path and is held to the one-at-a-time answer by
    `tests/test_batching.py`.
    """

    def __init__(self, engine: Engine, batch_size: int = 16) -> None:
        self.engine = engine
        self.batch_size = max(1, batch_size)
        self._cache: dict[tuple[str, str, str], tuple[float, ...]] = {}
        #: Forwards actually run and the wall clock they took.
        self.forwards = 0
        self.seconds = 0.0

    def _request(self, case: Case, text: str) -> DecisionRequest:
        options = case.request.options.model_copy(
            update={
                "include_evidence": False,
                "include_raw_probabilities": True,
                "top_probabilities": None,
                "conformal_profile": None,
            }
        )
        return case.request.model_copy(update={"state": text, "options": options})

    def probabilities(self, case: Case, qid: str, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """The model's raw distribution over ``qid``'s labels, for each text."""
        missing = list(dict.fromkeys(t for t in texts if (case.case_id, qid, t) not in self._cache))
        if missing:
            started = time.perf_counter()
            requests = [self._request(case, t) for t in missing]
            if self.batch_size == 1 or len(requests) == 1:
                responses = [self.engine.answer(r) for r in requests]
            else:
                responses = self.engine.answer_many(requests, batch_size=self.batch_size)
            self.seconds += time.perf_counter() - started
            self.forwards += len(requests)
            for text, response in zip(missing, responses, strict=True):
                self._cache[(case.case_id, qid, text)] = _raw(response.answers[qid])
        return [self._cache[(case.case_id, qid, t)] for t in texts]


def _raw(answer) -> tuple[float, ...]:
    if answer.type == "noul":
        p = float(answer.raw_probability)
        return (1.0 - p, p)
    return tuple(float(v) for v in answer.raw_probabilities.values())


@dataclass(frozen=True)
class FaithfulnessResult:
    """One highlighter's faithfulness over a set of cases."""

    name: str
    method: str
    n: int
    bins: tuple[float, ...]
    #: AOPC, macro-averaged over cases. Higher is better.
    comprehensiveness: float
    #: AOPC, macro-averaged over cases. Lower is better.
    sufficiency: float
    #: Mean per bin, in ``bins`` order.
    comprehensiveness_by_bin: tuple[float, ...]
    sufficiency_by_bin: tuple[float, ...]
    #: Mean ``p(y | state)``: comprehensiveness cannot exceed it.
    full_probability: float
    #: Per-case AOPCs, in case order, for pairing against a control.
    per_case: tuple[tuple[float, float], ...] = field(repr=False, compare=False, default=())
    #: Paired differences against the controls, set by `run_faithfulness_suite`.
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method": self.method,
            "n": self.n,
            "bins": list(self.bins),
            "comprehensiveness": self.comprehensiveness,
            "sufficiency": self.sufficiency,
            "comprehensiveness_by_bin": list(self.comprehensiveness_by_bin),
            "sufficiency_by_bin": list(self.sufficiency_by_bin),
            "full_probability": self.full_probability,
            **self.extra,
        }


def _tiebreak(case: Case, qid: str, n: int, seed: int) -> list[float]:
    rng = random.Random(f"faithfulness-tie:{seed}:{case.case_id}:{qid}")
    return [rng.random() for _ in range(n)]


def faithfulness(
    name: str,
    scorer: Scorer,
    pairs: Sequence[tuple[Case, str]],
    prober: Prober,
    *,
    bins: Sequence[float] = FAITHFULNESS_BINS,
    seed: int = 0,
) -> FaithfulnessResult:
    """Comprehensiveness and sufficiency of ``scorer`` over ``(case, question)`` pairs.

    The label scored is the one the model selects on the full state (the raw
    argmax, which is what the gradient methods attribute), fixed for that
    case, so every bin asks how far *that* answer moved.
    """
    bins = tuple(bins)
    per_case: list[tuple[float, float]] = []
    comp_bins: list[list[float]] = [[] for _ in bins]
    suff_bins: list[list[float]] = [[] for _ in bins]
    fulls: list[float] = []
    methods: set[str] = set()
    for case, qid in pairs:
        text = render_state(case.request.state)
        units = words(text)
        if not units:
            continue
        scores, method = scorer(case, qid, text)
        if len(scores) != len(units):
            raise ValueError(f"{name}: {len(scores)} scores for {len(units)} words")
        methods.add(method)
        tiebreak = _tiebreak(case, qid, len(units), seed)
        everything = set(range(len(units)))
        texts = [text]
        for fraction in bins:
            chosen = top_words(scores, bin_count(fraction, len(units)), tiebreak)
            texts.append(delete_words(text, units, chosen))
            texts.append(delete_words(text, units, everything - chosen))
        probs = prober.probabilities(case, qid, texts)
        full = probs[0]
        label = max(range(len(full)), key=lambda i: full[i])
        comps = [full[label] - probs[1 + 2 * b][label] for b in range(len(bins))]
        suffs = [full[label] - probs[2 + 2 * b][label] for b in range(len(bins))]
        for b in range(len(bins)):
            comp_bins[b].append(comps[b])
            suff_bins[b].append(suffs[b])
        fulls.append(full[label])
        per_case.append((aopc(comps), aopc(suffs)))
    if not per_case:
        raise ValueError("no cases with a non-empty state to score")
    return FaithfulnessResult(
        name=name,
        method="+".join(sorted(methods)),
        n=len(per_case),
        bins=bins,
        comprehensiveness=statistics.fmean(c for c, _ in per_case),
        sufficiency=statistics.fmean(s for _, s in per_case),
        comprehensiveness_by_bin=tuple(statistics.fmean(v) for v in comp_bins),
        sufficiency_by_bin=tuple(statistics.fmean(v) for v in suff_bins),
        full_probability=statistics.fmean(fulls),
        per_case=tuple(per_case),
    )


def engine_scorer(engine: Engine) -> Scorer:
    """The engine's backend's own token scores, before any threshold or merge.

    Compiled and inferred exactly as `Engine.answer` does, with evidence
    asked for, but the backend's per-token triples are read directly: the
    engine merges them into thresholded spans, and faithfulness ranks the
    scores. A backend that returns no evidence is an error here, not a row of
    zeros -- "unavailable" is not a method to score.
    """

    def score(case: Case, qid: str, text: str) -> tuple[list[float], str]:
        request = case.request.model_copy(
            update={"options": case.request.options.model_copy(update={"include_evidence": True})}
        )
        served, _ = engine._apply_retrieval(request)
        compiled = engine.compiler.compile_request(served)
        output = engine.backend.infer(compiled, served).outputs[qid]
        if output.evidence is None:
            raise ValueError(f"the backend returned no evidence for {case.case_id}/{qid}")
        return word_scores(text, output.evidence), output.evidence_method or "unknown"

    return score


def random_scorer(seed: int = 0) -> Scorer:
    """A seeded uniform score per word: the control a method must beat."""

    def score(case: Case, qid: str, text: str) -> tuple[list[float], str]:
        rng = random.Random(f"faithfulness-random:{seed}:{case.case_id}:{qid}")
        return [rng.random() for _ in words(text)], "random"

    return score


def lexicon_scorer(lexicon: RationaleLexicon) -> Scorer:
    """Each word's training highlight rate, so the lexicon's words rank first."""

    def score(case: Case, qid: str, text: str) -> tuple[list[float], str]:
        return [lexicon.rate(text[s:e]) for s, e in words(text)], "fitted word list"

    return score


#: The controls every row is paired against, by row name, and the short name
#: the paired fields and the table columns carry.
CONTROLS = {"random": "random", "rationale lexicon": "lexicon"}


def _paired(row: FaithfulnessResult, control: FaithfulnessResult, tag: str) -> dict[str, float]:
    """Mean per-case difference from ``control``, and its 95% normal half-width."""
    out: dict[str, float] = {}
    for key, index in (("comprehensiveness", 0), ("sufficiency", 1)):
        diffs = [a[index] - b[index] for a, b in zip(row.per_case, control.per_case, strict=True)]
        spread = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        out[f"{key}_minus_{tag}"] = statistics.fmean(diffs)
        out[f"{key}_minus_{tag}_ci95"] = 1.96 * spread / math.sqrt(len(diffs))
    return out


def run_faithfulness_suite(
    scorers: dict[str, Scorer],
    engine: Engine,
    evaluation: Sequence[Case],
    training: Sequence[Case],
    *,
    n: int = 500,
    bins: Sequence[float] = FAITHFULNESS_BINS,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[list[FaithfulnessResult], Prober]:
    """Every scorer given, then the two controls, on the same cases and probes.

    The cases are the first ``n`` evaluation questions with a human rationale
    (0 is all of them) -- the plausibility table's cases, so the two tables
    describe the same posts. ``training`` fits the rationale lexicon only, as
    in `rationale.run_rationale_suite`. Every row is paired per case against
    each control it is not.
    """
    pairs = rationale_cases(evaluation)
    if n:
        pairs = pairs[:n]
    prober = Prober(engine, batch_size=batch_size)
    lexicon = RationaleLexicon().fit(training)
    everything = {
        **scorers,
        "rationale lexicon": lexicon_scorer(lexicon),
        "random": random_scorer(seed),
    }
    rows = [
        faithfulness(name, scorer, pairs, prober, bins=bins, seed=seed)
        for name, scorer in everything.items()
    ]
    controls = {row.name: row for row in rows if row.name in CONTROLS}
    paired = []
    for row in rows:
        extra: dict[str, float] = {}
        for name, control in controls.items():
            if control is not row:
                extra.update(_paired(row, control, CONTROLS[name]))
        paired.append(dataclasses.replace(row, extra=extra))
    return paired, prober


def _versus(row: FaithfulnessResult, key: str, tag: str) -> str:
    if f"{key}_minus_{tag}" not in row.extra:
        return "—"
    return f"{row.extra[f'{key}_minus_{tag}']:+.4f} ± {row.extra[f'{key}_minus_{tag}_ci95']:.4f}"


def render_faithfulness(rows: Sequence[FaithfulnessResult]) -> str:
    """The faithfulness tables: AOPC paired against both controls, then per bin."""
    lines = [
        "| Highlighter | Method | Cases | Comprehensiveness ↑ | − random | − lexicon | "
        "Sufficiency ↓ | − random | − lexicon | p(selected), full |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | `{r.method}` | {r.n:,} | {r.comprehensiveness:.4f} | "
            f"{_versus(r, 'comprehensiveness', 'random')} | "
            f"{_versus(r, 'comprehensiveness', 'lexicon')} | {r.sufficiency:.4f} | "
            f"{_versus(r, 'sufficiency', 'random')} | {_versus(r, 'sufficiency', 'lexicon')} | "
            f"{r.full_probability:.4f} |"
        )
    if rows:
        bins = rows[0].bins
        header = " | ".join(f"{b:.0%}" for b in bins)
        lines += [
            "",
            f"| Highlighter, comprehensiveness / sufficiency | {header} |",
            "| --- |" + " ---: |" * len(bins),
        ]
        for r in rows:
            cells = " | ".join(
                f"{c:.3f} / {s:.3f}"
                for c, s in zip(r.comprehensiveness_by_bin, r.sufficiency_by_bin, strict=True)
            )
            lines.append(f"| {r.name} | {cells} |")
    return "\n".join(lines)
