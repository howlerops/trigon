"""The large-cardinality stage: prefilter, then let the model score a shortlist.

Above the budgets in ``trigon.limits`` an option set no longer fits the
context, so the request goes embed -> prefilter -> model rescoring. The
competitor caps at 255 options and pushes two-stage scoring into cookbook code;
doing it inside the serving path is a convenience edge, not a moat, so the code
here is deliberately small and the interesting part is the release gate:
``recall_at_k`` must stay above ``RETRIEVAL_MIN_RECALL_AT_K`` or the prefilter
is silently capping accuracy in a way no amount of model quality can recover.

``LexicalShortlister`` is the default because it needs no embedding service and
is a genuinely strong prefilter on name-like option sets. ``VectorShortlister``
takes any embedding callable, which is where a Qdrant-backed ANN index plugs in
without this module knowing about Qdrant.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .limits import RETRIEVAL_SHORTLIST_SIZE
from .text import tokenize
from .types import ChoiceQuestion

__all__ = [
    "LexicalShortlister",
    "Shortlister",
    "VectorShortlister",
    "recall_at_k",
]


@runtime_checkable
class Shortlister(Protocol):
    def shortlist(
        self, question: ChoiceQuestion, state: str, k: int = RETRIEVAL_SHORTLIST_SIZE
    ) -> list[int]:
        """Indices into ``question.options``, best first, at most ``k`` of them."""
        ...


class LexicalShortlister:
    """BM25 over option text, with the option set as the corpus.

    The corpus is the option set, and an option set is stable across every
    request that carries it -- the same fact that makes a compiled schema a
    cacheable KV prefix. So the tokenized documents and their statistics are
    built once per option set and reused, keyed on the option text.

    This is not a micro-optimisation. Rebuilding the index per call is
    O(options) tokenization on every request, which at the 10k-option scale the
    retrieval stage exists for costs far more than the model does, and it
    scales with QPS rather than with the number of distinct schemas.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, cache_size: int = 32) -> None:
        self.k1 = k1
        self.b = b
        self.cache_size = cache_size
        self._index: dict[str, _Corpus] = {}

    def _corpus(self, question: ChoiceQuestion) -> _Corpus:
        key = _option_set_key(question)
        cached = self._index.get(key)
        if cached is not None:
            return cached
        docs = [
            tokenize(o.name if o.criteria is None else f"{o.name} {o.criteria}")
            for o in question.options
        ]
        doc_freq: Counter[str] = Counter()
        for doc in docs:
            doc_freq.update(set(doc))
        corpus = _Corpus(
            docs=docs,
            counts=[Counter(doc) for doc in docs],
            doc_freq=doc_freq,
            avg_len=(sum(len(d) for d in docs) / len(docs)) if docs else 0.0,
        )
        if len(self._index) >= self.cache_size:
            self._index.clear()
        self._index[key] = corpus
        return corpus

    def shortlist(
        self, question: ChoiceQuestion, state: str, k: int = RETRIEVAL_SHORTLIST_SIZE
    ) -> list[int]:
        n = len(question.options)
        if k >= n:
            return list(range(n))

        corpus = self._corpus(question)
        query = set(tokenize(state))
        # Only documents containing a query term can score above zero, so the
        # scan runs over the postings rather than over every option.
        candidates: set[int] = set()
        for term in query:
            candidates.update(corpus.postings(term))

        scores = []
        for i in candidates:
            counts = corpus.counts[i]
            length = len(corpus.docs[i]) or 1
            score = 0.0
            for term in query:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = corpus.doc_freq[term]
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * length / (corpus.avg_len or 1.0))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append((score, i))

        scores.sort(key=lambda t: (-t[0], t[1]))
        keep = [i for _, i in scores[:k]]
        if len(keep) < k:
            # Pad deterministically, so the shortlist is always the size the
            # caller budgeted for rather than silently narrower.
            chosen = set(keep)
            for i in range(n):
                if len(keep) == k:
                    break
                if i not in chosen:
                    keep.append(i)
        return keep


@dataclass
class _Corpus:
    """A tokenized option set and its BM25 statistics."""

    docs: list[list[str]]
    counts: list[Counter]
    doc_freq: Counter
    avg_len: float
    _postings: dict[str, list[int]] | None = None

    def postings(self, term: str) -> list[int]:
        """Documents containing ``term``. Inverted index, built lazily once."""
        if self._postings is None:
            postings: dict[str, list[int]] = {}
            for i, doc in enumerate(self.docs):
                for token in set(doc):
                    postings.setdefault(token, []).append(i)
            self._postings = postings
        return self._postings.get(term, [])


def _option_set_key(question: ChoiceQuestion) -> str:
    """A stable key for an option set: its text, hashed."""
    digest = hashlib.blake2b(digest_size=16)
    for option in question.options:
        digest.update(option.name.encode("utf-8"))
        digest.update(b"\x00")
        if option.criteria:
            digest.update(option.criteria.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


class VectorShortlister:
    """Cosine similarity over precomputed option vectors.

    ``embed`` is any callable that maps texts to vectors -- a local encoder in
    tests, a Qdrant-backed service in production. Option vectors are cached by
    text, because an option set is stable across requests even when state is
    not; that cache is what an ANN index replaces at scale.
    """

    def __init__(self, embed: Callable[[Sequence[str]], Sequence[Sequence[float]]]) -> None:
        self.embed = embed
        self._cache: dict[str, tuple[float, ...]] = {}

    def shortlist(
        self, question: ChoiceQuestion, state: str, k: int = RETRIEVAL_SHORTLIST_SIZE
    ) -> list[int]:
        texts = [
            o.name if o.criteria is None else f"{o.name} {o.criteria}" for o in question.options
        ]
        if k >= len(texts):
            return list(range(len(texts)))

        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self.embed(missing), strict=True):
                self._cache[text] = tuple(vector)

        query = tuple(self.embed([state])[0])
        scored = [(_cosine(query, self._cache[t]), i) for i, t in enumerate(texts)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [i for _, i in scored[:k]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def recall_at_k(shortlists: Sequence[Sequence[int]], labels: Sequence[int]) -> float:
    """Release gate: how often the true option survived the prefilter.

    Everything below this ceiling is accuracy the model cannot recover, so this
    is checked before any model quality number is believed.
    """
    if len(shortlists) != len(labels):
        raise ValueError(f"{len(shortlists)} shortlists against {len(labels)} labels")
    if not labels:
        raise ValueError("no shortlists to score")
    return sum(1 for s, y in zip(shortlists, labels, strict=True) if y in set(s)) / len(labels)
