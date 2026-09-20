"""The recall gate on the large-cardinality prefilter.

Decision D1 says the retrieval stage engages above 1,024 options or the
per-question token budget, with a 256-option shortlist, and names its own
falsifier: recall@256 of the true option must stay at or above 0.99, because
everything below the prefilter's recall is accuracy no model quality can
recover. This module runs that falsifier.

Why the option sets are generated rather than taken from a public corpus: the
licence audit in ``docs/data.md`` found that no permissively-licensed corpus
covers the regime the trigger actually fires in. UFET -- the ~10k-type set the
build plan named for this -- has no stated licence and its distant-supervision
half derives from LDC-licensed Gigaword, so it cannot be redistributed or
trained on. The largest green corpus is CLINC150 at 151 classes, below the
trigger. A generated set is therefore not a shortcut, it is the only option
that ships; and for a recall measurement it is arguably the better one, because
difficulty is a dial rather than whatever the corpus happened to contain.

The dial that matters is **distractor similarity**. An option set of unrelated
names makes any prefilter look good. These sets are built so that every true
option has near-neighbours sharing most of its words, which is what a real
intent taxonomy looks like at 10k classes.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..limits import RETRIEVAL_MIN_RECALL_AT_K, RETRIEVAL_SHORTLIST_SIZE
from ..retrieval import LexicalShortlister, Shortlister, recall_at_k
from ..types import ChoiceQuestion

__all__ = ["CardinalityResult", "build_cardinality_probe", "run_cardinality_gate"]

_DOMAINS = ["card", "account", "transfer", "payment", "refund", "device", "identity", "limit"]
_ACTIONS = ["declined", "pending", "reversed", "blocked", "expired", "duplicated", "delayed"]
_OBJECTS = ["charge", "transaction", "statement", "top_up", "direct_debit", "wire", "cheque"]
_QUALIFIERS = ["abroad", "online", "at_atm", "in_store", "recurring", "contactless", "by_phone"]


@dataclass(frozen=True)
class CardinalityResult:
    """One measurement of the prefilter at one option count."""

    options: int
    shortlist: int
    recall: float
    limit: float
    queries: int

    @property
    def passed(self) -> bool:
        return self.recall >= self.limit

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} recall@{self.shortlist} on {self.options} options: "
            f"{self.recall:.4f} (limit {self.limit:.2f}, n={self.queries})"
        )


def _option_name(rng: random.Random) -> str:
    return "_".join(
        [
            rng.choice(_DOMAINS),
            rng.choice(_OBJECTS),
            rng.choice(_ACTIONS),
            rng.choice(_QUALIFIERS),
        ]
    )


def build_cardinality_probe(
    n_options: int, n_queries: int = 200, seed: int = 0, *, with_criteria: bool = True
) -> tuple[ChoiceQuestion, list[str], list[int]]:
    """A question with ``n_options`` confusable options, plus labelled queries.

    Each query paraphrases its true option -- same content words, different
    order and wording -- so a prefilter has to do better than exact matching
    while the near-neighbours it must beat share most of those words.
    """
    rng = random.Random(f"cardinality:{n_options}:{seed}")
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n_options:
        name = _option_name(rng)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)

    options = []
    for name in names:
        domain, obj, action, qualifier = name.split("_", 3)
        criteria = (
            f"the customer's {obj.replace('_', ' ')} was {action} "
            f"{qualifier.replace('_', ' ')}, relating to their {domain}"
        )
        options.append({"name": name, "criteria": criteria} if with_criteria else {"name": name})

    question = ChoiceQuestion(
        instructions="Route this ticket to the most specific matching intent.",
        options=options,
    )

    queries, labels = [], []
    for _ in range(n_queries):
        index = rng.randrange(n_options)
        domain, obj, action, qualifier = names[index].split("_", 3)
        queries.append(
            f"Hi, my {obj.replace('_', ' ')} was {action} {qualifier.replace('_', ' ')} "
            f"and I need help with my {domain}."
        )
        labels.append(index)
    return question, queries, labels


def run_cardinality_gate(
    shortlister: Shortlister | None = None,
    option_counts: Sequence[int] = (256, 1_024, 4_096, 10_000),
    shortlist: int = RETRIEVAL_SHORTLIST_SIZE,
    n_queries: int = 200,
    seed: int = 0,
    limit: float = RETRIEVAL_MIN_RECALL_AT_K,
    on_result: Callable[[CardinalityResult], None] | None = None,
) -> list[CardinalityResult]:
    """Measure recall@shortlist across the regime the trigger fires in.

    Minutes, not seconds, at the top of the range: the 10,000-option pass
    scores thousands of BM25 candidates per query in Python, which is the
    honest cost of measuring a prefilter at the scale it exists for. Pass
    ``on_result`` to stream results as each option count completes rather than
    waiting for the whole sweep.
    """
    shortlister = shortlister or LexicalShortlister()
    results = []
    for n_options in option_counts:
        question, queries, labels = build_cardinality_probe(n_options, n_queries, seed)
        shortlists = [shortlister.shortlist(question, q, k=shortlist) for q in queries]
        result = CardinalityResult(
            options=n_options,
            shortlist=shortlist,
            recall=recall_at_k(shortlists, labels),
            limit=limit,
            queries=len(labels),
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
