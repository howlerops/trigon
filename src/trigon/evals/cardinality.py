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
    #: How many of the option's four slots the query left unstated.
    drop_slots: int = 0

    @property
    def passed(self) -> bool:
        return self.recall >= self.limit

    def __str__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict} recall@{self.shortlist} on {self.options:>6} options, "
            f"drop={self.drop_slots}: {self.recall:.4f} "
            f"(limit {self.limit:.2f}, n={self.queries})"
        )


def _vocabulary() -> list[str]:
    """Every name the base vocabulary can produce, in a deterministic order."""
    return [
        f"{domain}_{obj}_{action}_{qualifier}"
        for domain in _DOMAINS
        for obj in _OBJECTS
        for action in _ACTIONS
        for qualifier in _QUALIFIERS
    ]


def _names(n_options: int, rng: random.Random) -> list[str]:
    """``n_options`` distinct option names.

    Enumerate-and-shuffle rather than sample-until-distinct. The base
    vocabulary yields 8 x 7 x 7 x 7 = 2,744 combinations, and rejection
    sampling past that point does not terminate -- which it duly did not, the
    first time this ran at 10,000 options. Beyond the vocabulary the names get
    a numeric discriminator, which is also what real taxonomies do once they
    outgrow their naming scheme.
    """
    base = _vocabulary()
    rng.shuffle(base)
    if n_options <= len(base):
        return base[:n_options]

    names = list(base)
    generation = 2
    while len(names) < n_options:
        for stem in base:
            if len(names) == n_options:
                break
            names.append(f"{stem}_{generation}")
        generation += 1
    return names


def build_cardinality_probe(
    n_options: int,
    n_queries: int = 200,
    seed: int = 0,
    *,
    with_criteria: bool = True,
    drop_slots: int = 0,
) -> tuple[ChoiceQuestion, list[str], list[int]]:
    """A question with ``n_options`` confusable options, plus labelled queries.

    Each query paraphrases its true option -- same content words, different
    order and wording -- so a prefilter has to do better than exact matching
    while the near-neighbours it must beat share most of those words.

    ``drop_slots`` is the difficulty dial, and it is the reason this probe is
    worth running. At 0 the query names all four slots of its option, so the
    true option is a 4/4 match and every distractor is at best 3/4: recall@256
    reads 1.0000 at every size, which measures nothing. Real tickets do not
    name every field. At 1 or 2 the query underdetermines the answer, dozens or
    hundreds of options match equally well, and recall becomes a measurement
    rather than a formality.
    """
    if n_options < 2:
        raise ValueError(f"need at least 2 options, got {n_options}")
    rng = random.Random(f"cardinality:{n_options}:{seed}")
    names = _names(n_options, rng)

    options = []
    for name in names:
        domain, obj, action, qualifier = _parts(name)
        criteria = (
            f"the customer's {obj.replace('_', ' ')} was {action} "
            f"{qualifier.replace('_', ' ')}, relating to their {domain}"
        )
        options.append({"name": name, "criteria": criteria} if with_criteria else {"name": name})

    question = ChoiceQuestion(
        instructions="Route this ticket to the most specific matching intent.",
        options=options,
    )

    if not 0 <= drop_slots <= 3:
        raise ValueError(f"drop_slots must be 0-3, got {drop_slots}")

    queries, labels = [], []
    for _ in range(n_queries):
        index = rng.randrange(n_options)
        domain, obj, action, qualifier = _parts(names[index])
        slots = {
            "domain": f"I need help with my {domain}",
            "obj": f"it is about my {obj.replace('_', ' ')}",
            "action": f"it was {action}",
            "qualifier": f"this happened {qualifier.replace('_', ' ')}",
        }
        # Always keep at least one slot, and drop at random rather than always
        # dropping the same field -- a fixed drop would let a prefilter learn
        # the hole rather than handle underdetermined queries.
        for key in rng.sample(sorted(slots), drop_slots):
            del slots[key]
        queries.append("Hi, " + ", ".join(slots[k] for k in sorted(slots)) + ".")
        labels.append(index)
    return question, queries, labels


def _parts(name: str) -> tuple[str, str, str, str]:
    """Split an option name back into its four slots.

    Names past the base vocabulary carry a numeric discriminator; the query
    paraphrases the concept, not the discriminator, so those near-duplicates
    are exactly what the prefilter has to separate.
    """
    domain, obj, action, qualifier = name.split("_", 3)
    if qualifier[-1].isdigit():
        qualifier = qualifier.rsplit("_", 1)[0]
    return domain, obj, action, qualifier


def run_cardinality_gate(
    shortlister: Shortlister | None = None,
    option_counts: Sequence[int] = (256, 1_024, 4_096, 10_000),
    shortlist: int = RETRIEVAL_SHORTLIST_SIZE,
    n_queries: int = 200,
    seed: int = 0,
    limit: float = RETRIEVAL_MIN_RECALL_AT_K,
    drop_slots: Sequence[int] | int = (0, 1, 2),
    on_result: Callable[[CardinalityResult], None] | None = None,
) -> list[CardinalityResult]:
    """Measure recall@shortlist across the regime the trigger fires in.

    The default sweeps query difficulty as well as option count, because a
    recall number without a difficulty level is not a measurement. At
    ``drop_slots=0`` this prefilter returns 1.0000 at every size, which says
    only that an exactly-specified query is easy to match. The gate earns its
    keep at 2.

    Pass ``on_result`` to stream results as each cell completes.
    """
    shortlister = shortlister or LexicalShortlister()
    drops = (drop_slots,) if isinstance(drop_slots, int) else tuple(drop_slots)
    results = []
    for drop, n_options in ((d, n) for d in drops for n in option_counts):
        question, queries, labels = build_cardinality_probe(
            n_options, n_queries, seed, drop_slots=drop
        )
        shortlists = [shortlister.shortlist(question, q, k=shortlist) for q in queries]
        result = CardinalityResult(
            options=n_options,
            shortlist=shortlist,
            recall=recall_at_k(shortlists, labels),
            limit=limit,
            queries=len(labels),
            drop_slots=drop,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
