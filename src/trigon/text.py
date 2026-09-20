"""Shared tokenization for the lexical baseline and the retrieval prefilter.

Both are bag-of-words scorers over short strings, and both were getting this
subtly wrong in different ways before it was shared. The stemming is the part
that matters: without it "charged twice" does not match the criterion "a charge
or refund", and the floor returns a uniform distribution on exactly the tickets
it should be able to route. A floor that weak makes the bottom of every Pareto
plot uninformative.

Deliberately not a real stemmer. A suffix stripper is enough for option names
and criteria, costs no dependency, and behaves predictably -- which matters
more here than linguistic correctness, because this code is a measurement
instrument.
"""

from __future__ import annotations

import re

__all__ = ["STOPWORDS", "stem", "tokenize"]

_WORD = re.compile(r"[a-z0-9]+")

#: Words carried by almost every option name and criteria, which would
#: otherwise dominate an overlap score.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in is it its of on or
    that the this to was were will with you your""".split()
)

# Ordered longest-first so "-ing" wins over "-g" would-be matches; each entry is
# (suffix, minimum stem length to strip it).
_SUFFIXES: tuple[tuple[str, int], ...] = (
    ("ingly", 5),
    ("edly", 4),
    ("ing", 4),
    ("ies", 4),
    ("ied", 4),
    ("ed", 4),
    ("ly", 4),
    ("es", 4),
    ("s", 4),
)


def stem(word: str) -> str:
    """Normalise a word to a matching key.

    Four steps, in order: strip one inflectional suffix, drop a consonant-final
    "y", drop a trailing "e", then collapse a trailing doubled consonant. The
    last three exist so that forms the first step leaves apart still land on
    the same key -- "charge"/"charged" both become "charg", "ship"/"shipping"
    both become "ship", "delivery"/"deliveries" both become "deliver". The keys
    are not words, and do not need to be: they only need to agree.
    """
    for suffix, minimum in _SUFFIXES:
        if len(word) > minimum and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("y") and word[-2] not in "aeiou":
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase, split on word characters, drop stopwords, stem."""
    tokens = []
    for raw in _WORD.findall(text.lower()):
        if len(raw) < 2:
            continue
        if drop_stopwords and raw in STOPWORDS:
            continue
        tokens.append(stem(raw))
    return tokens
