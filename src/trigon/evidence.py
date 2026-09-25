"""Evidence: per-token scores in, character spans of the state out.

A backend scores the tokens of the state for one question; this module turns
those scores into what the contract returns -- spans of the caller's own
string, each with a score. It is pure Python for the reason the calibration
math is: the engine, the eval harness and the drift tests all import it, and
none of them should need torch to do it.

**One merge rule, here, for every backend.** A backend returns token-level
``(start, end, score)`` triples and never spans. If each backend merged its
own, the lexical floor and the trained model would be measured under two
different definitions of a span, and a plausibility number comparing them
would compare the rules as much as the models.

The rule: a token is kept when its score is at least the threshold; it is
trimmed of surrounding whitespace, so a byte-level token's leading space is
never evidence; a token that starts or ends inside a word is widened to the
whole word; kept tokens merge into one span when nothing but whitespace
separates them; a span's score is the highest of its tokens'.

**Why widen to the word.** A subword vocabulary splits "likes" into "li" and
"kes", and the first run of the spike on HateXplain returned exactly that --
``"kes"`` as evidence, which no reader can use and no person would highlight.
The offsets stay exact (they are still characters of the caller's string);
what changes is the unit, from the tokenizer's to the reader's, which is also
the unit the plausibility metrics score on. A run of word characters longer
than `MAX_WORD_CHARS` is left at token granularity rather than widened: in a
script written without spaces, "the word" is a sentence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = [
    "EVIDENCE_THRESHOLD",
    "MAX_WORD_CHARS",
    "TokenScore",
    "merge_spans",
    "token_labels",
    "trim",
    "widen",
]

#: A token is evidence when its score reaches this. For the supervised span
#: head the score is a probability that a person would highlight the token,
#: so 0.5 is the decision a proper scoring rule was trained to support. For
#: gradient x input the score is relative to the answer's strongest token, so
#: this keeps the tokens at least half as influential as that one -- a rule,
#: not a probability, and `docs/architecture.md` says so where it is used.
EVIDENCE_THRESHOLD = 0.5

#: The longest run of word characters a token is widened to. Longer runs --
#: unspaced scripts, base64, a URL's path -- keep the token's own span.
MAX_WORD_CHARS = 32

TokenScore = tuple[int, int, float]


def _wordish(char: str) -> bool:
    return char.isalnum() or char == "_"


def widen(text: str, start: int, end: int) -> tuple[int, int]:
    """``[start, end)`` widened to the word it starts or ends inside, if any."""
    if start >= end:
        return start, end
    left = start
    if _wordish(text[start]):
        while left > 0 and _wordish(text[left - 1]):
            left -= 1
    right = end
    if _wordish(text[end - 1]):
        while right < len(text) and _wordish(text[right]):
            right += 1
    if (left, right) != (start, end) and right - left > MAX_WORD_CHARS:
        return start, end
    return left, right


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """``[start, end)`` with leading and trailing whitespace removed.

    Returns an empty span (``start == end``) when the span is all whitespace.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def merge_spans(
    text: str, scores: Iterable[TokenScore], threshold: float = EVIDENCE_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Token scores to evidence spans: ``(start, end, score)``, in text order.

    ``scores`` are ``(start, end, score)`` per token, character offsets into
    ``text``. Tokens may overlap -- two byte-level tokens that split one
    character both claim it -- and that is handled by merging on overlap as
    well as on adjacency.
    """
    kept = []
    for start, end, score in scores:
        if score < threshold:
            continue
        start, end = trim(text, start, end)
        if end > start:
            start, end = widen(text, start, end)
            kept.append((start, end, score))
    kept.sort()

    spans: list[tuple[int, int, float]] = []
    for start, end, score in kept:
        if spans:
            last_start, last_end, last_score = spans[-1]
            gap = text[last_end:start] if start > last_end else ""
            if not gap.strip():
                spans[-1] = (last_start, max(last_end, end), max(last_score, score))
                continue
        spans.append((start, end, score))
    return [(s, e, min(1.0, max(0.0, float(v)))) for s, e, v in spans]


def token_labels(
    text: str, offsets: Sequence[tuple[int, int]], spans: Iterable[tuple[int, int]]
) -> list[int]:
    """1 for each token whose non-whitespace characters touch a gold span.

    What rationale supervision trains against: human highlights are character
    spans of the state and the model scores tokens, so this is the one bridge
    between them. A byte-level token's leading space is trimmed first, so the
    token after a highlighted word is not labelled for sharing its space.
    """
    gold = sorted(spans)
    labels = []
    for start, end in offsets:
        start, end = trim(text, start, end)
        hit = end > start and any(g_start < end and start < g_end for g_start, g_end in gold)
        labels.append(int(hit))
    return labels
