"""Is this a model of typed decisions, or a model of one training set?

The drop-in claim is that a caller who declares *their own* question gets a
usable, calibrated answer. Every certification before this suite was on the
option set a model was trained with, which cannot tell the two apart -- and
the certified Banking77 model turned out to be the second kind: 0.90 on its
77 declared options, 0.007 on 50 of the same ones (`docs/ledger.md`).

So the suite asks three things, on data no training mix reads:

* **Schema shift** -- a trained task under an option set it was never shown:
  a 50-option subset (the incumbent's per-question cap), shuffled, the true
  option always present. The gap to the declared set is how much of the
  model's accuracy was the set.
* **Held-out tasks** -- CLINC150 intents, which no mix trains on by decision,
  and BoolQ, which none may (share-alike), whose question is different on
  every case. Accuracy here is accuracy on a task the model has never seen.
* **Invariance** -- the same case, same options, a different order. Any
  answer that changes with the order was read off a position.

Every task is drawn once from a fixed seed, so the same cases can be sent to
any system that speaks the contract (`scripts/generality.py`), including an
incumbent -- which is the comparison the claim is actually about.

Pure Python: the case construction imports without `torch`, like every other
eval module.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..types import ChoiceQuestion
from .corpora import load
from .harness import Case
from .schema_shift import COMPAT_MAX_OPTIONS, Reshape, reshape_all

#: Cases per task. Accuracy at 1,000 carries a 95% interval of at most ±3.1
#: points. ECE at 1,000 is below `MIN_CALIBRATION_SAMPLES` and is only ever
#: reported beside its simulated noise floor.
DEFAULT_N = 1000


@dataclass(frozen=True)
class Task:
    """One slice of the suite: where its cases come from and how they are shown."""

    name: str
    corpus: str
    kind: str  # "declared" | "shift" | "heldout" | "invariance"
    reshape: Reshape | None = None
    #: Only systems without the incumbent's option cap can be asked this.
    over_compat_cap: bool = False
    note: str = ""


SHIFT = Reshape(min_options=COMPAT_MAX_OPTIONS, max_options=COMPAT_MAX_OPTIONS, shuffle=True)
HELDOUT_CHOICE = Reshape(min_options=COMPAT_MAX_OPTIONS, max_options=COMPAT_MAX_OPTIONS)
RENAMED = Reshape(
    min_options=COMPAT_MAX_OPTIONS, max_options=COMPAT_MAX_OPTIONS, shuffle=True, rename=1.0
)

TASKS: tuple[Task, ...] = (
    Task(
        "banking77/declared",
        "banking77",
        "declared",
        over_compat_cap=True,
        note="the 77 options it was trained on, in training order: the control",
    ),
    Task("banking77/shift", "banking77", "shift", SHIFT, note="50 of the same options, shuffled"),
    Task(
        "banking77/renamed",
        "banking77",
        "shift",
        RENAMED,
        note="50, shuffled, every name redrawn in another case style",
    ),
    Task("clinc150", "clinc150", "heldout", HELDOUT_CHOICE, note="150 intents, 50 per case"),
    Task("boolq", "boolq", "heldout", note="yes/no; a different question on every case"),
    Task(
        "banking77/order",
        "banking77",
        "invariance",
        SHIFT,
        note="banking77/shift again, in a different order",
    ),
)


def task(name: str) -> Task:
    for t in TASKS:
        if t.name == name:
            return t
    raise KeyError(f"unknown task {name!r}; the suite has {[t.name for t in TASKS]}")


def _draw(corpus: str, n: int, seed: int) -> list[Case]:
    cases = load(corpus, "test", purpose="eval")
    # Several corpora are ordered by label; a slice off the front is one class.
    random.Random(seed).shuffle(cases)
    return cases[:n]


def _reorder(case: Case, rng: random.Random) -> Case:
    """The same options in another order, and nothing else."""
    return Reshape(min_options=10**9, max_options=None, shuffle=True)(case, rng)


def build(
    t: Task, n: int = DEFAULT_N, seed: int = 20260930, *, loader: Callable[..., list[Case]] = _draw
) -> list[Case]:
    """The task's cases, identical on every call with the same arguments.

    The invariance task is its shift task re-ordered: the same case ids, the
    same option subset, so a pair can be joined by case id and compared.
    """
    cases = loader(t.corpus, n, seed)
    if t.reshape is not None:
        cases = reshape_all(cases, t.reshape, seed=seed + 1)
    if t.kind == "invariance":
        rng = random.Random(seed + 2)
        cases = [_reorder(case, rng) for case in cases]
    return cases


def selected_names(answers: Sequence[dict | None], cases: Sequence[Case]) -> list[str | None]:
    """The chosen option's name per case, for joining an invariance pair."""
    out: list[str | None] = []
    for answer, case in zip(answers, cases, strict=True):
        question = next(iter(case.request.questions.values()))
        if answer is None or not isinstance(question, ChoiceQuestion):
            out.append(None)
            continue
        probabilities = answer.get("probabilities") or {}
        out.append(max(probabilities, key=lambda k: probabilities[k]) if probabilities else None)
    return out
