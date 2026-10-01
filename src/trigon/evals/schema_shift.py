"""Change a Choice's option set without changing what the right answer is.

A caller of a typed-decision API declares its own options on every request,
and nothing obliges two callers -- or two requests from one caller -- to
declare the same set, in the same order, under the same names. A model that
has only ever seen one option set cannot be told apart from one that reads the
options until that set changes, and then it can: the certified Banking77 model
answers 90% of the test split with its 77 training options and 0.7% (below the
2% chance rate) with 50 of the same options, the true one always among them.
`docs/ledger.md` has the run.

So the same transformation serves twice. As **augmentation** it is drawn fresh
per case per epoch, so no option set is ever seen twice and position carries no
signal. As an **evaluation** it is drawn once from a fixed seed, and the
difference between a model's accuracy on the declared set and on the reshaped
one is how much of its accuracy was the set rather than the state.

Only Choice is reshaped. A Score's levels are ordered low to high and that
order is part of the question; a Noul has no options. Both pass through.

Pure Python and `pydantic` only, like the rest of the eval harness, so the
gateway, CI and the trainer can all import it.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from ..schema.compiler import DOT_PRODUCT_CROSSOVER
from ..types import ChoiceQuestion, OptionSpec
from .harness import Case, Expectation

#: The incumbent's contract caps a Choice at 50 options (a 422 above it,
#: observed 2026-09-30 against a compatible service). A subset drawn for
#: evaluation never exceeds it, so the same cases can be sent to both sides.
COMPAT_MAX_OPTIONS = 50


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[\s_\-./]+", name) if w]


#: Surface forms of one option name. None of them changes what the option
#: means, which is the property that lets the label follow the option.
RENAMES: dict[str, Callable[[str], str]] = {
    "spaces": lambda n: " ".join(_words(n)),
    "snake": lambda n: "_".join(w.lower() for w in _words(n)),
    "kebab": lambda n: "-".join(w.lower() for w in _words(n)),
    "title": lambda n: " ".join(w[:1].upper() + w[1:] for w in _words(n)),
    "upper": lambda n: "_".join(w.upper() for w in _words(n)),
    "sentence": lambda n: (lambda s: s[:1].upper() + s[1:])(" ".join(_words(n)).lower()),
}


@dataclass(frozen=True)
class Reshape:
    """How to redraw a Choice's options.

    ``min_options``/``max_options`` bound the subset size, the true option
    always included; ``None`` for ``max_options`` keeps up to every option.
    ``shuffle`` reorders; ``rename`` redraws each name's surface form with this
    probability; ``paraphrases`` maps an option name to other names that mean
    the same, one of which replaces it with probability ``paraphrase``.
    ``criteria_only`` replaces names with opaque ids and moves the name into
    the criteria, so the only way to find the answer is to read the criteria.
    """

    min_options: int = 2
    max_options: int | None = None
    shuffle: bool = True
    rename: float = 0.0
    paraphrases: dict[str, Sequence[str]] | None = None
    paraphrase: float = 0.0
    criteria_only: float = 0.0
    #: Share of reshapes drawn *above* the scoring crossover, where a question
    #: has more options than it. A Choice over more than
    #: `DOT_PRODUCT_CROSSOVER` options is scored by the dot-product head and
    #: one over fewer by the per-option readout head, so a size drawn uniformly
    #: from 2..77 trains the second and leaves the first near its
    #: initialisation: validation at 77 options sat at ln 77 for three epochs
    #: while training loss halved. 0.5 trains both. Ignored for a question
    #: that cannot exceed the crossover.
    crossover_fraction: float = 0.0
    crossover: int = DOT_PRODUCT_CROSSOVER

    def __call__(self, case: Case, rng: random.Random) -> Case:
        return reshape_case(case, rng, self)


def _distribution(expected: Expectation, keep: Sequence[int]) -> tuple[float, ...] | None:
    if expected.distribution is None:
        return None
    mass = [expected.distribution[i] for i in keep]
    total = sum(mass)
    if total <= 0:
        return None
    return tuple(m / total for m in mass)


def _redraw_names(
    options: list[OptionSpec], rng: random.Random, spec: Reshape
) -> list[OptionSpec] | None:
    if spec.criteria_only and rng.random() < spec.criteria_only:
        width = len(str(len(options)))
        return [
            OptionSpec(
                name=f"option_{i + 1:0{width}d}",
                criteria=(f"{o.name}: {o.criteria}" if o.criteria else o.name),
            )
            for i, o in enumerate(options)
        ]
    renamed = []
    for option in options:
        name = option.name
        alternatives = (spec.paraphrases or {}).get(name)
        if alternatives and spec.paraphrase and rng.random() < spec.paraphrase:
            name = rng.choice(list(alternatives))
        if spec.rename and rng.random() < spec.rename:
            name = RENAMES[rng.choice(sorted(RENAMES))](name) or name
        renamed.append(OptionSpec(name=name[:256], criteria=option.criteria))
    # A redraw that collides two names would make the answer ambiguous; keep
    # the declared names rather than return a question with no right answer.
    if len({o.name for o in renamed}) != len(renamed):
        return None
    return renamed


def reshape_case(case: Case, rng: random.Random, spec: Reshape) -> Case:
    """``case`` with every Choice question's options redrawn under ``spec``.

    The expectation follows its option: the label is re-indexed, and an
    annotator or teacher distribution is restricted to the kept options and
    renormalised. A question whose true option is unknown keeps its full set,
    because a subset drawn without it could remove the right answer.
    """
    questions = dict(case.request.questions)
    expected = dict(case.expected)
    changed = False
    for qid, question in case.request.questions.items():
        if not isinstance(question, ChoiceQuestion):
            continue
        truth = expected.get(qid)
        label = truth.hard_label if truth is not None else None
        n = len(question.options)
        indices = list(range(n))
        if label is not None:
            upper = n if spec.max_options is None else min(n, spec.max_options)
            lower = min(max(2, spec.min_options), upper)
            if spec.crossover_fraction and upper > spec.crossover >= lower:
                if rng.random() < spec.crossover_fraction:
                    lower = spec.crossover + 1
                else:
                    upper = spec.crossover
            size = rng.randint(lower, upper)
            others = [i for i in indices if i != label]
            keep = sorted([label, *rng.sample(others, size - 1)])
        else:
            keep = indices
        if spec.shuffle:
            rng.shuffle(keep)
        options = [question.options[i] for i in keep]
        redrawn = _redraw_names(options, rng, spec)
        if redrawn is not None:
            options = redrawn
        questions[qid] = question.model_copy(update={"options": options})
        if truth is not None:
            expected[qid] = replace(
                truth,
                label=keep.index(label) if truth.label is not None and label is not None else None,
                distribution=_distribution(truth, keep),
            )
        changed = True
    if not changed:
        return case
    request = case.request.model_copy(update={"questions": questions})
    return replace(case, request=request, expected=expected)


def reshape_all(cases: Sequence[Case], spec: Reshape, seed: int) -> list[Case]:
    """Every case reshaped once, reproducibly: the evaluation form."""
    rng = random.Random(seed)
    return [reshape_case(case, rng, spec) for case in cases]
