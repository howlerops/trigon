"""Real labelled corpora, loaded under the licence policy rather than beside it.

`docs/data.md` cleared nine corpora to green and four streams of five stayed
unbuilt, so every calibration number this project has published rests on
synthetic data. Synthetic data has clean state, decidable predicates and no
subjectivity; it exercises the machinery and it cannot tell you whether the
machinery transfers. This module is where that stops being true.

**The licence tier is enforced here, in code.** `docs/decisions.md` §3 says
green trains, evals and redistributes; amber evals only, never in a training
mix; red is a local copy and ships in nothing. A policy that lives only in a
document is one hurried afternoon away from a training mix with a research-only
corpus in it, which is not a bug you can find later by reading the weights. So
`load()` takes the purpose it is being loaded for and refuses the combinations
the policy refuses -- `CorpusLicenceError`, not a warning.

**Nothing here is committed.** The corpora are fetched to a cache directory on
first use and the cache is ignored by git. CC BY 4.0 permits redistribution
with attribution and we still do not redistribute: a checked-in copy is a
second source of truth for someone else's data, and it goes stale silently.
`CorpusSpec.attribution` carries the credit the licence requires, and
`scripts/train_corpus.py` prints it into every report it writes.

Downloading needs the network; loading a cached copy does not, and the tests
never touch either -- they build a tiny corpus on disk and read it back.
"""

from __future__ import annotations

import csv
import os
import pathlib
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from ..types import ChoiceQuestion, SystemOneRequest
from .harness import Case, Expectation

__all__ = [
    "CORPORA",
    "CorpusLicenceError",
    "CorpusSpec",
    "cache_root",
    "corpus",
    "load",
]

Purpose = Literal["train", "eval", "redistribute"]
Tier = Literal["green", "amber", "red"]

# docs/decisions.md §3, as a table the code can be held to.
_PERMITS: dict[Tier, frozenset[str]] = {
    "green": frozenset({"train", "eval", "redistribute"}),
    "amber": frozenset({"eval"}),
    "red": frozenset(),
}


class CorpusLicenceError(RuntimeError):
    """A corpus asked for a use its tier does not permit."""


@dataclass(frozen=True)
class CorpusSpec:
    """One corpus: where it comes from, what it may be used for, who gets credit."""

    name: str
    primitive: str
    tier: Tier
    licence: str
    # Attribution the licence requires. Printed into every report, because a
    # credit that lives only in a docs table is not attached to the number.
    attribution: str
    # Plain files, deliberately: a loader that needs `datasets` pulls a heavy
    # dependency into a package whose whole point is that the calibration math
    # imports without one.
    files: dict[str, str]
    instructions: str

    def permits(self, purpose: Purpose) -> bool:
        return purpose in _PERMITS[self.tier]


BANKING77 = CorpusSpec(
    name="banking77",
    primitive="choice",
    tier="green",
    licence="CC BY 4.0",
    attribution=(
        "Banking77 (Casanueva et al., 2020), PolyAI. CC BY 4.0. "
        "https://github.com/PolyAI-LDN/task-specific-datasets"
    ),
    files={
        "train": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
            "/master/banking_data/train.csv"
        ),
        "test": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets"
            "/master/banking_data/test.csv"
        ),
    },
    instructions="Which banking intent does this customer message express?",
)

CORPORA: dict[str, CorpusSpec] = {BANKING77.name: BANKING77}


def corpus(name: str) -> CorpusSpec:
    try:
        return CORPORA[name]
    except KeyError:
        known = ", ".join(sorted(CORPORA))
        raise KeyError(f"unknown corpus {name!r}; known corpora are {known}") from None


def cache_root() -> pathlib.Path:
    """Where fetched corpora live. Ignored by git, overridable for tests."""
    return pathlib.Path(os.environ.get("TRIGON_CORPUS_CACHE", "corpora")).expanduser()


def fetch(spec: CorpusSpec, *, root: pathlib.Path | None = None) -> dict[str, pathlib.Path]:
    """Download the corpus into the cache if it is not already there."""
    root = (root or cache_root()) / spec.name
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in spec.files.items():
        path = root / f"{split}.csv"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                path.write_bytes(response.read())
        paths[split] = path
    return paths


def _rows(path: pathlib.Path) -> Iterator[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("text") or "").strip()
            category = (row.get("category") or "").strip()
            if text and category:
                yield text, category


def load(
    name: str,
    split: str,
    *,
    purpose: Purpose,
    limit: int | None = None,
    root: pathlib.Path | None = None,
) -> list[Case]:
    """Cases from a corpus, refused outright if its tier forbids ``purpose``.

    ``purpose`` is required and has no default. A default would be the one the
    caller least often thinks about, and the expensive mistake here -- an
    amber corpus in a training mix -- is exactly the one a default hides.
    """
    spec = corpus(name)
    if not spec.permits(purpose):
        raise CorpusLicenceError(
            f"{spec.name} is {spec.tier} ({spec.licence}) and may not be used to "
            f"{purpose}; docs/decisions.md section 3 has the policy"
        )
    paths = fetch(spec, root=root)
    if split not in paths:
        raise KeyError(f"{spec.name} has no split {split!r}; it has {sorted(paths)}")

    rows = list(_rows(paths[split]))
    # The option set is every label in the corpus, in a fixed order, on every
    # request. It is not the labels present in this split: a model asked to
    # choose between 70 options on train and 77 on test is being asked two
    # different questions, and the second one is harder for a reason that has
    # nothing to do with the model.
    labels = sorted({category for path in paths.values() for _, category in _rows(path)})
    index = {label: i for i, label in enumerate(labels)}
    options = [{"name": label.replace("_", " ")} for label in labels]

    cases = []
    for i, (text, category) in enumerate(rows[:limit] if limit else rows):
        cases.append(
            Case(
                case_id=f"{spec.name}/{split}/{i}",
                request=SystemOneRequest(
                    state=text,
                    questions={
                        "intent": ChoiceQuestion(instructions=spec.instructions, options=options)
                    },
                ),
                expected={"intent": Expectation(label=index[category])},
                domain=spec.name,
                tags=(spec.name, split, "real"),
            )
        )
    return cases
