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
import gzip
import hashlib
import json
import os
import pathlib
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import ChoiceQuestion, ScoreQuestion, SystemOneRequest
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
    # -- Choice corpora --------------------------------------------------
    # One column holds the text and one holds the label.
    text_field: str = "text"
    label_field: str = "category"
    # -- Score corpora ---------------------------------------------------
    # Several ordered ratings over one piece of state, each its own question.
    # The levels are named by their index for the same reason the compat
    # adapter names them that way: an ordinal rating's number *is* its label,
    # and a Score anchored at its indices reports on the caller's own scale.
    score_fields: tuple[str, ...] = ()
    levels: int = 0
    # Fields concatenated to make the state, in order, each under its own
    # heading. A record is state; flattening it loses which part was which.
    state_fields: tuple[str, ...] = ()
    per_question_instructions: dict[str, str] = field(default_factory=dict)
    # -- Annotator-distribution corpora ------------------------------------
    # Each rating field holds every annotator's rating as a list rather than
    # one aggregated integer. The expectation carries the empirical
    # distribution, which is what training fits, and one annotator drawn at
    # random per case as the label the gates score against -- a model whose
    # probabilities match annotator disagreement is calibrated against a
    # random annotator by definition, so the existing gates test exactly the
    # claim this data exists for.
    annotator_lists: bool = False
    # A corpus shipped as one file is split here, by a hash of the field named
    # below, with this share held out as "test". Hashing the *prompt* rather
    # than the row keeps every response to one prompt on one side: several
    # share a prompt, and a row-level split would test on prompts it trained on.
    holdout_fraction: float = 0.0
    holdout_key: str = "prompt"

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

HELPSTEER2 = CorpusSpec(
    name="helpsteer2",
    primitive="score",
    tier="green",
    licence="CC BY 4.0",
    attribution=(
        "HelpSteer2 (Wang et al., 2024), NVIDIA. CC BY 4.0. "
        "https://huggingface.co/datasets/nvidia/HelpSteer2"
    ),
    files={
        "train": "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/train.jsonl.gz",
        "test": (
            "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/validation.jsonl.gz"
        ),
    },
    instructions="Rate this response.",
    # Five ordered ratings, 0-4, over the same prompt-and-response pair. This
    # is the first real **Score** corpus here, which matters because Score is
    # the primitive with the least real evidence behind it and the one the
    # synthetic `size` question failed on for seven straight interventions.
    #
    # These are aggregated integer ratings, **not** annotator distributions.
    # The distribution-carrying split lives in `disagreements/` and is a
    # separate thing to load; saying otherwise here would claim the product's
    # headline data stream on the strength of a file that does not carry it.
    score_fields=("helpfulness", "correctness", "coherence", "complexity", "verbosity"),
    levels=5,
    state_fields=("prompt", "response"),
    per_question_instructions={
        "helpfulness": "How helpful is the response to the prompt?",
        "correctness": "How correct and factually accurate is the response?",
        "coherence": "How coherent and easy to follow is the response?",
        "complexity": "How much domain expertise does writing the response require?",
        "verbosity": "How verbose is the response relative to what was asked?",
    },
)

HELPSTEER2_ANNOTATORS = CorpusSpec(
    name="helpsteer2-annotators",
    primitive="score",
    tier="green",
    licence="CC BY 4.0",
    attribution=HELPSTEER2.attribution,
    files={
        # Pinned: this split is the annotator ratings behind HelpSteer2, and a
        # moving file under a published number is a second source of truth.
        "all": (
            "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/"
            "990b2711a36180dd19d9c94b8627844866f8982a/disagreements/disagreements.jsonl.gz"
        ),
    },
    instructions=HELPSTEER2.instructions,
    # The same five questions over the same kind of state, with every
    # annotator's rating instead of their average: 23,652 pairs, two to six
    # annotators each. This is the annotator-distribution stream -- the one
    # that teaches a model what disagreement looks like, which is the product.
    score_fields=HELPSTEER2.score_fields,
    levels=HELPSTEER2.levels,
    state_fields=HELPSTEER2.state_fields,
    per_question_instructions=HELPSTEER2.per_question_instructions,
    annotator_lists=True,
    # A quarter, so the held-out split alone clears MIN_CALIBRATION_SAMPLES
    # and no evaluation row is topped up from training prompts.
    holdout_fraction=0.25,
)

CORPORA: dict[str, CorpusSpec] = {c.name: c for c in (BANKING77, HELPSTEER2, HELPSTEER2_ANNOTATORS)}


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
        path = root / f"{split}{_extension(url)}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                path.write_bytes(response.read())
        paths[split] = path
    return paths


def _records(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    """Rows from a corpus file, as dicts, whichever plain format it is in.

    CSV and gzipped JSONL, both stdlib. A Parquet-only corpus needs a reader
    this module deliberately does not have -- `pyarrow` in `trigon.evals`
    would put a compiled dependency in the import path of the calibration math
    and the drift tests, which is the thing the package's dependency rule
    exists to prevent. Such a corpus gets converted in `scripts/` first.
    """
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    opener = gzip.open if path.suffixes[-1:] == [".gz"] else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extension(url: str) -> str:
    """The suffix a cached file keeps, so `_records` can dispatch on it."""
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".csv"):
        if url.endswith(suffix):
            return suffix
    return ".csv"


def _labelled(spec: CorpusSpec, path: pathlib.Path) -> Iterator[tuple[str, str]]:
    for row in _records(path):
        text = str(row.get(spec.text_field) or "").strip()
        label = str(row.get(spec.label_field) or "").strip()
        if text and label:
            yield text, label


def _state(spec: CorpusSpec, row: dict[str, Any]) -> str:
    """A record's fields under their own headings, in declared order.

    Concatenating them bare would make a prompt and a response
    distinguishable only by position, which is a thing the model would have to
    learn rather than be told.
    """
    return "\n\n".join(
        f"{name.upper()}:\n{str(row.get(name) or '').strip()}" for name in spec.state_fields
    )


def _held_out(spec: CorpusSpec, row: dict[str, Any]) -> bool:
    key = str(row.get(spec.holdout_key) or "").strip().encode()
    bucket = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") / 2**32
    return bucket < spec.holdout_fraction


def _score_cases(spec: CorpusSpec, split: str, path: pathlib.Path, limit: int | None) -> list[Case]:
    """Several ordered ratings over one piece of state, each its own question.

    One request, several Score questions, which is exactly the shape the
    independence claim is about -- and the first time it is exercised on real
    data rather than on the generator.
    """
    levels = [{"name": str(i), "value": float(i)} for i in range(spec.levels)]
    questions = {
        name: ScoreQuestion(
            instructions=spec.per_question_instructions.get(name, spec.instructions),
            levels=levels,
        )
        for name in spec.score_fields
    }
    cases: list[Case] = []
    for i, row in enumerate(_records(path)):
        if limit is not None and len(cases) >= limit:
            break
        if spec.holdout_fraction and _held_out(spec, row) != (split == "test"):
            continue
        expected = {}
        for name in spec.score_fields:
            value = row.get(name)
            if spec.annotator_lists:
                ratings = value if isinstance(value, list) else []
                if not ratings or not all(
                    isinstance(r, int) and 0 <= r < spec.levels for r in ratings
                ):
                    expected = {}
                    break
                counts = [ratings.count(level) for level in range(spec.levels)]
                # One annotator per case and question, fixed by the case: the
                # outcome the gates score, drawn so a rerun draws the same one.
                draw = int.from_bytes(
                    hashlib.blake2b(f"{spec.name}/{i}/{name}".encode(), digest_size=4).digest(),
                    "big",
                )
                expected[name] = Expectation(
                    label=ratings[draw % len(ratings)],
                    distribution=tuple(c / len(ratings) for c in counts),
                )
                continue
            # A rating outside the declared range is not clipped into it: a
            # silently clamped label trains the model on an answer nobody
            # gave. The row is dropped and the drop is visible in the count.
            if not isinstance(value, int) or not 0 <= value < spec.levels:
                expected = {}
                break
            expected[name] = Expectation(label=value)
        if not expected:
            continue
        cases.append(
            Case(
                case_id=f"{spec.name}/{split}/{i}",
                request=SystemOneRequest(state=_state(spec, row), questions=questions),
                expected=expected,
                domain=spec.name,
                tags=(spec.name, split, "real"),
            )
        )
    return cases


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
    if spec.holdout_fraction:
        # One file, split here by prompt; see `holdout_fraction`.
        if split not in ("train", "test"):
            raise KeyError(f"{spec.name} has splits 'train' and 'test'; got {split!r}")
        paths = {split: paths["all"]}
    if split not in paths:
        raise KeyError(f"{spec.name} has no split {split!r}; it has {sorted(paths)}")
    if spec.primitive == "score":
        return _score_cases(spec, split, paths[split], limit)

    rows = list(_labelled(spec, paths[split]))
    # The option set is every label in the corpus, in a fixed order, on every
    # request. It is not the labels present in this split: a model asked to
    # choose between 70 options on train and 77 on test is being asked two
    # different questions, and the second one is harder for a reason that has
    # nothing to do with the model.
    labels = sorted({label for path in paths.values() for _, label in _labelled(spec, path)})
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
