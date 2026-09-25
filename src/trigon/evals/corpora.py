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
    # A corpus whose shape the generic paths below do not cover brings its own
    # loader, registered beside its spec. The licence check above runs first
    # for every corpus, whichever loader reads it.
    custom = _LOADERS.get(spec.name)
    if custom is not None:
        return custom(spec, split, limit=limit, root=root)
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


# -- Loaders for corpora the generic paths do not cover -----------------------
#
# Each registers itself here, beside its spec, so adding one touches nothing
# above but `load()`'s single lookup.

_LOADERS: dict[str, Any] = {}


# -- HateXplain: a label per annotator and the words each one highlighted -----
#
# The one corpus here whose annotators said *why*: for every post labelled
# hateful or offensive, each annotator who said so marked the tokens their
# label rested on. That is what the evidence head trains on and what its
# plausibility is scored against (`trigon.evals.rationale`).
#
# **Licence, verified 2026-09-25 against both primary sources.** The authors'
# repository, which is where the file is fetched from, carries an MIT LICENSE
# ("Copyright (c) 2020 Punyajoy Saha"); the authors' own Hugging Face dataset
# card (`Hate-speech-CNERG/hatexplain`) declares CC BY 4.0. Both are green
# under docs/decisions.md section 3. The posts themselves are Twitter and Gab
# text; neither source attaches platform terms to them, and `docs/data.md`
# records that as the caveat it is.
#
# **The split is by a hash of the post id**, a quarter held out, so a rerun
# holds out the same posts and no post is on both sides. The authors publish
# an 8:1:1 split of their own; it is not used, so that every corpus here is
# split by one rule, and published HateXplain numbers are therefore not
# directly comparable with ours.

HATEXPLAIN_REVISION = "01d742279dac941981f53806154481c0e15ee686"
#: SHA-256 of `Data/dataset.json` at that revision. A pinned commit cannot
#: move, but a cache can be written by anything: checked on every download, and
#: on every load from the default cache (a test's hand-built fixture under an
#: explicit ``root`` is not the pinned file and is not held to its hash).
HATEXPLAIN_SHA256 = "63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb"
#: Declared low to high. `label` is the index into this.
HATEXPLAIN_LABELS = ("normal", "offensive", "hatespeech")

HATEXPLAIN = CorpusSpec(
    name="hatexplain",
    primitive="choice",
    tier="green",
    licence="MIT",
    attribution=(
        "HateXplain (Mathew et al., AAAI 2021), Punyajoy Saha and co-authors. "
        "MIT (repository LICENSE, Copyright (c) 2020 Punyajoy Saha); the authors' "
        "dataset card states CC BY 4.0. "
        f"https://github.com/punyajoy/HateXplain at {HATEXPLAIN_REVISION[:12]}"
    ),
    files={
        "all": (
            "https://raw.githubusercontent.com/punyajoy/HateXplain/"
            f"{HATEXPLAIN_REVISION}/Data/dataset.json"
        ),
    },
    instructions="Is this post hate speech, offensive, or neither?",
    holdout_fraction=0.25,
    holdout_key="post_id",
)

_HATEXPLAIN_OPTIONS = [
    {"name": "normal", "criteria": "neither hateful nor offensive"},
    {"name": "offensive", "criteria": "abusive or offensive, but not hate speech"},
    {
        "name": "hatespeech",
        "criteria": "attacks or demeans a group for who they are: race, religion, "
        "gender, sexual orientation, origin or disability",
    },
]


def _hatexplain_file(spec: CorpusSpec, root: pathlib.Path | None) -> pathlib.Path:
    """The pinned file, fetched once into the cache and checked on every load."""
    folder = (root or cache_root()) / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dataset.json"
    if not path.exists():
        with urllib.request.urlopen(spec.files["all"], timeout=120) as response:  # noqa: S310
            blob = response.read()
        if hashlib.sha256(blob).hexdigest() != HATEXPLAIN_SHA256:
            raise ValueError(f"{spec.files['all']} does not match the pinned SHA-256")
        path.write_bytes(blob)
    return path


def hatexplain_case(spec: CorpusSpec, post: dict[str, Any], case_id: str) -> Case | None:
    """One post as a case, or ``None`` for a post the annotators split three ways.

    * **State** is the post's tokens joined by single spaces -- the exact text
      the annotators highlighted, so a highlighted token is a character span of
      the state with no alignment step to get wrong.
    * **Label** is the majority of the three annotators; a three-way split has
      no majority and is dropped, as the authors drop it. **Distribution** is
      all three, which is what training fits.
    * **Rationale** is the majority over the annotators who gave one: a token
      is in it when at least half of them marked it. Only annotators who
      labelled a post hateful or offensive were asked, so a post whose
      majority is ``normal`` has no rationale (``None``), which is not the
      same as an empty one.
    """
    tokens = [str(t) for t in post.get("post_tokens") or []]
    votes = [a.get("label") for a in post.get("annotators") or []]
    if not tokens or not votes or any(v not in HATEXPLAIN_LABELS for v in votes):
        return None
    counts = [votes.count(label) for label in HATEXPLAIN_LABELS]
    top = max(counts)
    if counts.count(top) > 1 or top * 2 <= len(votes):
        return None
    label = counts.index(top)

    offsets, cursor = [], 0
    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token) + 1
    state = " ".join(tokens)

    rationale = None
    marks = [r for r in post.get("rationales") or [] if len(r) == len(tokens)]
    if HATEXPLAIN_LABELS[label] != "normal" and marks:
        chosen = [sum(m[i] for m in marks) * 2 >= len(marks) for i in range(len(tokens))]
        spans: list[tuple[int, int]] = []
        for i, keep in enumerate(chosen):
            if not keep:
                continue
            start, end = offsets[i]
            if spans and chosen[i - 1]:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        rationale = tuple(spans)

    return Case(
        case_id=case_id,
        request=SystemOneRequest(
            state=state,
            questions={
                "label": ChoiceQuestion(instructions=spec.instructions, options=_HATEXPLAIN_OPTIONS)
            },
        ),
        expected={
            "label": Expectation(
                label=label,
                distribution=tuple(c / len(votes) for c in counts),
                rationale=rationale,
            )
        },
        domain=spec.name,
        tags=(spec.name, "real", "rationale"),
    )


def _load_hatexplain(
    spec: CorpusSpec, split: str, *, limit: int | None, root: pathlib.Path | None
) -> list[Case]:
    if split not in ("train", "test"):
        raise KeyError(f"{spec.name} has splits 'train' and 'test'; got {split!r}")
    path = _hatexplain_file(spec, root)
    blob = path.read_bytes()
    if root is None and hashlib.sha256(blob).hexdigest() != HATEXPLAIN_SHA256:
        raise ValueError(
            f"{path} does not match the pinned revision's SHA-256; delete it to refetch"
        )
    posts = json.loads(blob)
    cases = []
    # Sorted by post id, so the order -- and so what `limit` keeps -- does not
    # depend on how the file happens to be serialized.
    for post_id in sorted(posts):
        if limit is not None and len(cases) >= limit:
            break
        post = posts[post_id]
        if _held_out(spec, {"post_id": post_id}) != (split == "test"):
            continue
        case = hatexplain_case(spec, post, f"{spec.name}/{split}/{post_id}")
        if case is not None:
            cases.append(case)
    return cases


CORPORA[HATEXPLAIN.name] = HATEXPLAIN
_LOADERS[HATEXPLAIN.name] = _load_hatexplain
