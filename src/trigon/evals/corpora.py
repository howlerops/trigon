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

**Share-alike is refused by its licence, not only by its tier.** The owner's
decision of 2026-09-25 (`docs/decisions.md` Q17): a CC BY-SA corpus may
evaluate the model and never train it, so the shipped weights carry no
ShareAlike obligation. A tier is a field somebody types; the licence string is
what the decision is about. So a share-alike corpus refuses training even if
its tier is mistyped as green -- and declaring one green is refused when the
spec is built, before anything is loaded.

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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from .harness import Case, Expectation

__all__ = [
    "CORPORA",
    "CorpusLicenceError",
    "CorpusNotConverted",
    "CorpusSpec",
    "cache_root",
    "corpus",
    "is_share_alike",
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

# What a share-alike licence permits, whatever tier it has been given: the
# owner's decision of 2026-09-25 (docs/decisions.md Q17). Evaluating a model
# on the data does not make the weights a derivative of it; training does.
_SHARE_ALIKE_PERMITS = frozenset({"eval"})


def is_share_alike(licence: str) -> bool:
    """Whether a licence string names a ShareAlike term (CC BY-SA, any version).

    Read off the licence rather than the tier, so the rule holds for a corpus
    somebody tiered by hand. Also used by the test that holds every row of the
    audit table in `docs/data.md` to it -- BoolQ and FEVER live only there.
    """
    normalised = licence.upper().replace("‑", "-")
    return "BY-SA" in normalised or "SHAREALIKE" in normalised.replace(" ", "")


class CorpusLicenceError(RuntimeError):
    """A corpus asked for a use its tier does not permit."""


class CorpusNotConverted(FileNotFoundError):
    """A corpus published only in a format this module deliberately cannot read.

    It is converted once, by a script that may import a compiled reader, into
    the plain file this module does read. See `CorpusSpec.converted_from`.
    """


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
    # SHA-256 of each downloaded file, by split. A URL with no revision in it
    # (a storage bucket) is pinned by its content instead, and a file that has
    # changed under a published number fails the fetch rather than the reader.
    sha256: dict[str, str] = field(default_factory=dict)
    # A corpus published only as Parquet: `scripts/convert_corpus.py`
    # downloads `files` and writes `<split>.jsonl.gz` into the cache, and this
    # module reads that. Never read here -- see `_records`.
    converted_from: str = ""
    # -- Choice corpora --------------------------------------------------
    # One column holds the text and one holds the label.
    text_field: str = "text"
    label_field: str = "category"
    # A declared option set, in order, for a Choice whose labels are
    # annotator judgements. Declared rather than read off the data so a slice
    # in which nobody chose one option still asks the same question.
    choice_labels: tuple[str, ...] = ()
    # Every annotator's judgement in one field, joined by this separator.
    label_separator: str = ""
    # -- Score corpora ---------------------------------------------------
    # Several ordered ratings over one piece of state, each its own question.
    # The levels are named by their index for the same reason the compat
    # adapter names them that way: an ordinal rating's number *is* its label,
    # and a Score anchored at its indices reports on the caller's own scale.
    score_fields: tuple[str, ...] = ()
    levels: int = 0
    # Per-question level counts where the items of one survey differ.
    score_levels: dict[str, int] = field(default_factory=dict)
    # Fields concatenated to make the state, in order, each under its own
    # heading. A record is state; flattening it loses which part was which.
    state_fields: tuple[str, ...] = ()
    per_question_instructions: dict[str, str] = field(default_factory=dict)
    # -- Noul corpora ------------------------------------------------------
    # Each question is a yes/no over a group of binary columns: an annotator
    # says yes when they marked any column in the group.
    noul_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # -- Annotator-distribution corpora ------------------------------------
    # Each rating field holds every annotator's rating as a list rather than
    # one aggregated integer. The expectation carries the empirical
    # distribution, which is what training fits, and one annotator drawn at
    # random per case as the label the gates score against -- a model whose
    # probabilities match annotator disagreement is calibrated against a
    # random annotator by definition, so the existing gates test exactly the
    # claim this data exists for.
    annotator_lists: bool = False
    # The lists arrive as one row per annotator instead, grouped here by this
    # field into one case. Every row of a group lands in the same case, and so
    # on the same side of the holdout.
    group_field: str = ""
    # A per-annotator flag meaning "I could not judge this"; such a row is an
    # abstention, not a rating of zero on every question.
    abstain_field: str = ""
    # Fewer usable ratings than this and the item is dropped: one annotator's
    # rating written as a distribution is a hard label pretending otherwise.
    min_raters: int = 1
    # A corpus shipped as one file is split here, by a hash of the field named
    # below, with this share held out as "test". Hashing the *prompt* rather
    # than the row keeps every response to one prompt on one side: several
    # share a prompt, and a row-level split would test on prompts it trained on.
    holdout_fraction: float = 0.0
    holdout_key: str = "prompt"

    def __post_init__(self) -> None:
        if self.share_alike and self.tier == "green":
            raise ValueError(
                f"{self.name} is {self.licence}: a share-alike corpus evaluates and "
                "never trains (docs/decisions.md Q17), so it cannot be green"
            )

    @property
    def share_alike(self) -> bool:
        return is_share_alike(self.licence)

    def permits(self, purpose: Purpose) -> bool:
        if self.share_alike and purpose not in _SHARE_ALIKE_PERMITS:
            return False
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

# The authors' own grouping of the 27 emotions into Ekman's six
# (goemotions/data/ekman_mapping.json in google-research), plus neutral.
_EKMAN: dict[str, tuple[str, ...]] = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": (
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ),
    "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
    "surprise": ("surprise", "realization", "confusion", "curiosity"),
    "neutral": ("neutral",),
}

_GOEMOTIONS_BUCKET = "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset"

GOEMOTIONS = CorpusSpec(
    name="goemotions",
    primitive="noul",
    tier="green",
    licence="Apache-2.0",
    attribution=(
        "GoEmotions (Demszky et al., 2020), Google Research. Apache-2.0. "
        "https://github.com/google-research/google-research/tree/master/goemotions"
    ),
    # The raw release: one row per rater per comment, 211,225 rows over
    # 58,011 comments and 82 raters. The filtered train/dev/test TSVs keep only
    # labels two raters agreed on, which is the disagreement thrown away.
    files={f"part{i}": f"{_GOEMOTIONS_BUCKET}/goemotions_{i}.csv" for i in (1, 2, 3)},
    # A bucket URL carries no revision, so the content is the pin.
    sha256={
        "part1": "cac049036bad5d68d1081f72b65f2cc51e4df82af05e3e22cfa747051cac1af3",
        "part2": "f699ecc5aa425c1720c1d02475f1e41815244b680bd75b282eb770d2c76cd84d",
        "part3": "467f1e7191af00f2e76cc7f425885c2dc304bea8aff284b10e8c460d22f2e1af",
    },
    instructions="Does this Reddit comment express the emotion described?",
    # **Seven Nouls, one per Ekman group plus neutral -- not a Choice, and not
    # 28 Nouls.** Raters mark as many emotions as apply, so a rater's answer is
    # not one option of a Choice and forcing it into one would need a
    # tie-break nobody annotated. Per emotion, 28 questions would multiply the
    # schema block by four and most of them are marked on under 3% of ratings,
    # so their gates would read the marginal. The Ekman grouping is the
    # authors' own, and each group is common enough to carry a distribution:
    # a rater says yes to "anger" when they marked anger, annoyance or
    # disapproval, and the fraction of raters who did is the target.
    noul_groups=_EKMAN,
    per_question_instructions={
        "anger": "Does this comment express anger, annoyance or disapproval?",
        "disgust": "Does this comment express disgust?",
        "fear": "Does this comment express fear or nervousness?",
        "joy": (
            "Does this comment express joy or another positive emotion: amusement, "
            "approval, excitement, gratitude, love, optimism, relief, pride, "
            "admiration, desire or caring?"
        ),
        "sadness": (
            "Does this comment express sadness, disappointment, embarrassment, grief or remorse?"
        ),
        "surprise": "Does this comment express surprise, realization, confusion or curiosity?",
        "neutral": "Is this comment emotionally neutral?",
    },
    state_fields=("text",),
    annotator_lists=True,
    group_field="id",
    # A rater who marked the comment "very unclear" chose no emotion; that is
    # an abstention, not seven noes.
    abstain_field="example_very_unclear",
    min_raters=2,
    # By the text, not the comment id: 173 texts are posted under more than
    # one id, and a duplicate on both sides is a test item trained on.
    holdout_key="text",
    # A fifth of 58k comments is ~11k, well over MIN_CALIBRATION_SAMPLES.
    holdout_fraction=0.2,
)

_MHS_ITEMS = (
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hatespeech",
)

MEASURING_HATE_SPEECH = CorpusSpec(
    name="measuring_hate_speech",
    primitive="score",
    tier="green",
    licence="CC BY 4.0",
    attribution=(
        "Measuring Hate Speech (Kennedy et al., 2020; Sachdeva et al., 2022), "
        "UC Berkeley D-Lab. CC BY 4.0. "
        "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech"
    ),
    # Parquet only. `scripts/convert_corpus.py measuring_hate_speech` turns it
    # into the gzipped JSONL read here. Pinned to a revision.
    files={
        "all": (
            "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/"
            "5468f6e118396646b02a2f691e771f6b6d9502ea/measuring-hate-speech.parquet"
        ),
    },
    sha256={"all": "6819525ce61bc24344df9fc3f7bf48270b31038273cc27c67fc225b51433b0e1"},
    converted_from="parquet",
    instructions="Rate this social media comment.",
    # The ten ordinal survey items the corpus's hate speech score is built
    # from, each its own Score question and each per annotator: 135,556
    # ratings of 39,565 comments by 7,912 annotators. All ten are coded so a
    # higher level is more hateful -- each correlates +0.50 to +0.85 with the
    # corpus's IRT score -- and the instructions say which end is which,
    # because a level named "3" means nothing on its own. The IRT score itself
    # is not a question: it is a model's output, not anyone's judgement.
    score_fields=_MHS_ITEMS,
    levels=5,
    score_levels={"hatespeech": 3},
    state_fields=("text",),
    per_question_instructions={
        "sentiment": "Overall sentiment of the comment: 0 strongly positive, 4 strongly negative.",
        "respect": (
            "Is the comment respectful towards the group it targets? "
            "0 strongly respectful, 4 strongly disrespectful."
        ),
        "insult": "Does the comment insult the group? 0 strongly disagree, 4 strongly agree.",
        "humiliate": (
            "Does the comment humiliate the group? 0 strongly disagree, 4 strongly agree."
        ),
        "status": (
            "Does the comment say the group is inferior to others? "
            "0 strongly disagree, 4 strongly agree."
        ),
        "dehumanize": (
            "Does the comment dehumanize the group, comparing it to animals or objects? "
            "0 strongly disagree, 4 strongly agree."
        ),
        "violence": (
            "Does the comment call for violence against the group? "
            "0 strongly disagree, 4 strongly agree."
        ),
        "genocide": (
            "Does the comment call for the deliberate killing of the group? "
            "0 strongly disagree, 4 strongly agree."
        ),
        "attack_defend": (
            "Does the comment attack or defend the group? "
            "0 strongly defends, 2 neither, 4 strongly attacks."
        ),
        "hatespeech": "Is the comment hate speech? 0 no, 1 unsure, 2 yes.",
    },
    annotator_lists=True,
    group_field="comment_id",
    # A comment rated once -- 10,077 of them -- is one person's answer, not a
    # distribution; dropping them leaves ~29,500 comments.
    min_raters=2,
    holdout_key="text",
    holdout_fraction=0.25,
)

CIRCA_LABELS = (
    "Yes",
    "Probably yes / sometimes yes",
    "Yes, subject to some conditions",
    "No",
    "Probably no",
    "In the middle, neither yes nor no",
    "I am not sure how X will interpret Y’s answer",
    # Not in the README's list, and chosen 2,361 times out of 171,320. It is
    # an annotator's answer, so it is an option: dropping it would change the
    # distribution the other seven are measured against.
    "Other",
)

CIRCA = CorpusSpec(
    name="circa",
    primitive="choice",
    # **Amber: evaluation only.** The repository's licence section reads
    # "made available under the Creative Commons Attribution 4.0 License"
    # and then links the *BY-SA* 4.0 text as "a full copy of the license";
    # the Hugging Face card says cc-by-4.0. The two primary statements
    # disagree, and the one that binds is the stricter until the authors say
    # otherwise -- the licence is recorded as BY-SA, so Q17 applies to it.
    tier="amber",
    licence="CC BY-SA 4.0",
    attribution=(
        "Circa (Louis, Roth and Radlinski, 2020), Google. CC BY-SA 4.0 (the "
        "repository's licence section names CC BY 4.0 and links the BY-SA 4.0 "
        "text; read as the stricter). https://github.com/google-research-datasets/circa"
    ),
    # A plain TSV on GitHub, pinned to a commit: no conversion needed.
    files={
        "all": (
            "https://raw.githubusercontent.com/google-research-datasets/circa/"
            "02ad965518ab2fbd8bb24463d312ebb03bac5368/circa-data.tsv"
        ),
    },
    sha256={"all": "98454df6b716dd7ff5f83a3db298849f05414688e81c2ee21b8e5a548ed897aa"},
    instructions=(
        "X asked Y a yes/no question and Y answered indirectly. How would X most "
        "likely interpret Y's answer?"
    ),
    # One Choice over the eight interpretations, with the five annotators'
    # judgements as its distribution. Five per pair, 34,268 pairs.
    choice_labels=CIRCA_LABELS,
    label_field="judgements",
    label_separator="#",
    state_fields=("context", "question-X", "answer-Y"),
    annotator_lists=True,
    min_raters=2,
    # By question: each was answered by ten people, and a split by pair would
    # evaluate on questions the answers of which it had seen.
    holdout_key="question-X",
    holdout_fraction=0.25,
)

CORPORA: dict[str, CorpusSpec] = {
    c.name: c
    for c in (
        BANKING77,
        HELPSTEER2,
        HELPSTEER2_ANNOTATORS,
        GOEMOTIONS,
        MEASURING_HATE_SPEECH,
        CIRCA,
    )
}


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
        if spec.converted_from:
            path = root / f"{split}.jsonl.gz"
            if not path.exists():
                raise CorpusNotConverted(
                    f"{spec.name} is published as {spec.converted_from} only; run "
                    f"`python scripts/convert_corpus.py {spec.name}` to write {path}"
                )
            paths[split] = path
            continue
        path = root / f"{split}{_extension(url)}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                data = response.read()
            verify(spec, split, data)
            path.write_bytes(data)
        paths[split] = path
    return paths


def verify(spec: CorpusSpec, split: str, data: bytes) -> None:
    """Refuse a download whose content is not the pinned content."""
    expected = spec.sha256.get(split)
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(
            f"{spec.name}/{split} does not match its pinned sha256; the file has changed "
            "upstream, and a number published on the old one does not describe it"
        )


def _records(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    """Rows from a corpus file, as dicts, whichever plain format it is in.

    CSV, TSV and gzipped JSONL, all stdlib. A Parquet-only corpus needs a
    reader this module deliberately does not have -- `pyarrow` in
    `trigon.evals` would put a compiled dependency in the import path of the
    calibration math and the drift tests, which is the thing the package's
    dependency rule exists to prevent. Such a corpus gets converted in
    `scripts/` first.
    """
    if path.suffix in (".csv", ".tsv"):
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
        return
    opener = gzip.open if path.suffixes[-1:] == [".gz"] else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extension(url: str) -> str:
    """The suffix a cached file keeps, so `_records` can dispatch on it."""
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".csv", ".tsv"):
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


def _questions(spec: CorpusSpec) -> dict[str, Any]:
    """The request's questions, built from the spec alone and identical on every case."""
    if spec.primitive == "score":
        return {
            name: ScoreQuestion(
                instructions=spec.per_question_instructions.get(name, spec.instructions),
                levels=[
                    {"name": str(i), "value": float(i)}
                    for i in range(spec.score_levels.get(name, spec.levels))
                ],
            )
            for name in spec.score_fields
        }
    if spec.primitive == "noul":
        return {
            name: NoulQuestion(
                instructions=spec.per_question_instructions.get(name, spec.instructions)
            )
            for name in spec.noul_groups
        }
    return {
        "interpretation": ChoiceQuestion(
            instructions=spec.instructions,
            options=[{"name": label} for label in spec.choice_labels],
        )
    }


def _rating(spec: CorpusSpec, name: str, row: dict[str, Any]) -> int | None:
    """One annotator's answer to one question, from their own row; None if unusable."""
    if spec.primitive == "noul":
        marks = [str(row.get(column, "")).strip() for column in spec.noul_groups[name]]
        if not all(mark in ("0", "1", "0.0", "1.0", "True", "False") for mark in marks):
            return None
        return int(any(mark in ("1", "1.0", "True") for mark in marks))
    value = row.get(name)
    # Floats that are integers are accepted (Parquet stores these ratings as
    # doubles); anything else is not rounded into range.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if not 0 <= value < spec.score_levels.get(name, spec.levels):
        return None
    return value


def _items(
    spec: CorpusSpec, paths: Iterable[pathlib.Path]
) -> Iterator[tuple[int, dict[str, Any], dict[str, list[int]] | None]]:
    """Each item once: its index, a representative row, every annotator's ratings.

    The ratings are None when the item is unusable; it is still yielded, so
    the index -- which seeds the drawn annotator -- is the same whatever the
    filter, and the draw on HelpSteer2 is the one the committed runs used.
    """
    rows = (row for path in paths for row in _records(path))
    names = list(_questions(spec))
    index = {label: k for k, label in enumerate(spec.choice_labels)}
    if not spec.group_field:
        for i, row in enumerate(rows):
            ratings: dict[str, list[int]] | None = {}
            if spec.label_separator:
                raw = str(row.get(spec.label_field) or "").split(spec.label_separator)
                judgements = [index.get(r.strip()) for r in raw if r.strip()]
                # A judgement outside the declared set drops the item: mapping
                # it to a neighbour is an answer nobody gave.
                if judgements and all(j is not None for j in judgements):
                    ratings = {names[0]: judgements}  # type: ignore[dict-item]
                else:
                    ratings = None
            else:
                for name in names:
                    value = row.get(name)
                    if not isinstance(value, list) or not value:
                        ratings = None
                        break
                    checked = [_rating(spec, name, {name: v}) for v in value]
                    if any(c is None for c in checked):
                        ratings = None
                        break
                    ratings[name] = checked  # type: ignore[assignment]
            yield i, row, ratings
        return

    # Streamed, not held: GoEmotions is 211,225 rows of 37 columns, and all a
    # group needs is its first row (for the state and the holdout key) and
    # the ratings so far.
    abstaining = ("True", "true", "1")
    groups: dict[str, tuple[dict[str, Any], dict[str, list[int]]]] = {}
    for row in rows:
        key = str(row.get(spec.group_field))
        if key not in groups:
            first = {k: row.get(k) for k in (spec.holdout_key, *spec.state_fields)}
            groups[key] = (first, {name: [] for name in names})
        if spec.abstain_field and str(row.get(spec.abstain_field)).strip() in abstaining:
            continue
        answer = {name: _rating(spec, name, row) for name in names}
        # A rater with an unusable answer to any question is left out whole,
        # so every question of a case is judged by the same panel.
        if any(value is None for value in answer.values()):
            continue
        collected = groups[key][1]
        for name, value in answer.items():
            collected[name].append(value)  # type: ignore[arg-type]
    for i, (first, collected) in enumerate(groups.values()):
        yield i, first, collected


def _levels(spec: CorpusSpec, name: str) -> int:
    if spec.primitive == "noul":
        return 2
    if spec.primitive == "choice":
        return len(spec.choice_labels)
    return spec.score_levels.get(name, spec.levels)


def _annotated_cases(
    spec: CorpusSpec, split: str, paths: list[pathlib.Path], limit: int | None
) -> list[Case]:
    """Every annotator's answer to each question, as a distribution and one drawn label."""
    questions = _questions(spec)
    cases: list[Case] = []
    for i, row, ratings in _items(spec, paths):
        if limit is not None and len(cases) >= limit:
            break
        if spec.holdout_fraction and _held_out(spec, row) != (split == "test"):
            continue
        if not ratings or any(len(r) < max(spec.min_raters, 1) for r in ratings.values()):
            continue
        expected = {}
        for name, given in ratings.items():
            counts = [given.count(level) for level in range(_levels(spec, name))]
            # One annotator per case and question, fixed by the case: the
            # outcome the gates score, drawn so a rerun draws the same one.
            draw = int.from_bytes(
                hashlib.blake2b(f"{spec.name}/{i}/{name}".encode(), digest_size=4).digest(),
                "big",
            )
            expected[name] = Expectation(
                label=given[draw % len(given)],
                distribution=tuple(c / len(given) for c in counts),
            )
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


def _score_cases(spec: CorpusSpec, split: str, path: pathlib.Path, limit: int | None) -> list[Case]:
    """Several ordered ratings over one piece of state, each its own question.

    One request, several Score questions, which is exactly the shape the
    independence claim is about -- and the first time it is exercised on real
    data rather than on the generator.
    """
    questions = _questions(spec)
    cases: list[Case] = []
    for i, row in enumerate(_records(path)):
        if limit is not None and len(cases) >= limit:
            break
        if spec.holdout_fraction and _held_out(spec, row) != (split == "test"):
            continue
        expected = {}
        for name in spec.score_fields:
            value = row.get(name)
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
        if spec.share_alike:
            raise CorpusLicenceError(
                f"{spec.name} is {spec.licence}, a share-alike licence, and may not be "
                f"used to {purpose}: share-alike corpora evaluate and never train "
                "(docs/decisions.md Q17)"
            )
        raise CorpusLicenceError(
            f"{spec.name} is {spec.tier} ({spec.licence}) and may not be used to "
            f"{purpose}; docs/decisions.md section 3 has the policy"
        )
    paths = fetch(spec, root=root)
    if spec.holdout_fraction:
        # One pool, split here by a hash of the grouping text; see
        # `holdout_fraction`. Several files are one pool read in order.
        if split not in ("train", "test"):
            raise KeyError(f"{spec.name} has splits 'train' and 'test'; got {split!r}")
        if spec.annotator_lists:
            return _annotated_cases(spec, split, [paths[k] for k in spec.files], limit)
        paths = {split: paths["all"]}
    if split not in paths:
        raise KeyError(f"{spec.name} has no split {split!r}; it has {sorted(paths)}")
    if spec.annotator_lists:
        return _annotated_cases(spec, split, [paths[split]], limit)
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
