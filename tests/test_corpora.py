"""The licence policy, enforced rather than documented.

`docs/data.md` cleared nine corpora and `docs/decisions.md` §3 says what each
tier permits. Until now that was a table in a document, and a table in a
document does not stop an amber corpus ending up in a training mix -- a
mistake you cannot find afterwards by reading the weights.

These tests never touch the network. A corpus is written to a temp directory
and read back, so the loader's behaviour is tested without making CI depend on
somebody else's raw.githubusercontent.com.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from trigon.evals.corpora import (
    CORPORA,
    CorpusLicenceError,
    CorpusSpec,
    corpus,
    load,
)

TRAIN = "text,category\nmy card has not arrived,card_arrival\nwhere is my refund,refund_status\n"
TEST = "text,category\nthe atm ate my card,card_swallowed\nrefund still missing,refund_status\n"


@pytest.fixture
def fake(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    """A corpus already in the cache, so nothing is fetched."""
    root = tmp_path / "cache"
    (root / "banking77").mkdir(parents=True)
    (root / "banking77" / "train.csv").write_text(TRAIN)
    (root / "banking77" / "test.csv").write_text(TEST)
    return root


def test_the_option_set_is_every_label_in_the_corpus_not_this_split(fake):
    """A model choosing between 70 options on train and 77 on test is being
    asked two different questions, and the second is harder for a reason that
    has nothing to do with the model."""
    train = load("banking77", "train", purpose="train", root=fake)
    test = load("banking77", "test", purpose="eval", root=fake)
    train_options = [o.name for o in train[0].request.questions["intent"].options]
    test_options = [o.name for o in test[0].request.questions["intent"].options]
    assert train_options == test_options
    assert train_options == ["card arrival", "card swallowed", "refund status"]


def test_labels_index_into_the_declared_option_order(fake):
    """An off-by-one here trains the model on the wrong answer and every
    downstream number stays plausible."""
    cases = load("banking77", "train", purpose="train", root=fake)
    for case in cases:
        options = [o.name for o in case.request.questions["intent"].options]
        label = case.expected["intent"].label
        assert 0 <= label < len(options)
    first, second = cases
    assert first.request.state == "my card has not arrived"
    assert first.expected["intent"].label == 0  # card arrival
    assert second.expected["intent"].label == 2  # refund status


def test_a_red_corpus_may_not_be_loaded_for_anything(fake, monkeypatch):
    red = dataclasses.replace(CORPORA["banking77"], tier="red", licence="no stated licence")
    monkeypatch.setitem(CORPORA, "banking77", red)
    for purpose in ("train", "eval", "redistribute"):
        with pytest.raises(CorpusLicenceError, match="red"):
            load("banking77", "train", purpose=purpose, root=fake)


def test_an_amber_corpus_evals_but_never_trains(fake, monkeypatch):
    """The expensive mistake this module exists to prevent."""
    amber = dataclasses.replace(CORPORA["banking77"], tier="amber", licence="CC BY-SA 4.0")
    monkeypatch.setitem(CORPORA, "banking77", amber)
    assert load("banking77", "test", purpose="eval", root=fake)
    with pytest.raises(CorpusLicenceError, match="may not be used to train"):
        load("banking77", "train", purpose="train", root=fake)
    with pytest.raises(CorpusLicenceError):
        load("banking77", "train", purpose="redistribute", root=fake)


def test_purpose_has_no_default():
    """A default is the argument the caller least often thinks about, and the
    mistake here is exactly the one a default would hide."""
    with pytest.raises(TypeError):
        load("banking77", "train")  # type: ignore[call-arg]


def test_every_committed_corpus_carries_the_attribution_its_licence_requires():
    for name, spec in CORPORA.items():
        assert isinstance(spec, CorpusSpec)
        assert spec.attribution.strip(), f"{name} has no attribution"
        assert spec.licence in spec.attribution, (
            f"{name}'s attribution does not name its licence, so a report "
            "carrying it would not carry the credit"
        )
        assert spec.tier in {"green", "amber", "red"}


def test_the_committed_tiers_match_the_licence_audit():
    """One source of truth: docs/data.md is the audit, this is the code.

    They are separate files and they will drift; this is the test that fails
    when they do, the same way test_docs_drift.py pins the budgets.
    """
    audit = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "data.md").read_text()
    for name, spec in CORPORA.items():
        row = [line for line in audit.splitlines() if line.lower().startswith(f"| {name}")]
        assert row, f"{name} is loadable but absent from the licence audit in docs/data.md"
        assert f"**{spec.tier}**" in row[0], (
            f"{name} is {spec.tier} in code and the audit row says otherwise: {row[0]}"
        )


def test_an_unknown_corpus_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="banking77"):
        corpus("definitely_not_a_corpus")


# -- Score corpora: several ordered ratings over one piece of state ----------

HS2 = [
    {
        "prompt": "why is the sky blue",
        "response": "Rayleigh scattering.",
        "helpfulness": 3,
        "correctness": 4,
        "coherence": 4,
        "complexity": 2,
        "verbosity": 0,
    },
    {
        "prompt": "why is the sky blue",
        "response": "Because.",
        "helpfulness": 0,
        "correctness": 1,
        "coherence": 2,
        "complexity": 0,
        "verbosity": 0,
    },
    # A rating outside the declared range. Never clipped into it: a silently
    # clamped label trains the model on an answer nobody gave.
    {
        "prompt": "x",
        "response": "y",
        "helpfulness": 9,
        "correctness": 1,
        "coherence": 1,
        "complexity": 1,
        "verbosity": 1,
    },
    # A missing rating. Same treatment, for the same reason.
    {"prompt": "x", "response": "y", "helpfulness": 2, "correctness": 2, "coherence": 2},
]


@pytest.fixture
def scored(tmp_path: pathlib.Path) -> pathlib.Path:
    """A gzipped-JSONL corpus in the cache, so nothing is fetched."""
    import gzip
    import json

    root = tmp_path / "cache"
    (root / "helpsteer2").mkdir(parents=True)
    for split in ("train", "test"):
        with gzip.open(root / "helpsteer2" / f"{split}.jsonl.gz", "wt") as handle:
            for row in HS2:
                handle.write(json.dumps(row) + "\n")
    return root


def test_a_score_corpus_asks_one_question_per_rating(scored):
    cases = load("helpsteer2", "test", purpose="eval", root=scored)
    questions = cases[0].request.questions
    assert list(questions) == [
        "helpfulness",
        "correctness",
        "coherence",
        "complexity",
        "verbosity",
    ]
    assert [level.name for level in questions["helpfulness"].levels] == ["0", "1", "2", "3", "4"]
    # Anchored at their own indices, so the reported score is on the corpus's
    # scale rather than on a positional one that happens to match.
    assert [level.value for level in questions["helpfulness"].levels] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert cases[0].expected["helpfulness"].label == 3
    assert cases[1].expected["verbosity"].label == 0


def test_each_question_gets_its_own_instructions(scored):
    """Five questions sharing one instruction string would make five
    identical schema blocks and ask the model to tell them apart by position.
    """
    questions = load("helpsteer2", "test", purpose="eval", root=scored)[0].request.questions
    texts = {q.instructions for q in questions.values()}
    assert len(texts) == len(questions)
    assert "verbose" in questions["verbosity"].instructions.lower()


def test_a_rating_outside_the_declared_levels_drops_the_row(scored):
    """Not clipped. A clamped label is an answer nobody gave, and it would
    train the model on it while every count still looked right."""
    cases = load("helpsteer2", "test", purpose="eval", root=scored)
    assert len(cases) == 2, "the out-of-range row and the incomplete row must both be dropped"
    for case in cases:
        for expectation in case.expected.values():
            assert 0 <= expectation.label <= 4


def test_the_state_labels_which_field_is_which(scored):
    """A prompt and a response concatenated bare are distinguishable only by
    position, which is a thing the model would have to learn rather than be
    told."""
    state = load("helpsteer2", "test", purpose="eval", root=scored)[0].request.state
    assert "PROMPT:" in state and "RESPONSE:" in state
    assert state.index("PROMPT:") < state.index("RESPONSE:")


def test_gzipped_jsonl_and_csv_both_load_without_a_compiled_dependency():
    """The dependency rule, checked rather than asserted.

    `trigon.evals.corpora` is imported by the drift tests and by the gateway's
    budgeting path, neither of which has a GPU stack. A Parquet reader here
    would put `pyarrow` in all of them.
    """
    import ast

    import trigon.evals.corpora as module

    # Parsed, not grepped. The first version searched the source text and
    # failed on "task-specific-datasets" inside a URL and on the word
    # `pyarrow` inside the comment explaining why it is absent -- a test
    # tripping over the documentation of the rule it is enforcing.
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    for banned in ("pyarrow", "pandas", "datasets", "torch", "numpy"):
        assert banned not in imported, f"{banned} must not be imported by the corpus loader"


# -- Annotator distributions: every rating, not their average -----------------


@pytest.fixture
def annotated(tmp_path: pathlib.Path) -> pathlib.Path:
    import gzip
    import json

    root = tmp_path / "cache"
    (root / "helpsteer2-annotators").mkdir(parents=True)
    rows = []
    for p in range(40):
        for r in range(3):  # several responses per prompt, as the real split has
            rows.append(
                {
                    "prompt": f"prompt {p}" + (" " if r == 2 else ""),  # whitespace variant
                    "response": f"response {r}",
                    **{
                        q: [r, (r + 1) % 5, 4]
                        for q in ("helpfulness", "correctness", "coherence", "complexity")
                    },
                    "verbosity": [1, 1],
                }
            )
    with gzip.open(root / "helpsteer2-annotators" / "all.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return root


def test_annotator_ratings_become_a_distribution_and_one_drawn_label(annotated):
    cases = load("helpsteer2-annotators", "train", purpose="train", root=annotated)
    cases += load("helpsteer2-annotators", "test", purpose="eval", root=annotated)
    for case in cases:
        verbosity = case.expected["verbosity"]
        assert verbosity.distribution == (0.0, 1.0, 0.0, 0.0, 0.0)
        assert verbosity.label == 1
        helpful = case.expected["helpfulness"]
        assert sum(helpful.distribution) == pytest.approx(1.0)
        # The label is one annotator's rating, never a level nobody chose.
        assert helpful.distribution[helpful.label] > 0


def test_the_split_is_by_prompt_so_no_prompt_is_on_both_sides(annotated):
    """Several responses share a prompt; a row split would test on trained prompts."""
    train = load("helpsteer2-annotators", "train", purpose="train", root=annotated)
    test = load("helpsteer2-annotators", "test", purpose="eval", root=annotated)
    assert train and test
    assert len(train) + len(test) == 120

    def prompts(cases):
        return {c.request.state.split("RESPONSE:")[0].strip() for c in cases}

    assert not prompts(train) & prompts(test)


def test_the_drawn_label_is_the_same_on_every_load(annotated):
    first = load("helpsteer2-annotators", "test", purpose="eval", root=annotated)
    again = load("helpsteer2-annotators", "test", purpose="eval", root=annotated)
    assert [c.expected["helpfulness"].label for c in first] == [
        c.expected["helpfulness"].label for c in again
    ]
