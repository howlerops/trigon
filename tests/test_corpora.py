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
