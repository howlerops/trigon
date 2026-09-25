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


# -- Share-alike: evaluates, never trains (docs/decisions.md Q17) -------------


def test_a_share_alike_corpus_evaluates_and_never_trains(fake, monkeypatch):
    """The owner's decision of 2026-09-25, enforced on the licence itself.

    Built as amber here, the tier a share-alike corpus must have; the next
    test shows the licence refuses training even where a tier would not.
    """
    share_alike = dataclasses.replace(CORPORA["banking77"], tier="amber", licence="CC BY-SA 3.0")
    monkeypatch.setitem(CORPORA, "banking77", share_alike)
    assert load("banking77", "test", purpose="eval", root=fake)
    with pytest.raises(CorpusLicenceError, match="share-alike"):
        load("banking77", "train", purpose="train", root=fake)
    with pytest.raises(CorpusLicenceError, match="share-alike"):
        load("banking77", "train", purpose="redistribute", root=fake)


def test_a_share_alike_corpus_cannot_be_declared_green():
    """A tier is a field somebody types. The licence is what Q17 is about."""
    with pytest.raises(ValueError, match="share-alike"):
        dataclasses.replace(CORPORA["banking77"], licence="CC BY-SA 4.0")


def test_share_alike_is_read_off_the_licence_whatever_the_version():
    from trigon.evals.corpora import is_share_alike

    for licence in ("CC BY-SA 3.0", "CC BY-SA 4.0", "cc-by-sa-4.0", "CC BY-SA 3.0 + GFDL"):
        assert is_share_alike(licence), licence
    for licence in ("CC BY 4.0", "CC BY 3.0", "Apache-2.0", "CC0-1.0", "MIT"):
        assert not is_share_alike(licence), licence


def test_circa_is_share_alike_and_refuses_training():
    """Its README names CC BY 4.0 and links the BY-SA 4.0 text; the stricter binds."""
    circa = corpus("circa")
    assert circa.share_alike and circa.tier == "amber"
    assert circa.permits("eval")
    assert not circa.permits("train")
    with pytest.raises(CorpusLicenceError, match="share-alike"):
        load("circa", "train", purpose="train")


def test_no_share_alike_row_in_the_licence_audit_is_green():
    """BoolQ and FEVER are rows in docs/data.md, not loaders, and the rule is theirs too.

    Every audit row whose licence column names a share-alike licence must be
    amber or stricter, so a share-alike corpus cannot be cleared to train by
    editing a table.
    """
    from trigon.evals.corpora import is_share_alike

    audit = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "data.md").read_text()
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in audit.splitlines()
        if line.startswith("| ") and line.count("|") >= 6
    ]
    header = next(r for r in rows if r[:2] == ["Dataset", "Primitive"])
    licence, tier = header.index("Licence found"), header.index("Tier")
    share_alike = [r for r in rows if r is not header and is_share_alike(r[licence])]
    names = {r[0] for r in share_alike}
    assert {"BoolQ", "FEVER", "Circa"} <= names, names
    for row in share_alike:
        assert "**green**" not in row[tier], f"{row[0]} is share-alike and green: {row}"


# -- Rater-row and per-item annotator corpora ----------------------------------


def test_verify_refuses_a_download_that_is_not_the_pinned_content():
    from trigon.evals.corpora import verify

    with pytest.raises(ValueError, match="pinned sha256"):
        verify(corpus("circa"), "all", b"not the circa data")


def test_a_parquet_corpus_asks_for_its_conversion_rather_than_fetching(tmp_path):
    """The loader never downloads a format it cannot read."""
    from trigon.evals.corpora import CorpusNotConverted

    with pytest.raises(CorpusNotConverted, match="convert_corpus.py measuring_hate_speech"):
        load("measuring_hate_speech", "train", purpose="train", root=tmp_path)


GOEMOTIONS_HEADER = (
    "text,id,author,subreddit,link_id,parent_id,created_utc,rater_id,example_very_unclear,"
    "admiration,amusement,anger,annoyance,approval,caring,confusion,curiosity,desire,"
    "disappointment,disapproval,disgust,embarrassment,excitement,fear,gratitude,grief,joy,"
    "love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral"
)
_EMOTIONS = GOEMOTIONS_HEADER.split(",")[9:]


def _goemotions_row(text, cid, rater, marked=(), unclear=False):
    flags = ",".join("1" if e in marked else "0" for e in _EMOTIONS)
    return f'"{text}",{cid},a,s,l,p,0.0,{rater},{unclear},{flags}'


@pytest.fixture
def goemotions(tmp_path: pathlib.Path) -> pathlib.Path:
    """Three part files, one row per rater, a comment's raters split across parts."""
    root = tmp_path / "cache"
    (root / "goemotions").mkdir(parents=True)
    parts: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for c in range(60):
        text = f"comment {c % 50}"  # ten texts posted under two ids each
        parts[1].append(_goemotions_row(text, f"id{c}", 1, marked=("annoyance",)))
        parts[2].append(_goemotions_row(text, f"id{c}", 2, marked=("joy", "anger")))
        parts[3].append(_goemotions_row(text, f"id{c}", 3, unclear=True))
    # One usable rater and one abstention: below min_raters, dropped.
    parts[1].append(_goemotions_row("lonely", "solo", 1, marked=("fear",)))
    parts[2].append(_goemotions_row("lonely", "solo", 2, unclear=True))
    for i, lines in parts.items():
        (root / "goemotions" / f"part{i}.csv").write_text(
            "\n".join([GOEMOTIONS_HEADER, *lines]) + "\n"
        )
    return root


def test_goemotions_groups_raters_into_one_noul_per_ekman_group(goemotions):
    cases = load("goemotions", "train", purpose="train", root=goemotions)
    cases += load("goemotions", "test", purpose="eval", root=goemotions)
    assert len(cases) == 60, "every comment once, the one-rater comment dropped"
    assert list(cases[0].request.questions) == [
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "neutral",
    ]
    assert all(q.type == "noul" for q in cases[0].request.questions.values())
    for case in cases:
        # Rater 1 marked annoyance (anger's group), rater 2 joy and anger;
        # rater 3 abstained, which is not a "no" to everything.
        assert case.expected["anger"].distribution == (0.0, 1.0)
        assert case.expected["joy"].distribution == (0.5, 0.5)
        assert case.expected["fear"].distribution == (1.0, 0.0)
        assert case.expected["joy"].label in (0, 1)


def test_goemotions_holds_out_by_text_so_a_duplicate_never_straddles(goemotions):
    train = load("goemotions", "train", purpose="train", root=goemotions)
    test = load("goemotions", "test", purpose="eval", root=goemotions)
    assert train and test
    assert not {c.request.state for c in train} & {c.request.state for c in test}


def _jsonl_gz(path: pathlib.Path, rows: list[dict]) -> None:
    import gzip
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


_MHS = (
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
)


@pytest.fixture
def hate_speech(tmp_path: pathlib.Path) -> pathlib.Path:
    """Converted rows as scripts/convert_corpus.py writes them: floats, one per rater."""
    root = tmp_path / "cache"
    rows = []
    for c in range(80):
        for rater, level in ((10, 1.0), (11, 3.0), (12, 3.0)):
            rows.append(
                {
                    "comment_id": c,
                    "text": f"post {c}",
                    "annotator_id": rater,
                    **dict.fromkeys(_MHS, level),
                    "hatespeech": 2.0 if level == 3.0 else 0.0,
                }
            )
    # A rating outside hatespeech's three levels: that rater is dropped whole.
    rows.append({**rows[0], "annotator_id": 13, "hatespeech": 3.0})
    # A comment rated once is one answer, not a distribution: dropped.
    rows.append({**rows[0], "comment_id": 999, "text": "rated once"})
    _jsonl_gz(root / "measuring_hate_speech" / "all.jsonl.gz", rows)
    return root


def test_measuring_hate_speech_is_ten_score_items_per_annotator(hate_speech):
    cases = load("measuring_hate_speech", "train", purpose="train", root=hate_speech)
    cases += load("measuring_hate_speech", "test", purpose="eval", root=hate_speech)
    assert len(cases) == 80
    questions = cases[0].request.questions
    assert list(questions) == [*_MHS, "hatespeech"]
    assert [lv.name for lv in questions["sentiment"].levels] == ["0", "1", "2", "3", "4"]
    assert [lv.name for lv in questions["hatespeech"].levels] == ["0", "1", "2"]
    # The instructions say which end is which: a level named "3" means nothing alone.
    assert "4 strongly" in questions["violence"].instructions
    for case in cases:
        assert case.expected["insult"].distribution == pytest.approx((0, 1 / 3, 0, 2 / 3, 0))
        assert case.expected["hatespeech"].distribution == pytest.approx((1 / 3, 0, 2 / 3))


def test_measuring_hate_speech_keeps_every_rater_of_a_comment_together(hate_speech):
    train = load("measuring_hate_speech", "train", purpose="train", root=hate_speech)
    test = load("measuring_hate_speech", "test", purpose="eval", root=hate_speech)
    assert train and test
    assert not {c.request.state for c in train} & {c.request.state for c in test}
    ids = [c.case_id.rsplit("/", 1)[1] for c in train + test]
    assert len(ids) == len(set(ids)), "a comment became two cases"


CIRCA_HEADER = "\t".join(
    [
        "id",
        "context",
        "question-X",
        "canquestion-X",
        "answer-Y",
        "judgements",
        "goldstandard1",
        "goldstandard2",
    ]
)


@pytest.fixture
def circa(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "cache"
    (root / "circa").mkdir(parents=True)
    lines = [CIRCA_HEADER]
    for i in range(90):
        judgements = "Yes#Yes#No#Probably no#Other" if i % 3 else "No#No#No#No"
        lines.append(
            f"{i}\tX wants to know about Y's food preferences.\tDo you like dish {i % 30}?\t"
            f"I like dish {i % 30}.\tanswer {i}\t{judgements}\tYes\tYes"
        )
    # A judgement outside the declared set drops the pair; never mapped to a neighbour.
    lines.append("90\tctx\tDo you swim?\tI swim.\tsometimes\tYes#Maybe-ish#No\tNA\tNA")
    (root / "circa" / "all.tsv").write_text("\n".join(lines) + "\n")
    return root


def test_circa_is_one_choice_over_eight_interpretations(circa):
    cases = load("circa", "train", purpose="eval", root=circa)
    cases += load("circa", "test", purpose="eval", root=circa)
    assert len(cases) == 90
    question = cases[0].request.questions["interpretation"]
    assert question.names[0] == "Yes" and question.names[-1] == "Other"
    assert len(question.names) == 8
    by_answer = {c.request.state.rsplit("\n", 1)[1]: c for c in cases}
    mixed = by_answer["answer 1"].expected["interpretation"]
    # Yes, Yes, No, Probably no, Other -- in the declared order.
    assert mixed.distribution == pytest.approx((0.4, 0, 0, 0.2, 0.2, 0, 0, 0.2))
    four = by_answer["answer 0"].expected["interpretation"]
    assert four.distribution[3] == 1.0 and four.label == 3
    assert "QUESTION-X:" in cases[0].request.state and "ANSWER-Y:" in cases[0].request.state


def test_circa_holds_out_by_question_so_its_answers_stay_together(circa):
    """Each question has several answers; a pair-level split tests on seen questions."""
    train = load("circa", "train", purpose="eval", root=circa)
    test = load("circa", "test", purpose="eval", root=circa)
    assert train and test

    def questions(cases):
        return {c.request.state.split("QUESTION-X:\n")[1].split("\n")[0] for c in cases}

    assert not questions(train) & questions(test)
