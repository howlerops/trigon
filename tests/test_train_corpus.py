"""The corpus training script's splits, which are the part that can lie.

A calibration number is only as good as the data it was measured on, and
every way of getting that wrong here produces a report that looks fine:
a calibrator fitted on the evaluation split reports its own fit, an
evaluation split that overlaps training reports memorisation, and an
evaluation split below `MIN_CALIBRATION_SAMPLES` reports estimator noise.

The last one is not hypothetical. The first run of this script on Banking77
failed `sample_size` and `gate_is_testable` on every seed: the corpus's test
split is 3,080 rows against a floor of 5,000, and a perfectly calibrated
model scored ECE 0.0221 on the 1,500 used — most of the 0.05 gate. The gate
was not a test, so the run was fixed rather than the gate.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from trigon.limits import MIN_CALIBRATION_SAMPLES

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module():
    spec = importlib.util.spec_from_file_location(
        "train_corpus", ROOT / "scripts" / "train_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _module()


@pytest.fixture
def corpus_on_disk(tmp_path, monkeypatch):
    """A corpus whose test split is deliberately below the floor."""
    root = tmp_path / "cache"
    (root / "banking77").mkdir(parents=True)

    def rows(n, offset):
        lines = ["text,category"]
        for i in range(n):
            lines.append(f"utterance number {i + offset},label_{i % 7}")
        return "\n".join(lines) + "\n"

    (root / "banking77" / "train.csv").write_text(rows(4_000, 0))
    (root / "banking77" / "test.csv").write_text(rows(1_200, 100_000))
    monkeypatch.setenv("TRIGON_CORPUS_CACHE", str(root))
    return root


def test_the_three_splits_never_overlap(script, corpus_on_disk):
    """Each of these overlaps produces a report that looks entirely normal."""
    from trigon.evals.corpora import corpus

    args = script.parse_args(["banking77", "--calibration-n", "500"])
    train, calibration, evaluation, _ = script.splits(corpus("banking77"), args)
    ids = [{case.case_id for case in split} for split in (train, calibration, evaluation)]
    assert ids[0] and ids[1] and ids[2]
    assert not ids[0] & ids[1], "training and calibration overlap"
    assert not ids[0] & ids[2], "training and evaluation overlap"
    assert not ids[1] & ids[2], "calibration and evaluation overlap"


def test_the_evaluation_split_is_topped_up_toward_the_floor(script, corpus_on_disk):
    """Below the floor, ECE is estimator noise and the gate is not a test."""
    from trigon.evals.corpora import corpus

    args = script.parse_args(["banking77", "--calibration-n", "500"])
    _, _, evaluation, topped_up = script.splits(corpus("banking77"), args)
    assert topped_up > 0, "a 1,200-row test split cannot reach the floor on its own"
    assert len(evaluation) == 1_200 + topped_up


def test_the_top_up_never_eats_the_training_set(script, corpus_on_disk):
    """The defect this test was written after finding.

    The first version capped the top-up at `len(remaining) - 1`, so on a
    corpus whose train split is small it reached the floor by leaving **one**
    training case. Every count in the report was correct and the report was
    worthless. When half the pool is not enough to reach the floor, the floor
    is not reached and `sample_size` fails -- which is the right answer, since
    cannibalising training to make a gate pass is the same move as widening
    it.
    """
    from trigon.evals.corpora import corpus

    args = script.parse_args(["banking77", "--calibration-n", "500"])
    train, _, evaluation, topped_up = script.splits(corpus("banking77"), args)
    assert len(train) >= topped_up, "the top-up took more than it left to train on"
    assert len(train) > 1_000
    # This corpus is deliberately too small to reach the floor without doing
    # that, so it does not reach it, and the gate says so rather than passing.
    assert len(evaluation) < MIN_CALIBRATION_SAMPLES


def test_a_corpus_large_enough_does_reach_the_floor(script, tmp_path, monkeypatch):
    """The case the real Banking77 is: 3,080 test rows and 10,003 to draw on."""
    from trigon.evals.corpora import corpus

    root = tmp_path / "cache"
    (root / "banking77").mkdir(parents=True)

    def rows(n, offset):
        lines = ["text,category"]
        for i in range(n):
            lines.append(f"utterance number {i + offset},label_{i % 7}")
        return "\n".join(lines) + "\n"

    (root / "banking77" / "train.csv").write_text(rows(10_003, 0))
    (root / "banking77" / "test.csv").write_text(rows(3_080, 100_000))
    monkeypatch.setenv("TRIGON_CORPUS_CACHE", str(root))

    args = script.parse_args(["banking77", "--calibration-n", "1000", "-n", "0"])
    train, _, evaluation, topped_up = script.splits(corpus("banking77"), args)
    assert len(evaluation) >= MIN_CALIBRATION_SAMPLES
    assert topped_up == MIN_CALIBRATION_SAMPLES - 3_080
    assert len(train) > len(evaluation), "training must still be the larger half"


def test_an_explicit_eval_n_is_honoured_and_not_topped_up(script, corpus_on_disk):
    """The top-up is the default, not a policy. A run deliberately measuring
    something at a smaller size must still be able to."""
    from trigon.evals.corpora import corpus

    args = script.parse_args(["banking77", "--calibration-n", "500", "--eval-n", "300"])
    _, _, evaluation, topped_up = script.splits(corpus("banking77"), args)
    assert len(evaluation) == 300
    assert topped_up == 0


def test_the_report_states_where_the_evaluation_rows_came_from(script, corpus_on_disk):
    """Topped-up rows come from the train distribution, which is a real
    difference from the corpus's own test split. Summing them into one number
    would hide it."""
    from trigon.evals.corpora import corpus

    spec = corpus("banking77")
    args = script.parse_args(["banking77", "--calibration-n", "500"])
    train, calibration, evaluation, topped_up = script.splits(spec, args)
    text = script.header(
        spec,
        args,
        train,
        calibration,
        evaluation,
        script.baseline_accuracy(train, evaluation),
        topped_up,
    )
    assert "from the corpus's own test split" in text
    assert "held out of train to reach the floor" in text
    assert f"{topped_up:,}" in text
    assert spec.attribution in text, "the licence's attribution must reach the report"


def test_the_marginal_baseline_is_per_question(script, corpus_on_disk):
    """Pooling lets a well-predicted question carry a badly predicted one --
    the same cancellation that made pooled ECE misleading here."""
    from trigon.evals.corpora import corpus

    args = script.parse_args(["banking77", "--calibration-n", "500"])
    train, _, evaluation, _ = script.splits(corpus("banking77"), args)
    marginal = script.baseline_accuracy(train, evaluation)
    assert set(marginal) == {"intent"}
    assert 0.0 <= marginal["intent"] <= 1.0


def test_the_baseline_is_fitted_on_train_not_on_evaluation(script, corpus_on_disk):
    """A baseline fitted on the split it is scored against is not a baseline,
    it is an oracle with one degree of freedom."""
    from trigon.evals.harness import Case, Expectation
    from trigon.types import ChoiceQuestion, SystemOneRequest

    def case(cid, label):
        return Case(
            case_id=cid,
            request=SystemOneRequest(
                state="x",
                questions={
                    "q": ChoiceQuestion(
                        instructions="Pick.", options=[{"name": "a"}, {"name": "b"}]
                    )
                },
            ),
            expected={"q": Expectation(label=label)},
        )

    # Train is overwhelmingly label 0; evaluation is overwhelmingly label 1.
    train = [case(f"t{i}", 0) for i in range(9)] + [case("t9", 1)]
    evaluation = [case(f"e{i}", 1) for i in range(9)] + [case("e9", 0)]
    # Predicting train's modal label (0) scores 0.1 on evaluation. An oracle
    # fitted on evaluation would report 0.9.
    assert script.baseline_accuracy(train, evaluation)["q"] == pytest.approx(0.1)


def test_a_named_cuda_is_refused_rather_than_replaced_by_the_cpu(script):
    """The Modal job asks for cuda by name so a missing GPU fails the run.

    The launcher once recorded the A10G it was given while every tensor stayed
    on the CPU. A fallback here would reproduce that with a different cause:
    a report naming a GPU that did no work.
    """
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine has the GPU the test needs to be missing")
    with pytest.raises(SystemExit, match="no CUDA device"):
        script.resolve_device("cuda")
    assert script.resolve_device("auto")[0] == "cpu"


def test_the_hard_label_ablation_trains_on_the_majority_and_nothing_else(script):
    from trigon.evals.harness import Case, Expectation
    from trigon.types import ScoreQuestion, SystemOneRequest

    request = SystemOneRequest(
        state="s",
        questions={
            "q": ScoreQuestion(instructions="Rate.", levels=[{"name": str(i)} for i in range(3)])
        },
    )
    tied = Case("c", request, {"q": Expectation(label=2, distribution=(0.5, 0.0, 0.5))})
    plain = Case("d", request, {"q": Expectation(label=1)})
    rewritten = script.majority_labels([tied, plain])
    # A tie goes to the lower rating, and the drawn label is dropped.
    assert rewritten[0].expected["q"].label == 0
    assert rewritten[0].expected["q"].distribution is None
    # A case without a distribution is left exactly as it was.
    assert rewritten[1] == plain
