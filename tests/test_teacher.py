"""The teacher-labelled stream: loaded as soft targets, never reported as calibration.

`docs/data.md` says a teacher's distributions buy coverage and not
calibration, because teachers are overconfident. These tests hold the code to
that: the loader marks every label as the teacher's, the trainer fits the
distribution rather than the argmax, and every path a calibration number or a
calibrator comes out of refuses those labels -- while the one function that
scores them says, in its output, that it measured agreement.

No network and no GPU. Records are built here in the shape
`scripts/modal_teacher.py` writes, and the committed fixture is a real sample
of the published build.
"""

from __future__ import annotations

import gzip
import json
import math
import pathlib

import pytest

from trigon.backends.lexical import LexicalBackend
from trigon.engine import Engine
from trigon.evals import run_calibration_suite, run_cases, summarize
from trigon.evals.corpora import (
    CORPORA,
    GENERATED,
    TEACHER_WORKFLOWS,
    CorpusNotFetched,
    corpus,
    load,
)
from trigon.evals.harness import Case, Expectation, TeacherLabelsAreNotCalibration
from trigon.evals.teacher import (
    DOMAINS,
    GENERATION_TEMPLATE_SHA256,
    LABEL_TEMPLATE_SHA256,
    TEACHER_MODEL,
    TEACHER_REVISION,
    Rejected,
    case_from_record,
    generation_messages,
    generation_plan,
    label_messages,
    option_continuations,
    parse_generated,
    plan_case,
    softmax,
    teacher_agreement,
)
from trigon.limits import MIN_CALIBRATION_SAMPLES
from trigon.types import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "teacher_workflows_sample.jsonl"


def _record(i: int, *, yes: float = 0.8, pick: int = 1) -> dict:
    """A record in the stored shape, with a Choice, a Noul and a Score."""
    choice = [math.log(0.1), math.log(0.1), math.log(0.1)]
    choice[pick] = math.log(0.7)
    return {
        "case_id": f"tw0-{i:06d}",
        "domain": DOMAINS[i % len(DOMAINS)].name,
        "state": {"ticket": f"order {i} arrived broken", "amount": 20 + i},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "options": [{"name": "billing"}, {"name": "shipping"}, {"name": "hardware"}],
            },
            "refund": {"type": "noul", "instructions": "Does the customer want money back?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "levels": [{"name": "low"}, {"name": "medium"}, {"name": "high"}],
            },
        },
        "labels": {
            "team": {"options": ["billing", "shipping", "hardware"], "logprobs": choice},
            "refund": {"options": ["no", "yes"], "logprobs": [math.log(1 - yes), math.log(yes)]},
            "urgency": {
                "options": ["low", "medium", "high"],
                "logprobs": [math.log(0.2), math.log(0.5), math.log(0.3)],
            },
        },
        "teacher": {"model": TEACHER_MODEL, "revision": TEACHER_REVISION},
    }


def _write(root: pathlib.Path, records: list[dict]) -> pathlib.Path:
    folder = root / TEACHER_WORKFLOWS.name
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / "cases.jsonl.gz", "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root


# -- the licence, in code -----------------------------------------------------


def test_the_teacher_is_the_apache_licensed_size_at_a_pinned_revision():
    """The 3B and 72B instruct models are under the Qwen licence; 7B is Apache-2.0."""
    assert TEACHER_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    assert len(TEACHER_REVISION) == 40 and int(TEACHER_REVISION, 16) >= 0
    spec = corpus("teacher-workflows")
    assert spec is TEACHER_WORKFLOWS
    assert spec.licence == "Apache-2.0" and spec.tier == "green"
    assert spec.permits("train") and spec.permits("eval")
    assert spec.teacher == f"{TEACHER_MODEL}@{TEACHER_REVISION}"
    # The credit travels with every report, and names the licence and revision.
    assert spec.licence in spec.attribution
    assert TEACHER_REVISION[:12] in spec.attribution
    assert "not calibration" in spec.attribution


def test_a_generated_stream_is_not_counted_as_a_real_corpus():
    """The ledger counts real corpora; a teacher's output is not somebody's traffic."""
    assert "teacher-workflows" in GENERATED
    assert "teacher-workflows" not in CORPORA


def test_only_the_teacher_stream_is_not_calibration_evidence():
    assert not TEACHER_WORKFLOWS.calibration_evidence
    assert all(spec.calibration_evidence for spec in CORPORA.values())


def test_the_audit_row_says_green_and_names_the_teacher():
    audit = (ROOT / "docs" / "data.md").read_text()
    row = [line for line in audit.splitlines() if line.startswith("| teacher-workflows")]
    assert row and "**green**" in row[0]
    assert "Qwen2.5-7B-Instruct" in row[0] and TEACHER_REVISION[:7] in row[0]


# -- generation: planned, templated, validated ----------------------------------


def test_twenty_domains_rotate_through_the_plan():
    assert len(DOMAINS) == 20 and len({d.name for d in DOMAINS}) == 20
    plans = generation_plan(40)
    assert {p.domain for p in plans} == {d.name for d in DOMAINS}


def test_a_plan_depends_on_its_seed_and_index_alone():
    """Extending a build never changes the cases already in it."""
    assert generation_plan(50)[:5] == generation_plan(5)
    assert generation_plan(3, start=7) == [plan_case(0, 7), plan_case(0, 8), plan_case(0, 9)]
    assert plan_case(0, 3) != plan_case(1, 3)


def test_the_plan_varies_the_schema_as_data_md_asks():
    """Option counts from 2 to 12, one to six questions, all three primitives."""
    plans = generation_plan(600)
    counts = {c for p in plans for k, c in p.questions if k == "choice"}
    assert min(counts) == 2 and max(counts) == 12
    assert {len(p.questions) for p in plans} == {1, 2, 3, 4, 5, 6}
    assert {k for p in plans for k, _ in p.questions} == {"choice", "noul", "score"}
    assert {p.state_format for p in plans} == {"prose", "record", "documents"}
    assert 0.3 < sum(p.borderline for p in plans) / len(plans) < 0.5


def test_the_generation_prompt_carries_the_plan():
    plan = plan_case(0, 0)
    text = generation_messages(plan)[1]["content"]
    for i, (kind, count) in enumerate(plan.questions, start=1):
        assert f'{i}. type "{kind}"' in text
        if kind != "noul":
            assert f"exactly {count}" in text
    assert len(GENERATION_TEMPLATE_SHA256) == 64 and len(LABEL_TEMPLATE_SHA256) == 64


def _output_for(plan) -> str:
    questions = {}
    for i, (kind, count) in enumerate(plan.questions):
        q: dict = {"type": kind, "instructions": f"question {i}?"}
        if kind == "choice":
            q["options"] = [{"name": f"option {j}", "criteria": None} for j in range(count)]
        elif kind == "score":
            q["levels"] = [{"name": f"level {j}"} for j in range(count)]
        questions[f"question_{i}"] = q
    return "```json\n" + json.dumps({"state": "a real ticket", "questions": questions}) + "\n```"


def test_a_well_formed_output_parses_into_a_valid_request():
    plan = plan_case(0, 0)
    request = parse_generated(_output_for(plan), plan)
    assert isinstance(request, DecisionRequest)
    assert [q.type for q in request.questions.values()] == [k for k, _ in plan.questions]


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda blob: blob.update(questions={}), "questions"),
        (lambda blob: blob.update(state=""), "state"),
        (
            lambda blob: blob["questions"].update(extra={"type": "noul", "instructions": "x?"}),
            "primitives",
        ),
        (lambda blob: blob["questions"].update({"Bad Id!": blob["questions"].popitem()[1]}), "id"),
    ],
)
def test_a_malformed_output_is_rejected_with_a_reason(mutate, reason):
    plan = plan_case(0, 0)
    blob = json.loads(_output_for(plan).strip("`json\n"))
    mutate(blob)
    with pytest.raises(Rejected, match=reason):
        parse_generated(json.dumps(blob), plan)


def test_the_label_prompt_asks_one_question_about_the_rendered_state():
    question = ChoiceQuestion(
        instructions="Which team?",
        options=[{"name": "billing", "criteria": "money"}, {"name": "shipping"}],
    )
    text = label_messages({"b": 1, "a": "x"}, question)[1]["content"]
    assert '{"a": "x","b": 1}' in text  # the student's canonical rendering
    assert "- billing: money" in text and "- shipping" in text


def test_a_noul_is_scored_in_the_aligned_order():
    """The harness aligns a Noul as (no, yes); the teacher's distribution must too."""
    assert option_continuations(NoulQuestion(instructions="Is it?")) == ["no", "yes"]
    score = ScoreQuestion(instructions="How?", levels=[{"name": "low"}, {"name": "high"}])
    assert option_continuations(score) == ["low", "high"]


def test_softmax_renormalises_over_the_declared_labels():
    probs = softmax([math.log(0.2), math.log(0.1)])
    assert probs == pytest.approx([2 / 3, 1 / 3])
    assert sum(softmax([-900.0, -901.0, -950.0])) == pytest.approx(1.0)


# -- the loader ----------------------------------------------------------------


def test_every_label_is_marked_as_the_teachers_and_carries_its_distribution():
    case = case_from_record(_record(0), name="teacher-workflows", split="train")
    assert case is not None
    for expectation in case.expected.values():
        assert expectation.from_teacher
        assert sum(expectation.distribution) == pytest.approx(1.0)
        assert expectation.label == expectation.hard_label
    assert case.expected["team"].label == 1
    assert case.expected["refund"].distribution == pytest.approx((0.2, 0.8))
    assert "teacher" in case.tags


def test_a_label_over_different_options_drops_the_case():
    """Labels scored against another option list are not labels for this one."""
    record = _record(0)
    record["labels"]["team"]["options"] = ["billing", "hardware", "shipping"]
    assert case_from_record(record, name="teacher-workflows", split="train") is None


def test_the_loader_splits_by_case_and_refuses_nothing_green_permits(tmp_path):
    root = _write(tmp_path, [_record(i) for i in range(60)])
    train = load("teacher-workflows", "train", purpose="train", root=root)
    test = load("teacher-workflows", "test", purpose="eval", root=root)
    assert len(train) + len(test) == 60 and test and train
    ids = lambda cases: {c.case_id.rsplit("/", 1)[1] for c in cases}  # noqa: E731
    assert not ids(train) & ids(test)


def test_a_build_not_yet_fetched_says_how_to_fetch_it(tmp_path):
    with pytest.raises(CorpusNotFetched, match="modal_teacher.py fetch"):
        load("teacher-workflows", "train", purpose="train", root=tmp_path)


# -- soft targets --------------------------------------------------------------


def test_the_trainer_fits_the_teachers_distribution_not_its_argmax():
    """Cross-entropy against the distribution is minimised by reporting it."""
    torch = pytest.importorskip("torch")
    from trigon.training.losses import question_loss

    case = case_from_record(_record(0, yes=0.8), name="t", split="train")
    choice = case.expected["team"]
    target = torch.tensor(choice.distribution).log()
    at_target = question_loss(
        target.clone(), "choice", choice.label, distribution=choice.distribution
    )
    sharp = torch.tensor([-9.0, 0.0, -9.0])
    at_argmax = question_loss(sharp, "choice", choice.label, distribution=choice.distribution)
    assert float(at_target) < float(at_argmax)

    noul = case.expected["refund"]
    losses = {
        p: float(
            question_loss(
                torch.tensor([math.log(p / (1 - p))]),
                "noul",
                noul.label,
                distribution=noul.distribution,
            )
        )
        for p in (0.5, 0.8, 0.99)
    }
    assert min(losses, key=losses.get) == 0.8


def test_the_trainer_takes_a_teacher_labelled_case_end_to_end():
    pytest.importorskip("torch")
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.training import TrainingConfig, train

    cases = [case_from_record(_record(i), name="t", split="train") for i in range(6)]
    backend = TorchReadoutBackend(config=ReadoutConfig(d_model=32, n_layers=1), seed=0)
    result = train(
        backend, cases, TrainingConfig(epochs=1, accumulate=2, validation_fraction=0.0, seed=0)
    )
    assert result.n_questions == 18


# -- never reported as calibration ------------------------------------------------


@pytest.fixture
def teacher_cases() -> list[Case]:
    return [case_from_record(_record(i, pick=i % 3), name="t", split="test") for i in range(12)]


def test_the_suite_summary_refuses_teacher_labels(teacher_cases):
    """`summarize` is what every published ECE and every release gate reads."""
    engine = Engine(LexicalBackend())
    with pytest.raises(TeacherLabelsAreNotCalibration, match="teacher_agreement"):
        run_calibration_suite(engine, teacher_cases)
    with pytest.raises(TeacherLabelsAreNotCalibration):
        summarize("any", "model", run_cases(engine, teacher_cases))


def test_one_teacher_label_in_a_mixed_set_is_enough_to_refuse(teacher_cases):
    engine = Engine(LexicalBackend())
    real = Case(
        case_id="real/0",
        request=teacher_cases[0].request,
        expected={q: Expectation(label=0) for q in teacher_cases[0].expected},
    )
    with pytest.raises(TeacherLabelsAreNotCalibration):
        run_calibration_suite(engine, [real, *teacher_cases[:1]])
    # And the same case without the teacher's mark is an ordinary suite.
    result, _ = run_calibration_suite(engine, [real], floor_trials=2)
    assert result.calibration is not None


def test_no_calibrator_is_fitted_to_a_teacher(teacher_cases):
    """A temperature fitted to a teacher calibrates the model to the teacher."""
    from trigon.cli import _fit_calibration

    with pytest.raises(TeacherLabelsAreNotCalibration, match="calibrator"):
        _fit_calibration(Engine(LexicalBackend()), teacher_cases)


def test_agreement_is_computed_and_says_it_is_not_calibration(teacher_cases):
    engine = Engine(LexicalBackend())
    agreement = teacher_agreement(run_cases(engine, teacher_cases), teacher_cases, floor_trials=5)
    assert "NOT calibration" in agreement.label
    assert "NOT calibration" in agreement.render_markdown()
    payload = agreement.to_dict()
    assert payload["is_calibration"] is False
    assert agreement.n_questions == 36 and 0 <= agreement.argmax_agreement <= 1
    assert agreement.mean_kl >= 0 and agreement.baseline_kl >= 0
    # Every ECE carries its floor, and below the sample floor it is not quoted.
    assert agreement.floor is not None
    assert agreement.n_questions < MIN_CALIBRATION_SAMPLES and not agreement.ece_quotable
    assert "not quoted" in agreement.render_markdown()


def test_agreement_refuses_labels_that_are_not_the_teachers(teacher_cases):
    """A ground-truth label through this function would come out named agreement."""
    engine = Engine(LexicalBackend())
    real = Case(
        case_id="real/0",
        request=teacher_cases[0].request,
        expected={q: Expectation(label=0) for q in teacher_cases[0].expected},
    )
    with pytest.raises(ValueError, match="not a teacher label"):
        teacher_agreement(run_cases(engine, [real]), teacher_cases)


def test_a_teacher_that_agrees_with_itself_scores_perfect_agreement(teacher_cases):
    """The metric's own sanity check: the teacher against itself."""
    from trigon.evals.harness import CaseOutcome, QuestionOutcome

    outcomes = [
        CaseOutcome(
            case=case,
            response=None,  # type: ignore[arg-type]
            questions={
                qid: QuestionOutcome(
                    question_id=qid,
                    primitive=question.type,
                    labels=(),
                    probabilities=case.expected[qid].distribution,
                    confidence=None,
                    expected=case.expected[qid],
                )
                for qid, question in case.request.questions.items()
            },
            latency_ms=0.0,
        )
        for case in teacher_cases
    ]
    agreement = teacher_agreement(outcomes, teacher_cases, floor_trials=5)
    assert agreement.argmax_agreement == 1.0
    assert agreement.mean_kl == pytest.approx(0.0, abs=1e-12)
    assert agreement.model_mean_confidence == pytest.approx(agreement.teacher_mean_confidence)


# -- the committed sample of the real build ------------------------------------------


def test_the_fixture_is_a_small_sample_of_the_pinned_build():
    """At most twenty cases, one per domain, every one loadable, every one
    labelled by the pinned teacher under the templates in the code."""
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert 1 <= len(records) <= 20
    assert len({r["domain"] for r in records}) == len(records)
    for record in records:
        assert record["teacher"]["model"] == TEACHER_MODEL
        assert record["teacher"]["revision"] == TEACHER_REVISION
        # A template edited after the build would label the next build
        # differently from the published one; this fails until it is rebuilt.
        assert record["teacher"]["generation_template_sha256"] == GENERATION_TEMPLATE_SHA256
        assert record["teacher"]["label_template_sha256"] == LABEL_TEMPLATE_SHA256
        case = case_from_record(record, name="teacher-workflows", split="fixture")
        assert case is not None, record["case_id"]
        for label in record["labels"].values():
            assert len(label["logprobs"]) == len(label["options"])
            assert all(lp <= 0 for lp in label["logprobs"])
            assert 0 < label["declared_mass"] <= 1 + 1e-6


def test_agent_plans_are_noul_weighted_and_cover_every_domain():
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
    from local_teacher import AGENT_DOMAINS, plan_agent_case

    plans = [plan_agent_case(0, i) for i in range(200)]
    assert {p.domain for p in plans} == {d.name for d in AGENT_DOMAINS}
    assert all(p.case_id.startswith("tg0-") for p in plans)
    kinds = [kind for p in plans for kind, _ in p.questions]
    assert kinds.count("noul") > kinds.count("choice") > 0
    # The same seed plans the same build.
    assert plan_agent_case(0, 7) == plans[7]
