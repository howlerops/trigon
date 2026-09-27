"""The adversarial + paired stream: labels by construction, pairs, and the flag.

The stream's whole claim is that each variant's label is known without an
annotator -- held fixed by an edit that cannot change it, or derived by one
whose effect is exact. These tests hold the generator to that, and hold the
trainer to training exactly as before when the stream is off.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import pathlib
import re

import pytest

from trigon.evals import synthetic_outcome_cases
from trigon.evals.harness import Case, Expectation
from trigon.evals.paired import (
    _AFFIRM,
    _DENY,
    _DISTRACTORS,
    _INJECTIONS,
    _PARAPHRASES,
    _WRAPPERS,
    CLAIM,
    KINDS,
    PairedBenchmark,
    PairedCase,
    derive,
    paired_benchmarks,
    paired_stream,
)
from trigon.schema import render_state
from trigon.types import ChoiceQuestion, DecisionRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent

_INTENTS = ["card arrival", "lost or stolen card", "top up failed", "exchange rate"]


def _banking_like(n: int = 24) -> list[Case]:
    """Banking77's shape -- a text state and one 'intent' Choice -- without the download."""
    texts = [
        "When will my new card arrive?",
        "Someone stole my card yesterday.",
        "My top up did not go through.",
        "What exchange rate do you use?",
    ]
    question = ChoiceQuestion(
        instructions="Which banking intent does this customer message express?",
        options=[{"name": name} for name in _INTENTS],
    )
    return [
        Case(
            case_id=f"banking77/train/{i}",
            request=DecisionRequest(state=texts[i % 4], questions={"intent": question}),
            expected={"intent": Expectation(label=i % 4)},
            domain="banking77",
        )
        for i in range(n)
    ]


SOURCES = {
    "banking77": _banking_like,
    "synthetic": lambda n=24: synthetic_outcome_cases(n, seed=3, noise=0.1),
}


def _dump(cases):
    return [
        (c.case_id, c.request.model_dump_json(), repr(c.expected), c.pair_id, c.relation)
        for c in cases
    ]


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_the_stream_is_deterministic_by_seed(source):
    base = SOURCES[source]()
    first = paired_stream(base, 1.0, seed=4)
    assert _dump(first) == _dump(paired_stream(base, 1.0, seed=4))
    assert _dump(first) != _dump(paired_stream(base, 1.0, seed=5))


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_every_kind_is_derived_in_rotation(source):
    stream = paired_stream(SOURCES[source](), 1.0, seed=0)
    assert {c.kind for c in stream} == set(KINDS)
    # A negation pair is two cases, every other pair one.
    pairs = {c.pair_id for c in stream}
    assert len(pairs) == 24
    assert len(stream) == 24 + sum(1 for c in stream if c.kind == "negation") // 2


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_held_fixed_labels_are_the_base_labels(source):
    base = {c.request.model_dump_json(): c for c in SOURCES[source]()}
    for case in paired_stream(list(base.values()), 1.0, seed=1):
        if case.relation != "same":
            continue
        origin = base[case.anchor.model_dump_json()]
        assert case.expected == origin.expected, case.case_id
        assert set(case.request.questions) == set(origin.request.questions)
        if case.kind == "paraphrase":
            assert case.request.state == origin.request.state
            for qid, question in case.request.questions.items():
                assert question.instructions != origin.request.questions[qid].instructions
        else:
            # The original state survives whole inside the edited one.
            assert case.request.questions == origin.request.questions
            if isinstance(origin.request.state, str):
                assert origin.request.state in case.request.state
            else:
                assert origin.request.state.items() <= case.request.state.items()
            assert render_state(case.request.state) != render_state(origin.request.state)


def test_an_injection_names_a_wrong_answer():
    """An injection that named the true answer would steer toward the label."""
    for case in paired_stream(_banking_like(), 2.0, seed=2, kinds=("injection",)):
        truth = _INTENTS[case.expected["intent"].label]
        injected = case.request.state.replace(case.anchor.state, "")
        named = [name for name in _INTENTS if name in injected]
        assert named and truth not in named


@pytest.mark.parametrize("source", sorted(SOURCES))
def test_negation_labels_are_derived_exactly(source):
    base = {c.request.model_dump_json(): c for c in SOURCES[source]()}
    stream = paired_stream(list(base.values()), 2.0, seed=3, kinds=("negation",))
    by_pair: dict[str, dict[str, PairedCase]] = {}
    for case in stream:
        by_pair.setdefault(case.pair_id, {})["deny" if case.anchor else "affirm"] = case
    named_true = 0
    for pair in by_pair.values():
        affirm, deny = pair["affirm"], pair["deny"]
        assert deny.anchor == affirm.request, "the deny is anchored on its own affirm"
        assert affirm.relation == deny.relation == "complement"
        assert set(affirm.request.questions) == set(deny.request.questions) == {CLAIM}
        p_affirm = affirm.expected[CLAIM].probability
        p_deny = deny.expected[CLAIM].probability
        assert p_affirm + p_deny == 1.0
        # Recover the named option and check the affirm against the truth.
        text = affirm.request.questions[CLAIM].instructions
        quoted = re.findall(r'"([^"]+)"', text)
        origins = [c for c in base.values() if c.request.state == affirm.request.state]
        assert origins
        origin = origins[0]
        truths = set()
        for qid, question in origin.request.questions.items():
            if question.instructions not in quoted:
                continue
            names = getattr(question, "names", ["no", "yes"])
            option = next(name for name in quoted if name in names)
            truths.add(names.index(option) == origin.expected[qid].hard_label)
        assert truths == {p_affirm == 1.0}
        named_true += p_affirm == 1.0
    # Named option is the true one about half the time: "no" is not a strategy.
    assert 0.2 < named_true / len(by_pair) < 0.8


def test_negation_of_a_distribution_is_soft():
    question = ChoiceQuestion(instructions="Pick one.", options=[{"name": "a"}, {"name": "b"}])
    case = Case(
        case_id="soft/0",
        request=DecisionRequest(state="text", questions={"q": question}),
        expected={"q": Expectation(distribution=(0.3, 0.7))},
    )
    import random

    affirm, deny = derive(case, "negation", rng=random.Random(0))
    a, d = affirm.expected[CLAIM].distribution, deny.expected[CLAIM].distribution
    assert a is not None and d is not None
    assert a[1] + d[1] == pytest.approx(1.0)
    assert a[1] in (0.3, 0.7)


def test_pair_ids_are_unique_per_pair_and_anchors_are_the_bases():
    base = _banking_like()
    stream = paired_stream(base, 1.0, seed=0)
    requests = {c.request.model_dump_json() for c in base}
    counts: dict[str, int] = {}
    for case in stream:
        assert case.pair_id and case.case_id.startswith(case.pair_id)
        counts[case.pair_id] = counts.get(case.pair_id, 0) + 1
        if case.relation == "same":
            assert case.anchor.model_dump_json() in requests
    assert set(counts.values()) <= {1, 2}
    assert len({c.case_id for c in stream}) == len(stream)


def test_the_train_and_eval_template_pools_share_no_wording():
    """A model measured on the templates it trained on is scored on memory."""
    for pools in (_INJECTIONS, _DISTRACTORS, _WRAPPERS, _AFFIRM, _DENY):
        assert not set(pools["train"]) & set(pools["eval"])
    for pools in _PARAPHRASES.values():
        assert not set(pools["train"]) & set(pools["eval"])


def test_fraction_zero_derives_nothing_and_bad_kinds_are_refused():
    assert paired_stream(_banking_like(), 0.0, seed=0) == []
    with pytest.raises(ValueError):
        paired_stream(_banking_like(), 0.5, seed=0, kinds=("typo",))


# -- The consistency term ------------------------------------------------------


def test_consistency_is_zero_for_identical_answers_and_positive_otherwise():
    torch = pytest.importorskip("torch")
    from trigon.training import consistency_loss

    a = torch.tensor([1.0, -0.5, 2.0])
    b = torch.tensor([0.0, 0.5, -1.0])
    for kind in ("choice", "score"):
        assert float(consistency_loss(a, a.clone(), kind)) == pytest.approx(0.0, abs=1e-7)
        assert float(consistency_loss(a, b, kind)) > 0
        # Symmetric.
        assert float(consistency_loss(a, b, kind)) == pytest.approx(
            float(consistency_loss(b, a, kind))
        )
    yes, no = torch.tensor([1.5]), torch.tensor([-0.3])
    assert float(consistency_loss(yes, yes.clone(), "noul")) == pytest.approx(0.0, abs=1e-7)
    assert float(consistency_loss(yes, no, "noul")) > 0
    # Complement: zero exactly when P(yes) + P(yes on the complement) = 1.
    assert float(consistency_loss(yes, -yes, "noul", "complement")) == pytest.approx(0, abs=1e-7)
    assert float(consistency_loss(yes, yes.clone(), "noul", "complement")) > 0
    with pytest.raises(ValueError):
        consistency_loss(a, b, "choice", "complement")


def test_consistency_gradient_reaches_both_sides():
    torch = pytest.importorskip("torch")
    from trigon.training import consistency_loss

    a = torch.tensor([1.0, 0.0], requires_grad=True)
    b = torch.tensor([0.0, 1.0], requires_grad=True)
    consistency_loss(a, b, "choice").backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


# -- The flag ------------------------------------------------------------------


def _spike_weights(cases, **config):
    torch = pytest.importorskip("torch")
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.training import TrainingConfig, train

    backend = TorchReadoutBackend(config=ReadoutConfig(d_model=16, n_layers=1), seed=0)
    report = train(
        backend,
        cases,
        TrainingConfig(epochs=2, learning_rate=3e-3, accumulate=4, seed=0, **config),
    )
    state = {k: v.detach().clone() for k, v in backend.model.state_dict().items()}
    return state, report, torch


def _plain(case: PairedCase) -> Case:
    return Case(
        case_id=case.case_id,
        request=case.request,
        expected=case.expected,
        domain=case.domain,
        tags=case.tags,
    )


def test_weight_zero_trains_paired_cases_exactly_as_plain_ones():
    """With the term off, an anchor is never run: bit-identical to plain data."""
    base = synthetic_outcome_cases(24, seed=1)
    paired = paired_stream(base, 1.0, seed=0)
    as_paired, _, torch = _spike_weights([*base, *paired])
    as_plain, _, _ = _spike_weights([*base, *(_plain(c) for c in paired)])
    for key in as_plain:
        assert torch.equal(as_paired[key], as_plain[key]), key


def test_the_defaults_are_off():
    pytest.importorskip("torch")
    from trigon.training import TrainingConfig

    config = TrainingConfig()
    assert config.consistency_weight == 0.0
    assert config.augment is None


def test_augment_sees_the_training_split_only_and_the_term_changes_training():
    base = synthetic_outcome_cases(40, seed=2)
    seen: list[int] = []

    def augment(cases):
        seen.append(len(cases))
        assert not any(isinstance(c, PairedCase) for c in cases)
        return paired_stream(cases, 0.5, seed=0)

    plain, _, torch = _spike_weights(base)
    mixed, report, _ = _spike_weights(base, augment=augment)
    # validation_fraction 0.1 holds out 4 of 40 before the stream is derived.
    assert seen == [36]
    assert report.n_cases > 36
    consistent, _, _ = _spike_weights(base, augment=augment, consistency_weight=1.0)
    assert any(not torch.equal(plain[k], mixed[k]) for k in plain)
    assert any(not torch.equal(mixed[k], consistent[k]) for k in mixed)
    assert all(torch.isfinite(v).all() for v in consistent.values())


def test_train_corpus_flags_default_off():
    spec = importlib.util.spec_from_file_location(
        "train_corpus", ROOT / "scripts" / "train_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["banking77"])
    assert args.paired_mix == 0.0 and args.consistency_weight == 0.0
    assert args.robustness_n == 0
    assert module.paired_augment(args) is None
    on = module.parse_args(["banking77", "--paired-mix", "0.5", "--paired-kinds", "negation"])
    augment = module.paired_augment(on)
    assert isinstance(augment, functools.partial)
    assert {c.kind for c in augment(_banking_like())} == {"negation"}


# -- The benchmarks ------------------------------------------------------------


def test_benchmarks_pair_every_variant_with_its_anchor(engine):
    from trigon.evals import run_jaggedness

    results = run_jaggedness(engine, paired_benchmarks(_banking_like(40), n=12))
    by_name = {r.suite: r.extra for r in results}
    assert set(by_name) == {f"jaggedness/paired_{k}" for k in KINDS}
    # The floor never reads a Choice's question, so no rewording moves it --
    # the reason paraphrase agreement is read beside accuracy.
    assert by_name["jaggedness/paired_paraphrase"]["flip_rate"] == 0.0
    # An injection naming another intent steers a keyword matcher.
    assert by_name["jaggedness/paired_injection"]["flip_rate"] > 0.0
    assert "rot" in by_name["jaggedness/paired_padding"]
    assert 0.0 <= by_name["jaggedness/paired_negation"]["mean_incoherence"] <= 1.0
    negation = by_name["jaggedness/paired_negation"]
    for key in ("mean_p_yes_affirm", "mean_p_yes_deny", "stdev_p_yes_affirm", "separation"):
        assert key in negation
    assert 0.0 <= negation["mean_p_yes_affirm"] <= 1.0
    anchors = {v for k, v in by_name["jaggedness/paired_padding"].items() if "anchor" in k}
    assert anchors == {by_name["jaggedness/paired_injection"]["accuracy_anchor"]}


def test_benchmark_cases_use_the_eval_pool_only():
    bench = PairedBenchmark(n=24, base=tuple(_banking_like()), kind="injection")
    texts = " ".join(c.request.state for c in bench.cases())
    for template in _INJECTIONS["train"]:
        assert template.split("{option}")[0] not in texts
    assert dataclasses.is_dataclass(bench)
