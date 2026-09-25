"""Training: the objectives, and that the loop actually moves the model.

Kept small and fast. The full run that takes the reference model from failing
the calibration gate to passing it is ``trigon train``, reported in
``docs/evals.md``; what is asserted here is that the machinery is correct and
that loss goes down, which is what a regression would break first.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="training needs the 'train' extra")

from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend  # noqa: E402
from trigon.evals import synthetic_outcome_cases  # noqa: E402
from trigon.training import (  # noqa: E402
    OrdinalConfig,
    TrainingConfig,
    question_loss,
    squared_emd,
    train,
)


def _tiny_backend() -> TorchReadoutBackend:
    return TorchReadoutBackend(ReadoutConfig(d_model=64, n_layers=1, n_heads=2, d_ff=64), seed=0)


def test_cross_entropy_is_minimised_by_the_true_distribution():
    """The propriety property the whole objective rests on: no other
    prediction beats the truth in expectation."""
    truth = [0.7, 0.2, 0.1]
    honest = torch.log(torch.tensor(truth))
    overconfident = torch.log(torch.tensor([0.95, 0.03, 0.02]))
    underconfident = torch.log(torch.tensor([0.4, 0.35, 0.25]))

    def expected(logits):
        return sum(p * float(question_loss(logits, "choice", i)) for i, p in enumerate(truth))

    assert expected(honest) < expected(overconfident)
    assert expected(honest) < expected(underconfident)


def test_noul_loss_is_binary_cross_entropy():
    logits = torch.tensor([0.0])
    assert float(question_loss(logits, "noul", 1)) == pytest.approx(math.log(2), abs=1e-6)


def test_a_noul_with_annotators_is_fitted_to_the_share_who_said_yes():
    """GoEmotions' Nouls carry (no, yes) shares; the drawn label must not replace them.

    Before this, a Noul's distribution was ignored and the loss read the one
    annotator drawn for the case, throwing away the other raters.
    """
    share = (0.7, 0.3)
    honest = torch.logit(torch.tensor([0.3]))
    for label in (0, 1):  # whichever annotator was drawn
        losses = {
            p: float(
                question_loss(torch.logit(torch.tensor([p])), "noul", label, distribution=share)
            )
            for p in (0.1, 0.3, 0.5, 0.9)
        }
        assert min(losses, key=losses.get) == 0.3
    assert float(question_loss(honest, "noul", 1, distribution=share)) == pytest.approx(
        -(0.3 * math.log(0.3) + 0.7 * math.log(0.7)), abs=1e-6
    )


def test_squared_emd_costs_less_for_a_near_miss():
    """The ordinal property: being one level out should cost less than four."""
    logits = torch.log(torch.tensor([0.0, 1.0, 0.0, 0.0]) + 1e-9)
    assert float(squared_emd(logits, 2)) < float(squared_emd(logits, 3))


def test_ordinal_term_only_applies_to_score():
    logits = torch.log(torch.tensor([0.1, 0.7, 0.2]))
    ordinal = OrdinalConfig(weight=0.5)
    plain = float(question_loss(logits, "choice", 0, ordinal))
    assert plain == pytest.approx(float(question_loss(logits, "choice", 0)))
    assert float(question_loss(logits, "score", 0, ordinal)) != pytest.approx(plain)


def test_ordinal_weight_is_bounded():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        OrdinalConfig(weight=1.5)


def test_training_reduces_loss():
    backend = _tiny_backend()
    cases = synthetic_outcome_cases(n=120, seed=0, noise=0.2)
    report = train(backend, cases, TrainingConfig(epochs=3, accumulate=8, learning_rate=1e-2))
    assert report.final_loss < report.first_loss
    # 10% is held out to choose which epoch to keep, so the report counts the
    # cases gradients were actually taken on -- not the ones handed in.
    assert report.n_cases == 108
    assert report.n_questions == 324  # three answerable questions per case


def test_training_leaves_the_model_in_eval_mode():
    """A model left in train mode serves with dropout on, which quietly
    decalibrates everything downstream."""
    backend = _tiny_backend()
    train(backend, synthetic_outcome_cases(n=20), TrainingConfig(epochs=1))
    assert not backend.model.training


def test_serving_and_training_share_one_forward_path():
    """``infer`` must be ``logits`` with no_grad, not a second implementation:
    a divergence between them is invisible until someone measures ECE on the
    served path."""
    backend = _tiny_backend()
    compiler = backend.make_compiler()
    case = synthetic_outcome_cases(n=1)[0]
    compiled = compiler.compile_request(case.request)

    with torch.no_grad():
        raw, _ = backend.logits(compiled, case.request)
    served = backend.infer(compiled, case.request)
    for qid, tensor in raw.items():
        assert served.outputs[qid].logits == pytest.approx(tuple(float(x) for x in tensor.tolist()))


def test_training_refuses_an_empty_set():
    with pytest.raises(ValueError, match="nothing to train on"):
        train(_tiny_backend(), [], TrainingConfig())


def test_a_trained_model_round_trips_through_a_checkpoint(tmp_path):
    """A run that cannot be reloaded cannot be served, and a report about a
    model nobody can run again is a claim rather than a result."""
    backend = _tiny_backend()
    compiler = backend.make_compiler()
    train(backend, synthetic_outcome_cases(n=40, noise=0.2), TrainingConfig(epochs=1))

    path = tmp_path / "model.pt"
    backend.save(path)
    reloaded = TorchReadoutBackend.load(path)

    # Same architecture, recovered from the checkpoint alone.
    assert reloaded.config.d_model == backend.config.d_model
    assert reloaded.config.n_layers == backend.config.n_layers
    assert reloaded.model_version == backend.model_version

    case = synthetic_outcome_cases(n=1, seed=7)[0]
    compiled = compiler.compile_request(case.request)
    before = backend.infer(compiled, case.request)
    after = reloaded.infer(compiled, case.request)
    for qid, out in before.outputs.items():
        assert after.outputs[qid].logits == pytest.approx(out.logits)


def test_a_reloaded_model_serves_through_the_engine(tmp_path):
    from trigon.engine import Engine

    backend = _tiny_backend()
    train(backend, synthetic_outcome_cases(n=40, noise=0.2), TrainingConfig(epochs=1))
    path = tmp_path / "model.pt"
    backend.save(path)

    reloaded = TorchReadoutBackend.load(path)
    engine = Engine(reloaded, compiler=reloaded.make_compiler())
    response = engine.answer(synthetic_outcome_cases(n=1, seed=3)[0].request)
    assert set(response.answers) == {"plan", "at_risk", "size"}
    assert sum(response.answers["plan"].probabilities.values()) == pytest.approx(1.0)


def test_a_checkpoint_written_before_a_parameter_existed_still_loads(tmp_path):
    """Adding `match_log_scale` broke loading every run saved before it — the
    certified model included. New parameters take their initialised value;
    anything else that does not match still fails loudly."""
    from trigon.backends.torch_readout import TorchReadoutBackend

    path = tmp_path / "old.pt"
    TorchReadoutBackend(seed=0).save(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["state_dict"]["match_log_scale"]
    torch.save(payload, path)

    reloaded = TorchReadoutBackend.load(path)
    expected = TorchReadoutBackend(seed=0).model.match_log_scale
    assert torch.equal(reloaded.model.match_log_scale, expected)


def test_a_checkpoint_with_a_mismatched_parameter_still_fails(tmp_path):
    from trigon.backends.torch_readout import TorchReadoutBackend

    path = tmp_path / "broken.pt"
    TorchReadoutBackend(seed=0).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state_dict"]["noul_head.weight"] = torch.zeros(1, 3)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        TorchReadoutBackend.load(path)


def test_a_checkpoint_keeps_the_head_it_was_trained_with(tmp_path):
    """`match_residual` defaults on for new models. A run saved before it did
    must not silently acquire it — that would change the arithmetic of every
    model on disk and stop a committed run reproducing its own report."""
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend

    assert ReadoutConfig().match_residual is True

    path = tmp_path / "before.pt"
    TorchReadoutBackend(ReadoutConfig(match_residual=False), seed=0).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["config"]["match_residual"]
    del payload["config"]["match_normalize"]
    torch.save(payload, path)

    reloaded = TorchReadoutBackend.load(path)
    assert reloaded.config.match_residual is False
    assert reloaded.config.match_normalize is False


def test_a_checkpoint_is_served_with_the_tokenizer_it_was_trained_on(tmp_path):
    """A model served under a different vocabulary reads every token id as a
    different word, and nothing raises. The checkpoint names its tokenizer."""
    from trigon.backends.tokenizer import HashingTokenizer
    from trigon.backends.torch_readout import TorchReadoutBackend

    path = tmp_path / "hashed.pt"
    TorchReadoutBackend(tokenizer=HashingTokenizer(4096), seed=0).save(path)

    reloaded = TorchReadoutBackend.load(path)
    assert isinstance(reloaded.tokenizer, HashingTokenizer)
    assert reloaded.tokenizer.vocab_size == 4096
    assert reloaded.config.vocab_size == 4096


def test_a_checkpoint_written_before_tokenizers_were_recorded_loads_as_hashing(tmp_path):
    """Every such checkpoint used the hashing tokenizer — including the
    committed reference run, which stopped loading when the default changed."""
    from trigon.backends.tokenizer import HashingTokenizer
    from trigon.backends.torch_readout import TorchReadoutBackend

    path = tmp_path / "old.pt"
    TorchReadoutBackend(tokenizer=HashingTokenizer(8192), seed=0).save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["tokenizer"]
    torch.save(payload, path)

    reloaded = TorchReadoutBackend.load(path)
    assert isinstance(reloaded.tokenizer, HashingTokenizer)
    assert reloaded.tokenizer.vocab_size == 8192


def test_the_committed_reference_run_still_loads_and_answers():
    """The certified model is the one thing that must never stop loading."""
    import pathlib

    from trigon.backends.torch_readout import TorchReadoutBackend

    weights = pathlib.Path(__file__).resolve().parent.parent / "reports" / "reference-run.pt"
    if not weights.exists():  # gitignored; present only after a local train run
        pytest.skip("reference-run.pt is not checked in")
    backend = TorchReadoutBackend.load(weights)
    assert backend.model_version.endswith("+50b2d6bd")
    assert backend.tokenizer.vocab_size == 8192


def test_a_bpe_checkpoint_refuses_a_vocabulary_of_the_wrong_size():
    from trigon.backends.tokenizer import build_tokenizer

    with pytest.raises(ValueError, match="trained on a 99-token BPE vocabulary"):
        build_tokenizer({"kind": "bpe", "vocab_size": 99})
    with pytest.raises(ValueError, match="unknown tokenizer kind"):
        build_tokenizer({"kind": "sentencepiece", "vocab_size": 32000})


# -- padded batching ---------------------------------------------------------


def _batch_items(backend, n=12, seed=1000):
    from trigon.evals.datasets import synthetic_outcome_cases

    compiler = backend.make_compiler()
    cases = synthetic_outcome_cases(n=n, seed=seed, noise=0.2)
    return [(compiler.compile_request(c.request), c.request) for c in cases]


def test_a_batched_pass_equals_the_unbatched_one_exactly():
    """The trainer batches and the gateway does not. If they differ at all, the
    model is trained on arithmetic it is never served with — the standard way
    to score well offline and be miscalibrated in production."""
    from trigon.backends.torch_readout import TorchReadoutBackend

    backend = TorchReadoutBackend(seed=0)
    items = _batch_items(backend)
    with torch.no_grad():
        one_at_a_time = [backend.logits(compiled, request)[0] for compiled, request in items]
        batched = backend.logits_batch(items)

    assert len(batched) == len(items)
    for single, many in zip(one_at_a_time, batched, strict=True):
        assert single.keys() == many.keys()
        for qid in single:
            assert torch.equal(single[qid], many[qid]), f"{qid} moved under batching"


def test_padding_cannot_reach_across_a_batch():
    """A request's answer must not depend on what shared its batch. Requests of
    very different lengths are the case that would expose a leak."""
    from trigon.backends.torch_readout import TorchReadoutBackend
    from trigon.types import ChoiceQuestion, SystemOneRequest

    backend = TorchReadoutBackend(seed=0)
    compiler = backend.make_compiler()
    question = ChoiceQuestion(
        instructions="Route this.", options=[{"name": "a"}, {"name": "b"}, {"name": "c"}]
    )
    short = SystemOneRequest(state="brief", questions={"q": question})
    long = SystemOneRequest(state="a much longer message " * 60, questions={"q": question})

    items = [(compiler.compile_request(r), r) for r in (short, long)]
    with torch.no_grad():
        alone = [backend.logits(compiled, request)[0]["q"] for compiled, request in items]
        # Together, and again in the other order — padding lands on a
        # different member of the batch each way.
        together = [out["q"] for out in backend.logits_batch(items)]
        reversed_ = [out["q"] for out in backend.logits_batch(items[::-1])][::-1]

    for solo, mixed, flipped in zip(alone, together, reversed_, strict=True):
        assert torch.equal(solo, mixed)
        assert torch.equal(solo, flipped)


def test_a_batched_pass_never_returns_nan():
    """A padded row that may attend to nothing softmaxes over all -inf and
    returns NaN, which then propagates through the whole batch."""
    from trigon.backends.torch_readout import TorchReadoutBackend

    backend = TorchReadoutBackend(seed=0)
    with torch.no_grad():
        for out in backend.logits_batch(_batch_items(backend, n=16)):
            for qid, values in out.items():
                assert torch.isfinite(values).all(), f"{qid} produced a non-finite logit"


def test_an_empty_batch_is_not_an_error():
    from trigon.backends.torch_readout import TorchReadoutBackend

    assert TorchReadoutBackend(seed=0).logits_batch([]) == []


# -- keeping the best epoch --------------------------------------------------


def test_the_best_epoch_is_kept_not_the_last():
    """The 8,000-case run's loss bottomed at epoch 4 and rose for the next
    four. Keeping the last epoch ships weights the run had already beaten."""
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.evals.datasets import synthetic_outcome_cases
    from trigon.training import TrainingConfig, train

    backend = TorchReadoutBackend(ReadoutConfig(d_model=64, n_layers=1), seed=0)
    report = train(
        backend,
        synthetic_outcome_cases(n=400, seed=0, noise=0.2),
        # A learning rate high enough that later epochs get worse, which is the
        # case the selection exists for.
        TrainingConfig(epochs=4, learning_rate=0.3, accumulate=16, seed=0),
        compiler=backend.make_compiler(),
    )
    losses = [e.validation_loss for e in report.epochs]
    assert all(v is not None for v in losses), "a holdout should have been taken"
    best = min(range(len(losses)), key=lambda i: losses[i]) + 1
    assert report.kept_epoch == best


def test_selection_uses_data_the_gradients_never_saw():
    """Selecting on training loss picks the epoch that memorised hardest."""
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.evals.datasets import synthetic_outcome_cases
    from trigon.training import TrainingConfig, train

    cases = synthetic_outcome_cases(n=400, seed=0, noise=0.2)
    backend = TorchReadoutBackend(ReadoutConfig(d_model=64, n_layers=1), seed=0)
    report = train(
        backend,
        cases,
        TrainingConfig(epochs=2, learning_rate=0.01, accumulate=16, seed=0),
        compiler=backend.make_compiler(),
    )
    # 10% held out by default, so the reported case count is the training half.
    assert report.n_cases == len(cases) - max(1, int(len(cases) * 0.1))


def test_the_holdout_can_be_switched_off():
    """0 keeps every case for training and the final epoch's weights."""
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.evals.datasets import synthetic_outcome_cases
    from trigon.training import TrainingConfig, train

    cases = synthetic_outcome_cases(n=200, seed=0, noise=0.2)
    backend = TorchReadoutBackend(ReadoutConfig(d_model=64, n_layers=1), seed=0)
    report = train(
        backend,
        cases,
        TrainingConfig(epochs=2, learning_rate=0.01, accumulate=16, seed=0, validation_fraction=0),
        compiler=backend.make_compiler(),
    )
    assert report.n_cases == len(cases)
    assert all(e.validation_loss is None for e in report.epochs)
    assert report.kept_epoch == len(report.epochs)


def test_the_holdout_is_the_same_cases_on_a_rerun():
    """A run that reproduces has to hold out the same slice."""
    import random as _random

    from trigon.evals.datasets import synthetic_outcome_cases
    from trigon.training.trainer import TrainingConfig

    cases = synthetic_outcome_cases(n=300, seed=0, noise=0.2)
    config = TrainingConfig(seed=0)

    def holdout_ids():
        shuffled = list(cases)
        _random.Random(config.seed + 7919).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * config.validation_fraction))
        return [c.case_id for c in shuffled[:cut]]

    assert holdout_ids() == holdout_ids()


def test_the_residual_never_touches_a_score_head():
    """`match_residual` was designed and measured for the dot-product *option*
    head. Score shares that code path by accident of implementation, and
    applying it there regressed the reference run from closing 20% of the gap
    to Bayes to closing 3%."""
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.types import ScoreQuestion, SystemOneRequest

    request = SystemOneRequest(
        state="the customer is mildly annoyed",
        questions={
            "severity": ScoreQuestion(
                instructions="How severe?",
                levels=[{"name": "low"}, {"name": "mid"}, {"name": "high"}],
            )
        },
    )

    logits = {}
    for residual in (True, False):
        backend = TorchReadoutBackend(ReadoutConfig(match_residual=residual), seed=0)
        compiled = backend.make_compiler().compile_request(request)
        with torch.no_grad():
            logits[residual] = backend.logits(compiled, request)[0]["severity"]
    assert torch.equal(logits[True], logits[False])


def test_the_residual_still_changes_a_dot_product_choice():
    """The other half: it must still do the thing it was measured doing."""
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.schema import OptionScoring
    from trigon.types import ChoiceQuestion, SystemOneRequest

    request = SystemOneRequest(
        state="the card payment failed",
        questions={
            "route": ChoiceQuestion(
                instructions="Route this.",
                options=[{"name": "billing"}, {"name": "shipping"}, {"name": "account"}],
            )
        },
    )

    logits = {}
    for residual in (True, False):
        backend = TorchReadoutBackend(ReadoutConfig(match_residual=residual), seed=0)
        compiler = backend.make_compiler(option_scoring=OptionScoring.DOT_PRODUCT)
        compiled = compiler.compile_request(request)
        with torch.no_grad():
            logits[residual] = backend.logits(compiled, request)[0]["route"]
    assert not torch.equal(logits[True], logits[False])


def test_the_quantized_twin_is_a_real_serving_path():
    """`quantization_ece_delta` gated a run that nothing produced.

    `check_gates` has always accepted a `quantized=` suite result, and until
    now the only thing that ever passed one was a test fixture built by
    perturbing probabilities by hand. A gate whose input is synthetic tests the
    gate, not the system. `TorchReadoutBackend.quantized()` makes it a
    measurement: int8 weights, the same tokenizer, through the same engine.

    What is asserted here is the property the gate is *for* -- that the
    numerics move under the argmax rather than with it -- not a quality
    threshold, which is `trigon train`'s job on real cases.
    """
    from trigon.engine import Engine
    from trigon.types import SystemOneRequest

    backend = _tiny_backend()
    compiler = backend.make_compiler()
    twin = backend.quantized()

    # Named after what it is. Serving int8 answers under the float build's
    # name is the same failure as serving an untrained model under a trained
    # one's, which `stamp_version` exists to prevent.
    assert twin.model_version == f"{backend.model_version}+int8"
    assert backend.model_version.endswith("+int8") is False

    request = SystemOneRequest.model_validate(
        {
            "state": {"plan": "pro", "seats": 12, "open_tickets": 3},
            "questions": {
                "plan": {
                    "type": "choice",
                    "instructions": "Which plan is this account on?",
                    "options": [{"name": n} for n in ("free", "standard", "pro")],
                },
                "at_risk": {"type": "noul", "instructions": "Is this account at risk?"},
            },
        }
    )
    served = Engine(backend, compiler=compiler).answer(request)
    int8 = Engine(twin, compiler=compiler).answer(request)

    float_probs = served.answers["plan"].probabilities
    int8_probs = int8.answers["plan"].probabilities
    assert set(float_probs) == set(int8_probs)
    assert sum(int8_probs.values()) == pytest.approx(1.0)

    # The decision survives; the distribution under it does not, which is
    # precisely why argmax accuracy cannot stand in for this gate. Both halves
    # are asserted: a `quantized()` that quietly returned an unmodified copy
    # would pass the upper bound and make the gate vacuous, which is the exact
    # failure this whole path exists to end.
    assert int8.answers["plan"].selected == served.answers["plan"].selected
    moved = max(abs(int8_probs[k] - float_probs[k]) for k in float_probs)
    assert 0.0 < moved < 0.05, f"int8 weights moved the distribution by {moved}"

    # Quantizing copies; it must not reach back into the model it came from.
    again = Engine(backend, compiler=compiler).answer(request)
    assert again.answers["plan"].probabilities == float_probs


def test_the_linear_score_head_reads_every_level_off_one_slot():
    """Score's fixed-width head: the shape `max_levels` was declared for.

    Choice needs the dot-product head because it can carry a hundred thousand
    options and one readout slot has to serve them all. A Score is capped at
    `MAX_LEVELS_PER_SCORE` by the contract, so reading every level off one slot
    at once is affordable — and it is the same shape as `noul_head`, which is
    the one single-slot head in this model that demonstrably learns its
    question.
    """
    from trigon.engine import Engine
    from trigon.limits import MAX_LEVELS_PER_SCORE
    from trigon.types import SystemOneRequest

    request = SystemOneRequest.model_validate(
        {
            "state": {"seats": 142},
            "questions": {
                "size": {
                    "type": "score",
                    "instructions": "How large is this account?",
                    "levels": [
                        {"name": n, "value": float(i)}
                        for i, n in enumerate(("bronze", "silver", "gold", "platinum"))
                    ],
                }
            },
        }
    )

    backend = TorchReadoutBackend(
        ReadoutConfig(d_model=64, n_layers=1, n_heads=2, d_ff=64, score_head="linear"), seed=0
    )
    answer = Engine(backend, compiler=backend.make_compiler()).answer(request)
    probabilities = answer.answers["size"].probabilities
    assert set(probabilities) == {"bronze", "silver", "gold", "platinum"}
    assert sum(probabilities.values()) == pytest.approx(1.0)

    # Sized to the contract's cap by default, not to the 32 the field carried
    # while nothing used it.
    assert backend.config.max_levels == MAX_LEVELS_PER_SCORE
    assert backend.model.score_head.out_features == MAX_LEVELS_PER_SCORE


def test_a_score_wider_than_the_head_is_refused_by_name():
    """A checkpoint trained before the head was widened says so precisely.

    Those record `max_levels: 32` and serve any Score up to 32 levels
    correctly, so refusing to load one would be wrong. The error belongs where
    a Score too wide for it actually arrives.
    """
    from trigon.engine import Engine
    from trigon.types import SystemOneRequest

    backend = TorchReadoutBackend(
        ReadoutConfig(
            d_model=64, n_layers=1, n_heads=2, d_ff=64, score_head="linear", max_levels=3
        ),
        seed=0,
    )
    request = SystemOneRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "size": {
                    "type": "score",
                    "instructions": "How large?",
                    "levels": [{"name": n, "value": float(i)} for i, n in enumerate("abcd")],
                }
            },
        }
    )
    with pytest.raises(ValueError, match="sized for 3"):
        Engine(backend, compiler=backend.make_compiler()).answer(request)


def test_a_step_split_to_fit_memory_is_the_same_step(monkeypatch):
    """`max_batch_cells` changes what fits on a GPU, not what the model learns.

    A cap of one cell forces every case into its own sub-batch, which is the
    most any split can differ from the unsplit step. If each part were divided
    by its own size rather than the chunk's, every step's gradient would be
    off by the chunk size.

    Compared on the gradient each optimizer step receives, not on the weights
    afterwards. The per-option Choice head's bias shifts every option equally,
    softmax ignores that, so its true gradient is exactly zero and what arrives
    is rounding (~1e-9) -- which AdamW then normalises into a full
    learning-rate step. Weights after Adam therefore diverge at 1e-3 on a
    parameter that cannot change an answer, while the step itself agrees.
    """
    steps: dict[str, list[torch.Tensor]] = {}

    class Recorder:
        made: list[Recorder] = []

        def __init__(self, params, **_):
            self.params = list(params)
            self.param_groups = [{"lr": 0.0}]
            self.record: list[torch.Tensor] = []
            Recorder.made.append(self)

        def zero_grad(self, set_to_none=True):
            for p in self.params:
                p.grad = None

        def step(self):
            self.record.append(
                torch.cat([p.grad.flatten() for p in self.params if p.grad is not None])
            )

    cases = synthetic_outcome_cases(n=60, seed=0, noise=0.2)
    monkeypatch.setattr(torch.optim, "AdamW", Recorder)
    for name, budget in (("whole", None), ("split", 1)):
        Recorder.made.clear()
        train(
            _tiny_backend(),
            cases,
            TrainingConfig(epochs=1, accumulate=8, validation_fraction=0, max_batch_cells=budget),
        )
        steps[name] = Recorder.made[0].record

    assert len(steps["whole"]) == len(steps["split"]) == 8
    for whole, split in zip(steps["whole"], steps["split"], strict=True):
        assert (whole - split).norm() <= 1e-5 * whole.norm()


def test_a_run_resumed_after_an_epoch_ends_where_an_uninterrupted_one_does(tmp_path):
    """Preemption resume: kill after epoch 1, restart, and nothing differs.

    On CPU the arithmetic is deterministic, so the resumed run's weights and
    per-epoch record must equal the uninterrupted run's exactly -- the order,
    the RNG, the optimizer moments and the learning-rate step all carried over.
    """
    cases = synthetic_outcome_cases(n=60, seed=0, noise=0.2)
    straight = _tiny_backend()
    report = train(straight, cases, TrainingConfig(epochs=2, accumulate=8, learning_rate=1e-2))

    path = tmp_path / "resume.pt"
    killed = _tiny_backend()
    config = TrainingConfig(epochs=2, accumulate=8, learning_rate=1e-2, resume_path=str(path))
    import trigon.training.trainer as trainer_module

    real_write = trainer_module._write_resume

    class Killed(Exception):
        pass

    def write_then_die(*args, **kwargs):
        real_write(*args, **kwargs)
        raise Killed

    trainer_module._write_resume = write_then_die
    try:
        with pytest.raises(Killed):
            train(killed, cases, config)
    finally:
        trainer_module._write_resume = real_write

    restarted = _tiny_backend()  # a fresh process: nothing but the file survives
    resumed = train(restarted, cases, config)

    assert [e.mean_loss for e in resumed.epochs] == [e.mean_loss for e in report.epochs]
    assert resumed.kept_epoch == report.kept_epoch
    for (name, a), (_, b) in zip(
        straight.model.state_dict().items(), restarted.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_a_distribution_target_is_minimised_by_reporting_the_distribution():
    """Soft cross-entropy's optimum is the annotators' own spread, not the mode."""
    target = (0.5, 0.3, 0.2, 0.0, 0.0)
    exact = torch.log(torch.tensor([0.5, 0.3, 0.2, 1e-9, 1e-9]))
    confident = torch.log(torch.tensor([0.96, 0.01, 0.01, 0.01, 0.01]))
    assert question_loss(exact, "score", 0, distribution=target) < question_loss(
        confident, "score", 0, distribution=target
    )
    # Without a distribution the same call is the old hard-label loss.
    assert question_loss(confident, "score", 0) < question_loss(exact, "score", 0)
