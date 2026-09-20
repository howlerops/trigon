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
    assert report.n_cases == 120
    assert report.n_questions == 360  # three answerable questions per case


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
