"""`train_corpus.py --init-weights` and `scripts/resume_to_checkpoint.py`, on the spike.

Continuing from someone else's weights is the one way a run can start from a
checkpoint it could not have built -- a different backbone, shape or
vocabulary -- and nothing raises when the shapes agree. These pin the
refusals, the naming (a continued build names its init), and that a resume
file converts to the very weights its run kept.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

torch = pytest.importorskip("torch")

from trigon.backends.tokenizer import HashingTokenizer  # noqa: E402
from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend  # noqa: E402
from trigon.evals import synthetic_outcome_cases  # noqa: E402
from trigon.training import TrainingConfig, train  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _script("train_corpus")


def _spike(d_model=32, layers=1, **kwargs):
    return TorchReadoutBackend(config=ReadoutConfig(d_model=d_model, n_layers=layers), **kwargs)


def _saved(tmp_path, backend, name="init.pt"):
    path = tmp_path / name
    backend.save(path)
    return str(path)


def _args(script, path, *extra):
    return script.parse_args(["banking77", "--init-weights", path, *extra])


def test_a_matching_checkpoint_loads_with_its_weights_and_an_unstamped_name(script, tmp_path):
    source = _spike(seed=3)
    path = _saved(tmp_path, source)
    backend, init_version = script.initialise_from(
        _args(script, path, "--d-model", "32", "--layers", "1")
    )
    assert init_version == source.model_version and "+" in init_version
    # Cleared, so the trainer stamps the new weights rather than keeping the init's name.
    assert "+" not in backend.model_version
    for (name, a), (_, b) in zip(
        source.model.state_dict().items(), backend.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


@pytest.mark.parametrize(
    "flags, reason",
    [
        (["--d-model", "64", "--layers", "1"], "d_model 32 x 1 layers"),
        (["--d-model", "32", "--layers", "2"], "d_model 32 x 1 layers"),
        (["--d-model", "32", "--layers", "1", "--backbone", "qwen2.5-1.5b"], "backbone spike"),
    ],
)
def test_a_checkpoint_this_run_could_not_have_built_is_refused(script, tmp_path, flags, reason):
    path = _saved(tmp_path, _spike())
    with pytest.raises(SystemExit, match=reason):
        script.initialise_from(_args(script, path, *flags))


def test_a_checkpoint_under_another_vocabulary_is_refused(script, tmp_path):
    path = _saved(tmp_path, _spike(tokenizer=HashingTokenizer()))
    with pytest.raises(SystemExit, match="tokenizer"):
        script.initialise_from(_args(script, path, "--d-model", "32", "--layers", "1"))


def test_a_custom_qwen_shape_is_not_a_spike(script, tmp_path):
    from trigon.backends.qwen_readout import QwenPrefillModel, QwenReadoutBackend, QwenShape
    from trigon.backends.tokenizer import default_tokenizer

    tokenizer = default_tokenizer()
    shape = QwenShape(
        vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32
    )
    model = QwenPrefillModel(shape, lora_rank=2, lora_alpha=4.0)
    path = _saved(tmp_path, QwenReadoutBackend(model, tokenizer, backbone=None))
    with pytest.raises(SystemExit, match="custom Qwen2 shape"):
        script.initialise_from(_args(script, path))


def test_the_init_is_named_in_the_build(script):
    assert script.init_suffix("trigon-qwen2.5-1.5b-0.1.0+770672fa") == ".init.770672fa"
    assert script.init_suffix("trigon-reference-0.1.0+ab12.int8") == ".init.ab12-int8"


def test_weights_and_init_weights_are_exclusive(script, tmp_path):
    path = _saved(tmp_path, _spike())
    with pytest.raises(SystemExit, match="Pick one"):
        script.main(["banking77", "--weights", path, "--init-weights", path])


def test_a_resume_file_converts_to_the_weights_its_run_kept(tmp_path):
    convert = _script("resume_to_checkpoint")
    cases = synthetic_outcome_cases(n=60, seed=0, noise=0.2)
    trained = _spike(seed=0)
    resume = tmp_path / "run.resume.pt"
    report = train(
        trained,
        cases,
        TrainingConfig(
            epochs=2,
            accumulate=8,
            learning_rate=1e-2,
            resume_path=str(resume),
            validation_fraction=0.2,
        ),
    )
    kept = report.epochs[report.kept_epoch - 1].validation_loss
    out = tmp_path / "converted.pt"
    base = ["--out", str(out), "--d-model", "32", "--layers", "1"]
    with pytest.raises(SystemExit, match="kept epoch"):
        convert.main([str(resume), *base, "--expect-epoch", str(report.kept_epoch + 1)])
    with pytest.raises(SystemExit, match="validation loss"):
        convert.main([str(resume), *base, "--expect-validation", str(kept + 0.5)])
    with pytest.raises(SystemExit, match="does not fit"):
        convert.main([str(resume), "--out", str(out), "--d-model", "32", "--layers", "2"])

    convert.main(
        [
            str(resume),
            *base,
            "--expect-epoch",
            str(report.kept_epoch),
            "--expect-validation",
            repr(kept),
        ]
    )
    loaded = TorchReadoutBackend.load(out)
    for (name, a), (_, b) in zip(
        trained.model.state_dict().items(), loaded.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    assert loaded.model_version == trained.model_version
