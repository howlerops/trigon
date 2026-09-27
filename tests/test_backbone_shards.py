"""A sharded backbone loads the same as a single-file one, on the CPU.

Qwen2.5-7B ships four safetensors files and an index; the smaller models ship
one file. The loader was written against one, and a 15 GB download has no
place in a unit test -- so this builds a tiny Qwen2 state, writes it in the
checkpoint's own naming as fake shards under a fake pinned revision, and
checks that what `from_backbone`'s path loads is every tensor, exactly.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="the backbone backend needs the 'train' extra")
safetensors_torch = pytest.importorskip("safetensors.torch")

from trigon.backends import hub  # noqa: E402
from trigon.backends.qwen_readout import (  # noqa: E402
    QwenPrefillModel,
    QwenShape,
    backbone_shards,
)

SHAPE = dict(vocab_size=50, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=24)


def _hf_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """A Qwen2 checkpoint's tensors, named as `transformers` names them, untied."""
    g = torch.Generator().manual_seed(seed)
    d, ff = SHAPE["d_model"], SHAPE["d_ff"]
    kv = SHAPE["n_kv_heads"] * (d // SHAPE["n_heads"])

    def r(*size):
        return torch.randn(*size, generator=g).to(torch.bfloat16)

    state = {"model.embed_tokens.weight": r(SHAPE["vocab_size"], d), "model.norm.weight": r(d)}
    for i in range(SHAPE["n_layers"]):
        p = f"model.layers.{i}."
        state |= {
            p + "input_layernorm.weight": r(d),
            p + "post_attention_layernorm.weight": r(d),
            p + "self_attn.q_proj.weight": r(d, d),
            p + "self_attn.q_proj.bias": r(d),
            p + "self_attn.k_proj.weight": r(kv, d),
            p + "self_attn.k_proj.bias": r(kv),
            p + "self_attn.v_proj.weight": r(kv, d),
            p + "self_attn.v_proj.bias": r(kv),
            p + "self_attn.o_proj.weight": r(d, d),
            p + "mlp.gate_proj.weight": r(ff, d),
            p + "mlp.up_proj.weight": r(ff, d),
            p + "mlp.down_proj.weight": r(d, ff),
        }
    # Untied, as on 7B: its own tensor, which the loader must skip.
    state["lm_head.weight"] = r(SHAPE["vocab_size"], d)
    return state


def _publish(tmp_path, monkeypatch, state, *, shards: int, drop: str | None = None):
    """Lay ``state`` out in the weights cache as a pinned revision would be."""
    monkeypatch.setenv("TRIGON_WEIGHTS_CACHE", str(tmp_path))
    backbone = hub.Backbone("fake-qwen", "fake/Qwen-Tiny", "0" * 40, "Apache-2.0", sharded=True)
    root = tmp_path / "fake--Qwen-Tiny" / backbone.revision
    root.mkdir(parents=True)
    names = sorted(k for k in state if k != drop)
    weight_map = {}
    for i in range(shards):
        shard = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        part = {k: state[k] for k in names[i::shards]}
        safetensors_torch.save_file(part, str(root / shard))
        weight_map |= {k: shard for k in part}
    (root / hub.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return backbone


def _model() -> QwenPrefillModel:
    return QwenPrefillModel(QwenShape(**SHAPE), lora_rank=2, lora_alpha=4.0, max_levels=8)


def test_every_shard_is_found_through_the_index(tmp_path, monkeypatch):
    backbone = _publish(tmp_path, monkeypatch, _hf_state(), shards=3)
    files = hub.weight_files(backbone)
    assert [f.name for f in files] == [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]


def test_sharded_weights_load_exactly_what_one_file_would(tmp_path, monkeypatch):
    state = _hf_state()
    backbone = _publish(tmp_path, monkeypatch, state, shards=3)
    sharded, whole = _model(), _model()
    sharded.load_backbone_shards(backbone_shards(backbone, "cpu"))
    whole.load_backbone(dict(state))
    frozen = {name for name, p in sharded.named_parameters() if not p.requires_grad}
    assert frozen, "the tiny model has a frozen half to load"
    whole_state = whole.state_dict()
    for name, tensor in sharded.state_dict().items():
        if name in frozen:
            assert torch.equal(tensor, whole_state[name]), name
    assert torch.equal(
        sharded.layers[1].mlp.down_proj.base.weight,
        state["model.layers.1.mlp.down_proj.weight"].float(),
    )
    assert torch.equal(sharded.embed.weight, state["model.embed_tokens.weight"].float())


def test_a_tensor_missing_from_every_shard_is_refused(tmp_path, monkeypatch):
    missing = "model.layers.1.self_attn.k_proj.weight"
    backbone = _publish(tmp_path, monkeypatch, _hf_state(), shards=2, drop=missing)
    with pytest.raises(ValueError, match="missing frozen weights"):
        _model().load_backbone_shards(backbone_shards(backbone, "cpu"))


def test_an_index_naming_a_path_is_refused(tmp_path, monkeypatch):
    backbone = _publish(tmp_path, monkeypatch, _hf_state(), shards=1)
    index = tmp_path / "fake--Qwen-Tiny" / backbone.revision / hub.INDEX
    index.write_text(json.dumps({"weight_map": {"x": "../../elsewhere.safetensors"}}))
    with pytest.raises(ValueError, match="unexpected shard"):
        hub.weight_files(backbone)


def test_the_7b_backbone_is_pinned_and_sharded():
    backbone = hub.BACKBONES["qwen2.5-7b"]
    assert backbone.repo == "Qwen/Qwen2.5-7B"
    assert len(backbone.revision) == 40 and backbone.sharded
    assert backbone.licence == "Apache-2.0"
    assert not hub.BACKBONES["qwen2.5-1.5b"].sharded
