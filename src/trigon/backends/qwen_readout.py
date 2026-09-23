"""A pretrained Qwen2 backbone converted to the prefix-only readout layout.

`docs/next.md` A.3. The reference model is a 128-wide, two-layer spike, and
every accuracy number in `reports/` shares it as a confound. This puts a real
pretrained transformer under the same contract without changing anything the
contract rests on:

* **The mask is ours.** The Qwen2 forward is written here -- RMSNorm, rotary
  positions, grouped-query attention, a gated MLP -- rather than borrowed from
  `transformers`, so the attention mask handed to it is the compiler's block
  mask verbatim, boolean, "may attend". A library that rebuilds its own causal
  mask from ours is the one place a leak between questions could hide, and
  `tests/test_qwen_backend.py` re-asserts the independence claims on this
  forward for exactly that reason. `scripts/backbone_parity.py` checks the
  arithmetic against `transformers` on the real weights under a causal mask.
* **Positions are group-local**, as in the spike: each schema block, the
  state and each readout group starts at 0. Rotary embeddings take any
  position ids, so this costs nothing, and it is what makes a schema's keys
  identical in every request -- the cacheability claim.
* **Conversion, not reinvention.** A causal decoder becomes a prefix encoder by
  being shown bidirectional masks and fine-tuned under them. The pretrained
  weights stay frozen in bf16; low-rank adapters on every projection, the
  segment embedding, the readout vector and the heads are what train. A
  checkpoint is those plus the backbone's pinned name -- about 100 MB, not 3 GB.

Everything above the model -- compilation, the heads, the trainer, the
calibrators, the gates, the KV prefix cache -- is `TorchReadoutBackend`'s,
unchanged. Training and serving share one forward path here exactly as they
do for the spike.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .hf_bpe import ByteLevelBPE
from .tokenizer import Tokenizer, build_tokenizer, describe
from .torch_readout import (
    ReadoutConfig,
    SchemaPrefix,
    TorchReadoutBackend,
    _to_int8_and_back,
)

__all__ = ["QwenPrefillModel", "QwenReadoutBackend", "QwenShape"]

ADAPTER_FORMAT = "trigon-backbone-adapter-1"
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class QwenShape:
    """The architecture numbers a Qwen2 `config.json` declares."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        rope_theta: float = 1_000_000.0,
        eps: float = 1e-6,
    ) -> None:
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError("heads must divide the width and the kv heads the heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.eps = eps

    @classmethod
    def from_config(cls, config: dict) -> QwenShape:
        return cls(
            vocab_size=config["vocab_size"],
            d_model=config["hidden_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config["num_key_value_heads"],
            d_ff=config["intermediate_size"],
            rope_theta=config.get("rope_theta", 10_000.0),
            eps=config.get("rms_norm_eps", 1e-6),
        )

    def to_dict(self) -> dict:
        return dict(vars(self))


class RMSNorm(nn.Module):
    """Qwen2's RMSNorm, with the same float32 detour and the same cast order."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class LoRALinear(nn.Module):
    """A frozen projection plus a trainable low-rank update, zero at init.

    Zero at init is the property that matters: before the first step the
    converted model computes exactly what the pretrained one did, so whatever
    the fine-tune changes, it changes from the backbone and not from noise.
    """

    def __init__(self, width_in: int, width_out: int, *, bias: bool, rank: int, alpha: float):
        super().__init__()
        self.base = nn.Linear(width_in, width_out, bias=bias)
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank if rank else 0.0
        if rank:
            self.lora_a = nn.Parameter(torch.empty(rank, width_in))
            self.lora_b = nn.Parameter(torch.zeros(width_out, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def reset_adapter(self) -> None:
        if self.rank:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.rank:
            out = out + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class QwenAttention(nn.Module):
    def __init__(self, shape: QwenShape, rank: int, alpha: float) -> None:
        super().__init__()
        self.n_heads = shape.n_heads
        self.n_kv = shape.n_kv_heads
        self.head_dim = shape.d_model // shape.n_heads
        d, kv = shape.d_model, shape.n_kv_heads * self.head_dim
        self.q_proj = LoRALinear(d, d, bias=True, rank=rank, alpha=alpha)
        self.k_proj = LoRALinear(d, kv, bias=True, rank=rank, alpha=alpha)
        self.v_proj = LoRALinear(d, kv, bias=True, rank=rank, alpha=alpha)
        self.o_proj = LoRALinear(d, d, bias=False, rank=rank, alpha=alpha)
        inv_freq = 1.0 / (
            shape.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def qkv(self, x: torch.Tensor, positions: torch.Tensor):
        """Projected, rotated queries, keys and values: (B, heads, T, head_dim)."""
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv, self.head_dim).transpose(1, 2)
        freqs = positions[..., None].float() * self.inv_freq
        angles = torch.cat((freqs, freqs), dim=-1)[:, None]
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        return q, k, v

    def attend(self, q, k, v, mask: torch.Tensor) -> torch.Tensor:
        """``mask`` is (B, Tq, Tk) boolean, True where a query may attend."""
        repeat = self.n_heads // self.n_kv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
        batch, _, length, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, -1))


class QwenMLP(nn.Module):
    def __init__(self, shape: QwenShape, rank: int, alpha: float) -> None:
        super().__init__()
        d, ff = shape.d_model, shape.d_ff
        self.gate_proj = LoRALinear(d, ff, bias=False, rank=rank, alpha=alpha)
        self.up_proj = LoRALinear(d, ff, bias=False, rank=rank, alpha=alpha)
        self.down_proj = LoRALinear(ff, d, bias=False, rank=rank, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenLayer(nn.Module):
    def __init__(self, shape: QwenShape, rank: int, alpha: float) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(shape.d_model, shape.eps)
        self.self_attn = QwenAttention(shape, rank, alpha)
        self.post_attention_layernorm = RMSNorm(shape.d_model, shape.eps)
        self.mlp = QwenMLP(shape, rank, alpha)

    def forward(self, hidden, positions, mask):
        q, k, v = self.self_attn.qkv(self.input_layernorm(hidden), positions)
        hidden = hidden + self.self_attn.attend(q, k, v, mask)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class QwenPrefillModel(nn.Module):
    """Qwen2 under a caller-supplied block mask, with the readout heads on top.

    Exposes what `TorchReadoutBackend` calls on `PrefillOnlyModel` -- `embed`,
    `forward`, `encode_prefix`, `forward_with_prefix` and the heads -- so every
    line above the model is shared with the spike.
    """

    def __init__(
        self,
        shape: QwenShape,
        *,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        max_levels: int = 64,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        d = shape.d_model
        self.embed = nn.Embedding(shape.vocab_size, d)
        self.embed.requires_grad_(False)
        self.layers = nn.ModuleList(
            QwenLayer(shape, lora_rank, lora_alpha) for _ in range(shape.n_layers)
        )
        self.norm = RMSNorm(d, shape.eps)
        self.norm.requires_grad_(False)
        for layer in self.layers:
            layer.input_layernorm.requires_grad_(False)
            layer.post_attention_layernorm.requires_grad_(False)
        # New parameters, all trainable. The segment embedding starts at zero
        # so the converted model's first forward is the backbone's own.
        self.segment_embed = nn.Embedding(3, d)
        self.readout = nn.Parameter(torch.zeros(d))
        self.choice_head = nn.Linear(d, 1)
        self.noul_head = nn.Linear(d, 1)
        self.match_query = nn.Linear(d, d, bias=False)
        self.match_key = nn.Linear(d, d, bias=False)
        self.match_log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.score_head = nn.Linear(d, max_levels)
        #: Recompute each layer in the backward pass instead of storing its
        #: activations. Twenty-eight layers of a 1.5B model at a few hundred
        #: tokens times a chunk of eight does not fit a 24 GB card otherwise.
        self.checkpointing = False
        self.reset_trainable()

    def reset_trainable(self) -> None:
        """Initialise everything that trains; the backbone is loaded, not drawn."""
        nn.init.zeros_(self.segment_embed.weight)
        nn.init.normal_(self.readout, std=0.02)
        for head in (self.choice_head, self.noul_head, self.match_query, self.match_key):
            head.reset_parameters()
        self.score_head.reset_parameters()
        with torch.no_grad():
            self.match_log_scale.fill_(math.log(1.0 / 0.07))
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.reset_adapter()

    def _autocast(self, like: torch.Tensor):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=like.is_cuda)

    @staticmethod
    def _batched(token_embeddings, mask, positions, segment_types):
        if positions.dim() == 1:
            positions, segment_types = positions[None], segment_types[None]
        if token_embeddings.dim() == 2:
            token_embeddings = token_embeddings[None]
        if mask.dim() == 2:
            mask = mask[None]
        return token_embeddings, mask, positions, segment_types

    def forward(self, token_embeddings, mask, positions, segment_types) -> torch.Tensor:
        embeddings, mask, positions, segments = self._batched(
            token_embeddings, mask, positions, segment_types
        )
        hidden = embeddings + self.segment_embed(segments)
        with self._autocast(hidden):
            for layer in self.layers:
                if self.training and self.checkpointing:
                    hidden = checkpoint(layer, hidden, positions, mask, use_reentrant=False)
                else:
                    hidden = layer(hidden, positions, mask)
        return self.norm(hidden.float())

    def encode_prefix(
        self, token_embeddings, mask, positions, segment_types, tokens: int, schema_hash: str
    ) -> SchemaPrefix:
        """Run the schema block alone and keep each layer's rotated keys and values.

        Correct because the block mask forbids a schema token from attending to
        anything outside the schema, so running it alone computes what running
        the whole sequence would -- the same property the independence tests
        assert, used rather than checked.
        """
        embeddings, mask, positions, segments = self._batched(
            token_embeddings, mask, positions, segment_types
        )
        hidden = (embeddings + self.segment_embed(segments))[:, :tokens]
        pos, block = positions[:, :tokens], mask[:, :tokens, :tokens]
        captured = []
        with self._autocast(hidden):
            for layer in self.layers:
                q, k, v = layer.self_attn.qkv(layer.input_layernorm(hidden), pos)
                captured.append((k.detach(), v.detach()))
                hidden = hidden + layer.self_attn.attend(q, k, v, block)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return SchemaPrefix(schema_hash, captured, self.norm(hidden.float()).detach(), tokens)

    def forward_with_prefix(
        self, token_embeddings, mask, positions, segment_types, prefix: SchemaPrefix
    ) -> torch.Tensor:
        embeddings, mask, positions, segments = self._batched(
            token_embeddings, mask, positions, segment_types
        )
        n = prefix.tokens
        hidden = (embeddings + self.segment_embed(segments))[:, n:]
        pos, block = positions[:, n:], mask[:, n:, :]
        with self._autocast(hidden):
            for layer, (cached_k, cached_v) in zip(self.layers, prefix.layers, strict=True):
                q, k, v = layer.self_attn.qkv(layer.input_layernorm(hidden), pos)
                k = torch.cat((cached_k, k), dim=2)
                v = torch.cat((cached_v, v), dim=2)
                hidden = hidden + layer.self_attn.attend(q, k, v, block)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return torch.cat((prefix.outputs, self.norm(hidden.float())), dim=1)

    # -- weights ---------------------------------------------------------

    def trainable_state(self) -> dict[str, torch.Tensor]:
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def load_backbone(self, tensors: dict[str, torch.Tensor]) -> None:
        """Copy a Qwen2 safetensors state into the frozen half of this model."""
        mapped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            if name == "lm_head.weight":
                continue  # tied to the embedding in these checkpoints; unused here
            key = name.removeprefix("model.")
            key = key.replace("embed_tokens.", "embed.")
            for proj in _PROJECTIONS:
                key = key.replace(f"{proj}.weight", f"{proj}.base.weight")
                key = key.replace(f"{proj}.bias", f"{proj}.base.bias")
            mapped[key] = tensor
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if unexpected:
            raise ValueError(f"backbone tensors this model has no place for: {unexpected[:5]}")
        frozen = {name for name, p in self.named_parameters() if not p.requires_grad}
        absent = sorted(frozen & set(missing))
        if absent:
            raise ValueError(f"backbone is missing frozen weights: {absent[:5]}")


class QwenReadoutBackend(TorchReadoutBackend):
    """`TorchReadoutBackend` with a pretrained Qwen2 underneath."""

    def __init__(
        self,
        model: QwenPrefillModel,
        tokenizer: Tokenizer,
        *,
        backbone: str | None,
        config: ReadoutConfig | None = None,
        version: str | None = None,
        cache_prefixes: bool = False,
    ) -> None:
        shape = model.shape
        config = config or ReadoutConfig()
        # The model's shape wins over whatever a config carried. A config
        # rebuilt from a checkpoint's head flags defaults to the spike's
        # width, and the dot-product head scales by sqrt(d_model): a reloaded
        # adapter answered differently from the one that was saved until this
        # line existed.
        config.d_model = shape.d_model
        config.n_layers = shape.n_layers
        config.n_heads = shape.n_heads
        config.d_ff = shape.d_ff
        self.backbone = backbone
        super().__init__(
            config,
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            version=version or f"trigon-{backbone or 'qwen2-custom'}-0.1.0-untrained",
            seed=None,
            cache_prefixes=cache_prefixes,
        )

    # -- construction ----------------------------------------------------

    @classmethod
    def from_backbone(
        cls,
        name: str,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        config: ReadoutConfig | None = None,
        base_dtype: torch.dtype | None = None,
    ) -> QwenReadoutBackend:
        """Load a pinned backbone's weights and attach freshly drawn trainables.

        Built on the meta device and materialised once on ``device``, so a
        1.5B model never exists twice -- once initialised at random, once
        loaded -- in a container's memory.
        """
        from safetensors.torch import load_file

        from .hub import BACKBONES, fetch

        backbone = BACKBONES[name]
        shape = QwenShape.from_config(json.loads(fetch(backbone, "config.json").read_text()))
        device = torch.device(device)
        base_dtype = base_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        max_levels = (config or ReadoutConfig()).max_levels
        with torch.device("meta"):
            model = QwenPrefillModel(
                shape, lora_rank=lora_rank, lora_alpha=lora_alpha, max_levels=max_levels
            )
        model.to_empty(device=device)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.base.to(base_dtype)
        for layer in model.layers:
            inv = layer.self_attn.inv_freq
            head_dim = layer.self_attn.head_dim
            inv.copy_(
                1.0
                / (
                    shape.rope_theta
                    ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
                )
            )
        weights = fetch(backbone, "model.safetensors")
        model.load_backbone(load_file(str(weights), device=str(device)))
        torch.manual_seed(seed)
        model.reset_trainable()
        return cls(model, ByteLevelBPE.for_backbone(name), backbone=name, config=config)

    # -- naming ----------------------------------------------------------

    def stamp_version(self) -> str:
        """Named after the backbone's revision and the weights that trained.

        Fingerprinting all 1.5B parameters would hash 3 GB to learn what the
        backbone's pinned revision already says.
        """
        if "+" not in self._version:
            digest = hashlib.blake2b(digest_size=4)
            if self.backbone:
                from .hub import BACKBONES

                digest.update(BACKBONES[self.backbone].revision.encode())
            for key, tensor in sorted(self.model.trainable_state().items()):
                digest.update(key.encode())
                digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
            self._version = f"trigon-{self.backbone or 'qwen2-custom'}-0.1.0+{digest.hexdigest()}"
        return self._version

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """The trainables and the backbone's name -- never the backbone itself."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model: QwenPrefillModel = self.model  # type: ignore[assignment]
        payload = {
            "format": ADAPTER_FORMAT,
            "version": self.stamp_version(),
            "backbone": self.backbone,
            "shape": model.shape.to_dict(),
            "lora": {"rank": model.lora_rank, "alpha": model.lora_alpha},
            "config": {
                "max_levels": self.config.max_levels,
                "score_head": self.config.score_head,
                "match_normalize": self.config.match_normalize,
                "match_residual": self.config.match_residual,
                "match_residual_score": self.config.match_residual_score,
            },
            "tokenizer": describe(self.tokenizer),
            "trainable": {k: v.detach().cpu() for k, v in model.trainable_state().items()},
        }
        if self.backbone is None:
            # A custom shape has no pinned weights to refetch, so it carries
            # its own -- which is what lets the CPU tests round-trip one.
            payload["frozen"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(payload, target)

    @classmethod
    def from_payload(
        cls, payload: dict, *, version: str | None = None, device: str | torch.device = "cpu"
    ) -> QwenReadoutBackend:
        stored = payload["config"]
        if payload["backbone"]:
            backend = cls.from_backbone(
                payload["backbone"],
                device=device,
                lora_rank=payload["lora"]["rank"],
                lora_alpha=payload["lora"]["alpha"],
                config=ReadoutConfig(**stored),
            )
        else:
            model = QwenPrefillModel(
                QwenShape(**payload["shape"]),
                lora_rank=payload["lora"]["rank"],
                lora_alpha=payload["lora"]["alpha"],
                max_levels=stored["max_levels"],
            )
            model.load_state_dict(payload["frozen"])
            backend = cls(
                model,
                build_tokenizer(payload["tokenizer"]),
                backbone=None,
                config=ReadoutConfig(**stored),
            )
        model: QwenPrefillModel = backend.model  # type: ignore[assignment]
        missing, unexpected = model.load_state_dict(
            {k: v.to(device) for k, v in payload["trainable"].items()}, strict=False
        )
        if unexpected:
            raise ValueError(f"adapter checkpoint has tensors this model lacks: {unexpected[:5]}")
        model.eval()
        backend._version = version or payload["version"]
        return backend

    # -- the int8 proxy --------------------------------------------------

    def quantized(self) -> QwenReadoutBackend:
        """A twin whose frozen projections are rounded onto the int8 grid.

        The same weight-only, per-channel proxy the spike's gate reads, applied
        to the backbone -- which is where a deployment would quantize.
        """
        twin_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for module in twin_model.modules():
                if isinstance(module, LoRALinear):
                    weight = module.base.weight
                    weight.copy_(_to_int8_and_back(weight.float()).to(weight.dtype))
        twin = QwenReadoutBackend(
            twin_model,  # type: ignore[arg-type]
            self.tokenizer,
            backbone=self.backbone,
            config=self.config,
            version=f"{self._version}+int8",
        )
        twin.model.eval()
        return twin
