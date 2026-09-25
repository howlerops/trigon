"""Reference implementation of the prefill-only readout architecture.

This is the phase-1 spike made executable: one forward pass, no decode loop,
every question answered in parallel, and the isolation rules enforced by an
additive attention mask rather than by hoping the model behaves. It is small
and untrained -- its logits mean nothing -- but its *structure* is the thing
under test, and ``tests/test_independence.py`` uses it to prove the two claims
the product rests on:

* adding a question does not move any other question's logits, at all, to
  floating-point equality;
* the schema half of the sequence encodes identically regardless of state,
  which is what makes it a cacheable cross-request KV prefix.

Both claims are properties of the layout, so they hold for an untrained model
exactly as they will for a trained one. Training changes the numbers; it cannot
change which tokens a readout was allowed to see.

One finding from building this that is easy to miss on paper: the mask alone
does *not* buy independence. With ordinary sequence-global position encodings,
inserting a question shifts every later token's position, so a question's
hidden states move even though the mask never let it see the new question.
Positions here are therefore **group-local** -- each schema block, the state,
and each question's readout slots all start at position 0 -- and a learned
segment-type embedding keeps the three kinds distinguishable despite the
overlapping indices. Group-local positions are also what make a schema block's
KV genuinely portable between requests, which is the whole point of caching it:
a cached prefix computed at one offset is wrong at another.

Heads, all categorical, none of them a regression:

* Choice with ``READOUT_PER_OPTION``: one readout slot per option, seeded with
  that option's mean input embedding so the slot knows which option it is, then
  projected to a scalar logit.
* Choice with ``DOT_PRODUCT`` and every Score: one readout slot for the whole
  question, scored against the pooled encoder states of each option or level.
  One slot regardless of cardinality -- this is what makes large option sets
  affordable.
* Noul: one readout slot, one linear logit.

And one head that is not an answer: **evidence**. Asked for, each question's
answer comes back with the spans of the state that drove it. Two sources, and
the response names which one answered (`evidence_method`):

* ``span_head`` -- a small bilinear head scoring every state token against the
  question's readout state, trained on human rationales when a case carries
  them (`Expectation.rationale`). Used only by a checkpoint that was;
* ``gradient_x_input`` -- otherwise. The gradient of the selected label's
  log-probability with respect to each state token's input embedding, dotted
  with that embedding. An untrained head is not evidence, and a model that was
  never shown a rationale has no business returning one from a head.

Both read only what the question's answer reads -- the state's hidden states
and the question's own readout -- so evidence inherits the independence the
mask gives the answer. `tests/test_independence.py` asserts it for both.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path

try:  # pragma: no cover - exercised by the import error path only
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "the torch readout backend needs the 'train' extra: pip install 'trigon[train]'"
    ) from exc

from ..limits import MAX_LEVELS_PER_SCORE
from ..schema import (
    CompiledRequest,
    OptionScoring,
    SchemaCompiler,
    SegmentKind,
    mask_shape_key,
)
from ..schema.compiler import MASK_CACHE_CELLS
from ..schema.tokens import CallableEstimator
from ..types import SystemOneRequest
from .base import BackendOutput, QuestionOutput
from .tokenizer import READOUT_ID, Tokenizer, build_tokenizer, default_tokenizer, describe

__all__ = ["PrefillOnlyModel", "ReadoutConfig", "TorchReadoutBackend"]

TRAINED_VERSION_PREFIX = "trigon-reference-0.1.0"
#: Width of the evidence head's bilinear space. Small on purpose: it reads one
#: bit per token -- in the rationale or not -- and is the one head that runs
#: over every state token of every question.
EVIDENCE_WIDTH = 64
#: `TorchReadoutBackend.evidence_mode`'s values.
EVIDENCE_MODES = ("auto", "span_head", "gradient_x_input")
# A randomly initialised model answers every question with noise. It says so in
# its own version string, because ``model_version`` travels in the response and
# is the only thing a caller downstream has to go on.
UNTRAINED_VERSION = f"{TRAINED_VERSION_PREFIX}-untrained"


def _weights_fingerprint(model) -> str:
    """Eight hex characters over every parameter, so two checkpoints differ."""
    digest = hashlib.blake2b(digest_size=4)
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ReadoutConfig:
    """Shape of the reference model. Defaults are spike-sized, not ship-sized."""

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        max_levels: int = MAX_LEVELS_PER_SCORE,
        score_head: str = "dotproduct",
        match_normalize: bool = False,
        match_residual: bool = True,
        match_residual_score: bool = False,
        evidence_supervised: bool = False,
    ) -> None:
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must divide by n_heads {n_heads}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        # Defaults to the contract's cap. It was 32 while the contract accepts
        # 64, which nothing noticed because the field was stored and serialized
        # and never used to build anything; it sizes `score_head` now.
        #
        # Deliberately not validated here. Checkpoints written before this head
        # existed record 32, and refusing to load one outright would be wrong:
        # it serves any Score up to 32 levels correctly. `_heads` raises
        # instead, naming the checkpoint's own cap, at the point where a Score
        # too wide for it actually arrives.
        self.max_levels = max_levels
        if score_head not in ("dotproduct", "linear"):
            raise ValueError(f"score_head must be 'dotproduct' or 'linear', got {score_head!r}")
        self.score_head = score_head
        # ``match_residual`` defaults ON: without it the dot-product head
        # collapses to the marginal and answers 0.254 against a 0.253 marginal
        # predictor; with it, 0.847. ``match_normalize`` defaults OFF: it does
        # nothing alone and cancels most of the residual's gain. Both measured
        # once, at spike scale -- see docs/decisions.md.
        self.match_normalize = match_normalize
        self.match_residual = match_residual
        # Off by default, and separately switchable, because the one
        # measurement of it regressed a run -- see `_heads`. It exists so the
        # question can be answered rather than argued about.
        self.match_residual_score = match_residual_score
        #: Set by the trainer once the evidence head has been fitted to human
        #: rationales, and saved with the checkpoint. It decides which evidence
        #: a request gets: the head when True, gradient x input when not --
        #: because an evidence head nobody trained is a random projection, and
        #: its spans would look exactly as confident as a trained one's.
        self.evidence_supervised = evidence_supervised


def _sinusoidal(length: int, d_model: int, device, dtype) -> torch.Tensor:
    """Absolute sinusoidal positions -- no learned length ceiling to trip over."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * scale)
    pe[:, 1::2] = torch.cos(position * scale)
    return pe


class SchemaPrefix:
    """A schema block's attention keys and values, computed once.

    The architectural claim this project rests on is that the schema half of
    the sequence encodes identically regardless of state --
    `tests/test_independence.py` asserts it on hidden states. This is what the
    claim is *for*: if those states do not move, neither do the keys and values
    derived from them, so a request carrying the same schema can skip
    recomputing them and attend to these instead.

    What is stored is the *pre-attention normed* hidden state at each layer,
    not the raw hidden state. With `norm_first=True` a layer computes
    `x + attn(norm1(x), norm1(x), norm1(x))`, so `norm1(x)` is the quantity
    the keys and values are projected from. Storing it rather than `x` keeps
    the cached path arithmetically identical to the uncached one instead of
    approximately equal.

    Keyed by `schema_hash`, which the compiler already computes over exactly
    the things a schema block depends on and nothing else.
    """

    __slots__ = ("schema_hash", "layers", "outputs", "tokens")

    def __init__(
        self,
        schema_hash: str,
        layers: list[torch.Tensor],
        outputs: torch.Tensor,
        tokens: int,
    ) -> None:
        self.schema_hash = schema_hash
        self.layers = layers
        #: The schema block's post-encoder states. Cached too, because a
        #: caller reads hidden states at every position -- a Choice's option
        #: keys are pooled from the schema block -- and the first version of
        #: this recomputed them, which meant a "cache" that recomputed
        #: everything it had just cached and saved nothing at all.
        self.outputs = outputs
        self.tokens = tokens

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SchemaPrefix(schema_hash={self.schema_hash!r}, "
            f"tokens={self.tokens}, layers={len(self.layers)})"
        )


class PrefillOnlyModel(nn.Module):
    """A prefix-LM encoder with a caller-supplied attention mask.

    Deliberately a plain ``nn.TransformerEncoder``: the novelty in this project
    is the mask, the heads and the training objective, not the block.
    """

    def __init__(self, config: ReadoutConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        # schema / state / readout. Positions restart per group, so without
        # this the model cannot tell state token 3 from schema token 3.
        self.segment_embed = nn.Embedding(3, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        # Choice with a slot per option, and Noul, both reduce one state to one
        # logit; they use separate heads because the calibration targets differ.
        self.choice_head = nn.Linear(config.d_model, 1)
        self.noul_head = nn.Linear(config.d_model, 1)
        # Projects a readout state before it is dotted with pooled member
        # states, so the dot-product head is not forced to use raw geometry.
        self.match_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.match_key = nn.Linear(config.d_model, config.d_model, bias=False)
        # CLIP's learnable logit scale, used only when match_normalize is on.
        # Held in log space so it stays positive under unconstrained descent;
        # exp(2.66) ~ 1477/1000, i.e. the 1/0.07 CLIP initialises to.
        self.match_log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        # A fixed-width head for Score, the shape `max_levels` was declared for
        # and never used. Choice needs the dot-product head because it can
        # carry a hundred thousand options and one slot has to serve them all;
        # a Score is capped at MAX_LEVELS_PER_SCORE by the contract, so a head
        # that reads all of them off one slot at once is affordable -- and it
        # is the same shape as `noul_head`, which is the one single-slot head
        # in this model that demonstrably learns its question.
        self.score_head = nn.Linear(config.d_model, config.max_levels)
        # The evidence head, declared last so every parameter above draws the
        # initialisation it drew before the head existed: a seed must keep
        # meaning the same model.
        self.evidence_query = nn.Linear(config.d_model, EVIDENCE_WIDTH, bias=False)
        self.evidence_key = nn.Linear(config.d_model, EVIDENCE_WIDTH, bias=False)
        self.evidence_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        token_embeddings: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        segment_types: torch.Tensor,
    ) -> torch.Tensor:
        """One forward pass.

        ``token_embeddings`` (B, T, d); ``mask`` (T, T) for a single request or
        (B, T, T) when the batch's requests have different layouts, boolean
        "may attend"; ``positions`` and ``segment_types`` (T,) or (B, T), the
        latter group-local indices and the former in {0: schema, 1: state,
        2: readout}.
        """
        batched = positions.dim() == 2
        table = _sinusoidal(
            int(positions.max().item()) + 1,
            self.config.d_model,
            token_embeddings.device,
            token_embeddings.dtype,
        )
        hidden = token_embeddings + table[positions] + self.segment_embed(segment_types)
        if not batched:
            hidden = hidden if hidden.dim() == 3 else hidden.unsqueeze(0)

        # nn.TransformerEncoder takes True as "block this pair".
        attn_mask = ~mask
        if attn_mask.dim() == 3:
            # A per-sample mask has to be repeated once per attention head:
            # the encoder flattens (batch, head) into one leading dimension,
            # and repeat_interleave is what keeps each sample's rows adjacent.
            # `repeat` here instead would silently give sample 0's mask to
            # every head of every sample.
            attn_mask = attn_mask.repeat_interleave(self.config.n_heads, dim=0)
        hidden = self.encoder(hidden, mask=attn_mask)
        return self.norm(hidden)

    # -- cacheable schema prefix -----------------------------------------

    def _embed_inputs(self, token_embeddings, positions, segment_types) -> torch.Tensor:
        table = _sinusoidal(
            int(positions.max().item()) + 1,
            self.config.d_model,
            token_embeddings.device,
            token_embeddings.dtype,
        )
        hidden = token_embeddings + table[positions] + self.segment_embed(segment_types)
        return hidden if hidden.dim() == 3 else hidden.unsqueeze(0)

    def encode_prefix(
        self,
        token_embeddings: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        segment_types: torch.Tensor,
        tokens: int,
        schema_hash: str,
    ) -> SchemaPrefix:
        """Run the first ``tokens`` positions and keep what a later request needs.

        Only correct because the block mask forbids a schema token from
        attending to state or readout tokens, so running the prefix alone
        computes the same states running the whole sequence would. That is the
        same property `tests/test_independence.py` asserts, used rather than
        merely checked.
        """
        hidden = self._embed_inputs(token_embeddings, positions, segment_types)[:, :tokens]
        block = ~mask[:tokens, :tokens]
        captured: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            normed = layer.norm1(hidden)
            captured.append(normed.detach())
            attended, _ = layer.self_attn(
                normed, normed, normed, attn_mask=block, need_weights=False
            )
            hidden = hidden + layer.dropout1(attended)
            hidden = hidden + layer._ff_block(layer.norm2(hidden))
        return SchemaPrefix(schema_hash, captured, hidden.detach(), tokens)

    def forward_with_prefix(
        self,
        token_embeddings: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        segment_types: torch.Tensor,
        prefix: SchemaPrefix,
    ) -> torch.Tensor:
        """Encode a request whose schema block is already computed.

        Returns the full sequence's hidden states, schema included, so callers
        downstream cannot tell the difference -- which is the point, and what
        `tests/test_independence.py` asserts to exact equality.
        """
        n = prefix.tokens
        hidden = self._embed_inputs(token_embeddings, positions, segment_types)
        rest = hidden[:, n:]
        # Rows for the non-schema tokens, columns for the whole sequence. The
        # schema is a prefix in this layout, so the concatenation below is
        # already in the order these columns expect; it is not a coincidence to
        # rely on silently, so `_heads` is given the reassembled sequence.
        block = ~mask[n:, :]
        for index, layer in enumerate(self.encoder.layers):
            normed = layer.norm1(rest)
            keys = torch.cat([prefix.layers[index], normed], dim=1)
            attended, _ = layer.self_attn(normed, keys, keys, attn_mask=block, need_weights=False)
            rest = rest + layer.dropout1(attended)
            rest = rest + layer._ff_block(layer.norm2(rest))

        # The schema's states come straight out of the cache. Nothing in the
        # schema block is recomputed, which is the entire point and is what
        # the first version of this got wrong.
        return self.norm(torch.cat([prefix.outputs, rest], dim=1))


def _to_int8_and_back(weight: torch.Tensor) -> torch.Tensor:
    """Round a weight matrix onto the int8 grid, per output channel.

    Symmetric, per-row scales -- the layout every weight-only int8 kernel
    uses, because one scale for the whole matrix lets a single large row set
    the step size for all the others.

    The result is an fp32 tensor holding only int8-representable values, so a
    forward pass through it computes what an int8 kernel computes up to
    accumulation order. That is the standard way quantization error is
    measured, and it keeps the module an ordinary `nn.Linear`: no packed
    weight, no engine dependency, and no API on a deprecation clock -- the
    gate has to still run in two years, and `torch.ao.quantization` is
    already scheduled for removal.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    # A row of exact zeros has no scale; leave it alone rather than dividing.
    scale = scale.clamp_min(torch.finfo(weight.dtype).tiny)
    return torch.round(weight / scale).clamp_(-127, 127) * scale


class TorchReadoutBackend:
    """Runs a ``PrefillOnlyModel`` over a compiled request."""

    def __init__(
        self,
        config: ReadoutConfig | None = None,
        *,
        model: PrefillOnlyModel | None = None,
        tokenizer: Tokenizer | None = None,
        version: str = UNTRAINED_VERSION,
        seed: int | None = 0,
        cache_prefixes: bool = False,
    ) -> None:
        self.config = config or ReadoutConfig()
        self.tokenizer = tokenizer or default_tokenizer()
        # The vocabulary decides the embedding table, not the other way round:
        # a config carrying a stale vocab_size would index past the table.
        self.config.vocab_size = self.tokenizer.vocab_size
        if seed is not None:
            torch.manual_seed(seed)
        self.model = model or PrefillOnlyModel(self.config)
        self.model.eval()
        self._version = version
        # Mask tensors are shape-keyed like the masks themselves: at spike
        # sizes building one costs more than the forward pass.
        # Bounded in cells rather than entries, for the reason
        # `trigon.schema.compiler` is: a mask is quadratic in sequence length,
        # so a count-based limit bounds nothing. These are bool tensors, one
        # byte a cell, and the compiler holds its own list-of-bool copy under
        # its own budget -- two caches of the same thing in two
        # representations, because the tests read one and torch needs the
        # other.
        self._mask_cache: dict[object, torch.Tensor] = {}
        self._mask_cache_cells = 0
        # The schema KV prefix, the thing the layout was designed to make
        # cacheable. Off by default: a prefix belongs to the weights that
        # produced it, and a gateway is the only place where the weights are
        # fixed for the process's lifetime.
        self.cache_prefixes = cache_prefixes
        self._prefix_cache: dict[str, SchemaPrefix] = {}
        # Reported as `usage.cached_schema_tokens`, set per request by
        # `logits` so the number describes the request the caller just made.
        self._last_cached_tokens = 0
        #: Which evidence a request gets. "auto" is the served behaviour: the
        #: span head if this checkpoint was trained on rationales, gradient x
        #: input otherwise. The other two force one, which is what an eval
        #: comparing them needs; forcing "span_head" on an untrained head is
        #: allowed there and labelled, never served by default.
        self.evidence_mode = "auto"

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def device(self) -> torch.device:
        """Where the weights live, and so where every input tensor is built."""
        return self.model.embed.weight.device

    def to(self, device: str | torch.device) -> TorchReadoutBackend:
        """Move the weights, and drop everything cached on the old device.

        Every tensor this backend builds is created on ``self.device``. Until
        this existed nothing was, and nothing raised: the Modal launcher asked
        for an A10G, recorded the A10G it got, and would have trained on the
        container's CPU -- a report naming hardware it never used.
        """
        self.model.to(device)
        self._mask_cache.clear()
        self._mask_cache_cells = 0
        self._prefix_cache.clear()
        return self

    def stamp_version(self) -> str:
        """Name this build after the weights it actually has.

        Called when training finishes and when a checkpoint is written, so the
        eval report, the checkpoint and the served response all name the same
        model. A no-op once the version carries a fingerprint, so an explicitly
        named build keeps its name.
        """
        if "+" not in self._version:
            self._version = f"{TRAINED_VERSION_PREFIX}+{_weights_fingerprint(self.model)}"
        return self._version

    @property
    def estimator(self) -> CallableEstimator:
        """Exact token counts, so compiled spans match the tensors exactly."""
        return CallableEstimator(self.tokenizer.encode, exact=True)

    def make_compiler(self, **kwargs) -> SchemaCompiler:
        """A compiler wired to this backend's tokenizer."""
        return SchemaCompiler(estimator=self.estimator, **kwargs)

    def quantized(self) -> TorchReadoutBackend:
        """A copy of this backend whose linear weights are int8-representable.

        Weight-only, symmetric, per-output-channel -- every `nn.Linear` in the
        model, encoder and heads alike. Activations stay fp32, which is what
        weight-only quantization means and is the cheap kind that needs no
        calibration pass.

        **What it is a proxy for.** Phase 3's quantization is KV-cache
        bit-width, not weight precision, and this is not that. What the two
        share is the thing the gate is about: a change to the serving numerics
        that leaves argmax almost untouched and can move the probabilities
        underneath it. `quantization_ece_delta` has existed since the first
        eval harness and until now the only thing that ever fed it was a test
        fixture built by perturbing probabilities by hand -- which tests the
        gate, not the system. This makes it a measurement.

        The copy is independent, so quantizing does not disturb this backend,
        and it keeps the same tokenizer: a quantized model served under a
        different vocabulary is two changes at once.
        """
        twin = TorchReadoutBackend(
            self.config,
            tokenizer=self.tokenizer,
            version=f"{self._version}+int8",
            seed=None,
        )
        twin.model.to(self.device)
        twin.model.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for module in twin.model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.copy_(_to_int8_and_back(module.weight))
                # `nn.MultiheadAttention` keeps its input projection as a bare
                # parameter rather than a child `nn.Linear`, so iterating over
                # linears alone would leave three quarters of the attention
                # weights at full precision and quietly understate the delta.
                if isinstance(module, nn.MultiheadAttention):
                    module.in_proj_weight.copy_(_to_int8_and_back(module.in_proj_weight))
        twin.model.eval()
        return twin

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the weights and the shape needed to rebuild them.

        A run that cannot be reloaded cannot be served, and a report about a
        model nobody can run again is a claim rather than a result. The config
        travels with the weights so a checkpoint is self-describing.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Two checkpoints must not answer under one name: /healthz reports
        # trained-ness, but ``model_version`` is the field the response carries,
        # and a caller reading it has to be able to tell which build answered.
        version = self.stamp_version()
        torch.save(
            {
                "version": version,
                "config": {
                    "vocab_size": self.config.vocab_size,
                    "d_model": self.config.d_model,
                    "n_layers": self.config.n_layers,
                    "n_heads": self.config.n_heads,
                    "d_ff": self.config.d_ff,
                    "dropout": self.config.dropout,
                    "max_levels": self.config.max_levels,
                    "match_normalize": self.config.match_normalize,
                    "match_residual": self.config.match_residual,
                    "evidence_supervised": self.config.evidence_supervised,
                },
                "tokenizer": describe(self.tokenizer),
                "state_dict": self.model.state_dict(),
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, *, version: str | None = None) -> TorchReadoutBackend:
        """Rebuild a backend from a checkpoint written by ``save``."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format", "").startswith("trigon-backbone-adapter"):
            # Adapters over a pinned pretrained backbone: the checkpoint names
            # the backbone and carries only what trained.
            from .qwen_readout import QwenReadoutBackend

            # On the GPU when there is one: a 1.5B backbone serves in tens of
            # milliseconds there and in seconds on a CPU.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return QwenReadoutBackend.from_payload(payload, version=version, device=device)
        stored = dict(payload["config"])
        # A flag absent from a checkpoint means "trained before this flag
        # existed", which is False -- never the current constructor default.
        # Otherwise flipping a default silently changes the arithmetic of every
        # model already on disk, and a committed run stops reproducing its own
        # report.
        for flag in ("match_normalize", "match_residual"):
            stored.setdefault(flag, False)
        config = ReadoutConfig(**stored)
        # Rebuild the tokenizer the run was trained with, never today's
        # default: the vocabulary sizes the embedding table, and a mismatch is
        # either a load error or -- worse, if the sizes happen to agree -- a
        # model reading every token as a different word.
        tokenizer = build_tokenizer(payload.get("tokenizer"))
        backend = cls(
            config,
            tokenizer=tokenizer,
            version=version or payload.get("version", TRAINED_VERSION_PREFIX),
        )

        # A checkpoint written before a parameter existed is still a valid
        # checkpoint: adding `match_log_scale` broke loading every run saved
        # before it, the certified model included. Missing parameters take
        # their freshly-initialised value; everything else still has to match
        # exactly, so a truncated or mismatched checkpoint fails as loudly as
        # before rather than loading as a half-random model.
        state = dict(payload["state_dict"])
        for name, value in backend.model.state_dict().items():
            state.setdefault(name, value)
        backend.model.load_state_dict(state)
        backend.model.eval()
        if version is None:
            # An unfingerprinted checkpoint -- written before the weights were
            # part of the name, or by hand -- is stamped from what was actually
            # loaded, so two such checkpoints cannot answer under one version.
            backend.stamp_version()
        return backend

    # -- inference -------------------------------------------------------

    def logits(
        self, compiled: CompiledRequest, request: SystemOneRequest
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Differentiable per-question logits, and the sequence length.

        Training and serving share this one path. A trainer carrying its own
        copy of the forward pass is the standard way to end up with a model
        that scores well offline and is miscalibrated in production, and the
        divergence is invisible until someone measures ECE on the served path.
        """
        embeddings, spans = self._embed(compiled)
        hidden = self._encode(compiled, embeddings, spans)
        return self._heads(compiled, request, hidden, spans), int(embeddings.shape[1])

    def _encode(self, compiled: CompiledRequest, embeddings: torch.Tensor, spans: _Spans):
        """One request's hidden states, through the schema prefix cache if it is on."""
        mask = self._cached_mask(compiled)
        if mask.shape[0] != embeddings.shape[1]:
            raise ValueError(
                f"mask is {mask.shape[0]} tokens but the sequence is "
                f"{embeddings.shape[1]}; the compiler's estimator must be the "
                f"backend's tokenizer"
            )
        positions, segments = spans.positions(self.device), spans.segment_types(self.device)
        self._last_cached_tokens = 0
        prefix = self._prefix_for(compiled, embeddings, mask, positions, segments)
        if prefix is None:
            return self.model(embeddings, mask, positions, segments)[0]
        return self.model.forward_with_prefix(embeddings, mask, positions, segments, prefix)[0]

    # -- evidence --------------------------------------------------------

    def evidence_logits(self, hidden: torch.Tensor, spans: _Spans, qid: str) -> torch.Tensor:
        """The span head: one logit per state token for question ``qid``.

        Bilinear between each state token's final hidden state and the mean of
        the question's readout states. Both are things the question's answer
        already reads and nothing else is: state tokens encode without seeing
        any schema, and a readout sees only its own question's schema. So the
        head is as independent of the other questions as the answer is, by
        the same mask and with no new path to prove.
        """
        query = self.model.evidence_query(hidden[spans.readout[qid]].mean(dim=0))
        keys = self.model.evidence_key(hidden[spans.state])
        return keys @ query / math.sqrt(EVIDENCE_WIDTH) + self.model.evidence_bias

    def state_offsets(self, compiled: CompiledRequest) -> list[tuple[int, int]]:
        """Character offsets of each state token, in the order they are embedded.

        From the tokenizer that built the tensors, and checked against it: an
        offset table one token out of step would put every span one token
        late, and the spans would still look like plausible text.
        """
        text = next(s.text for s in compiled.segments if s.kind is SegmentKind.STATE)
        triples = self.tokenizer.encode_with_offsets(text)
        if [t[0] for t in triples] != self.tokenizer.encode(text):
            raise ValueError("the tokenizer's offsets disagree with its own encoding")
        return [(start, end) for _, start, end in triples]

    def _resolved_evidence_mode(self) -> str:
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"evidence_mode must be one of {EVIDENCE_MODES}")
        if self.evidence_mode != "auto":
            return self.evidence_mode
        return "span_head" if self.config.evidence_supervised else "gradient_x_input"

    def _evidence(
        self,
        compiled: CompiledRequest,
        embeddings: torch.Tensor,
        hidden: torch.Tensor,
        spans: _Spans,
        raw: dict[str, torch.Tensor],
    ) -> tuple[dict[str, tuple[tuple[int, int, float], ...]], str]:
        """Per-question, per-state-token scores in [0, 1], and the method used.

        ``gradient_x_input`` attributes the label the head ranks first -- the
        selected one, since calibration is monotone -- as the gradient of its
        log-probability with respect to each state token's input embedding,
        dotted with that embedding. Only positive attribution is evidence *for*
        the answer; it is scaled by the answer's largest, so the scores say
        which tokens mattered most to this answer and not how much in absolute
        terms. For a Noul the selected side is the one its logit falls on.

        One backward pass per question, each through the same forward graph:
        prefill-only still, one forward, and no decoding.
        """
        offsets = self.state_offsets(compiled)
        if len(offsets) != len(spans.state):
            raise ValueError(f"{len(spans.state)} state tokens embedded but {len(offsets)} offsets")
        method = self._resolved_evidence_mode()
        out: dict[str, tuple[tuple[int, int, float], ...]] = {}
        state = spans.state
        for compiled_q in compiled.schema.questions:
            qid = compiled_q.question_id
            if not state:
                out[qid] = ()
                continue
            if method == "span_head":
                scores = torch.sigmoid(self.evidence_logits(hidden, spans, qid))
            else:
                logits = raw[qid]
                if compiled_q.kind == "noul":
                    target = nn.functional.logsigmoid(logits[0] if logits[0] >= 0 else -logits[0])
                else:
                    target = torch.log_softmax(logits, dim=-1)[int(torch.argmax(logits))]
                (gradient,) = torch.autograd.grad(target, embeddings, retain_graph=True)
                attribution = (gradient[0, state] * embeddings[0, state]).sum(dim=-1)
                attribution = attribution.clamp_min(0.0)
                peak = attribution.max()
                scores = attribution / peak if peak > 0 else torch.zeros_like(attribution)
            values = scores.detach().float().cpu().tolist()
            out[qid] = tuple(
                (start, end, float(value))
                for (start, end), value in zip(offsets, values, strict=True)
            )
        return out, method

    def _cached_mask(self, compiled: CompiledRequest) -> torch.Tensor:
        """The token mask for this request's shape, from the cache if it fits."""
        key = mask_shape_key(compiled)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = self._mask_tensor(compiled)
            cells = mask.shape[0] * mask.shape[1]
            if cells <= MASK_CACHE_CELLS:
                if self._mask_cache_cells + cells > MASK_CACHE_CELLS:
                    self._mask_cache.clear()
                    self._mask_cache_cells = 0
                self._mask_cache[key] = mask
                self._mask_cache_cells += cells
        return mask

    def _prefix_for(self, compiled, embeddings, mask, positions, segments):
        """The cached schema prefix for this request, computing it if needed.

        Returns None when caching is off, which is the default and is what
        training uses. **A prefix is only valid for the weights that produced
        it**, and training mutates the weights on every step, so a cache left
        on during training would serve a schema block from an earlier epoch
        into a later one and the gradient would be silently wrong.
        """
        if not self.cache_prefixes or self.model.training:
            return None
        schema_hash = compiled.schema.schema_hash
        tokens = sum(
            segment.tokens
            for segment in compiled.segments
            if segment.kind
            in {SegmentKind.SCHEMA_QUESTION, SegmentKind.SCHEMA_OPTION, SegmentKind.SCHEMA_LEVEL}
        )
        cached = self._prefix_cache.get(schema_hash)
        if cached is not None and cached.tokens == tokens:
            # Only a *hit* counts as cached. The request that fills the cache
            # paid full price for the schema block, and reporting otherwise
            # would overstate the saving by exactly one request per schema --
            # which on a gateway serving a handful of schemas is most of them.
            self._last_cached_tokens = tokens
            return cached
        prefix = self.model.encode_prefix(
            embeddings, mask, positions, segments, tokens, schema_hash
        )
        if len(self._prefix_cache) >= 64:
            # Bounded, and cleared rather than evicted one at a time: a schema
            # prefix is tens of kilobytes per layer and an unbounded cache on a
            # gateway serving many schemas is a slow memory leak.
            self._prefix_cache.clear()
        self._prefix_cache[schema_hash] = prefix
        return prefix

    def _heads(
        self,
        compiled: CompiledRequest,
        request: SystemOneRequest,
        hidden: torch.Tensor,
        spans: _Spans,
    ) -> dict[str, torch.Tensor]:
        """Per-question logits from one request's hidden states.

        Shared by the single and batched paths so they cannot drift: a trainer
        carrying its own copy of the heads is how a model ends up scoring well
        offline and miscalibrated in production.
        """
        out: dict[str, torch.Tensor] = {}
        for compiled_q in compiled.schema.questions:
            qid = compiled_q.question_id
            readouts = hidden[spans.readout[qid]]
            if compiled_q.kind == "score" and len(readouts) == len(compiled_q.labels) > 1:
                # A slot per level, read out exactly as a Choice with a slot
                # per option is -- the one head in this model that learns a
                # multi-way question. `choice_head` rather than a new one: the
                # arithmetic is identical and a second copy would drift.
                out[qid] = self.model.choice_head(readouts).squeeze(-1)
            elif compiled_q.kind == "score" and self.config.score_head == "linear":
                # One slot, every level read off it at once. The dot-product
                # alternative scores that slot against the pooled encoder
                # states of the level names -- and those states are
                # state-independent by construction, because the schema half of
                # the sequence encodes identically regardless of state (that is
                # the cacheability claim `tests/test_independence.py` asserts).
                # So the dot-product head is already a linear readout of one
                # vector through four fixed directions; this is the same thing
                # without the indirection that lets those directions collapse
                # into each other.
                levels = len(compiled_q.labels)
                if levels > self.config.max_levels:
                    raise ValueError(
                        f"question {qid!r} has {levels} levels and this build's Score "
                        f"head was sized for {self.config.max_levels}; it was trained "
                        "before the head was widened to the contract's cap, so retrain "
                        "or serve it with --score-head dotproduct"
                    )
                out[qid] = self.model.score_head(readouts[:1]).squeeze(0)[:levels]
            elif compiled_q.kind == "noul":
                out[qid] = self.model.noul_head(readouts[0]).reshape(1)
            elif (
                compiled_q.kind == "choice"
                and compiled_q.option_scoring is OptionScoring.READOUT_PER_OPTION
            ):
                out[qid] = self.model.choice_head(readouts).squeeze(-1)
            else:
                members = torch.stack([hidden[idx].mean(dim=0) for idx in spans.members[qid]])
                # The residual was designed and measured for the dot-product
                # *option* head, where the failure was that an option's
                # identity did not survive the encoder. A Score's members are
                # ordered levels and share this code path by accident of
                # implementation rather than because the mechanism applies, so
                # it is switched separately.
                #
                # The one measurement of it on Score regressed a run, and that
                # measurement read the *pooled* metric on a single seed -- it
                # could not see what the Score question itself did. The symptom
                # on the Score question is the collapse signature exactly:
                # `size` emits a near-constant answer, sd 0.019 across the
                # whole seat range, and sits on its marginal on every seed of
                # every configuration tried. That is what a head whose keys
                # have converged looks like, not what an unreadable input looks
                # like. `--score-residual` is how that gets measured per
                # question instead of pooled.
                applies = compiled_q.kind == "choice" or (
                    compiled_q.kind == "score" and self.config.match_residual_score
                )
                if self.config.match_residual and applies:
                    # Carry the option's own input embedding past the encoder,
                    # so which option this is survives layer norm rather than
                    # having to be rediscovered from a smoothed hidden state.
                    rows = spans.member_seed_rows.get(qid)
                    if rows:
                        members = members + torch.stack(rows)
                query = self.model.match_query(readouts[0])
                keys = self.model.match_key(members)
                if self.config.match_normalize:
                    # Cosine, with a learnable temperature. Raw dot products
                    # let descent equalise the logits by shrinking every key
                    # towards their shared mean -- the cheapest route to the
                    # marginal, and it destroys the option signal on the way.
                    query = query / query.norm().clamp_min(1e-6)
                    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    out[qid] = (keys @ query) * self.model.match_log_scale.exp()
                else:
                    out[qid] = keys @ query / math.sqrt(self.config.d_model)
        return out

    def logits_batch(
        self, items: Sequence[tuple[CompiledRequest, SystemOneRequest]]
    ) -> list[dict[str, torch.Tensor]]:
        """One forward pass over several requests, padded to the longest.

        Same arithmetic as calling :meth:`logits` on each, to floating-point
        equality -- asserted in ``tests/test_training.py``, because a trainer
        whose batched pass differs from the served one is the standard way to
        end up with a model that scores well offline and is miscalibrated in
        production.

        Padding needs care in both directions. A padded row must not be
        attended to by a real one, or the answer depends on what happened to
        share its batch; and a padded row must still attend to *something*,
        because a row that may attend to nothing softmaxes over all -inf and
        returns NaN, which then propagates through the whole batch.
        """
        hidden, spans_list = self._forward_batch(items)
        return [
            self._heads(compiled, request, hidden[i], spans_list[i])
            for i, (compiled, request) in enumerate(items)
        ]

    def evidence_logits_batch(
        self, items: Sequence[tuple[CompiledRequest, SystemOneRequest]]
    ) -> list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        """:meth:`logits_batch`, plus each question's span-head logits.

        The trainer's path for cases that carry human rationales: the answer
        and the rationale loss come from one forward pass, so supervising the
        evidence head costs a bilinear product per question and nothing else.
        """
        hidden, spans_list = self._forward_batch(items)
        out = []
        for i, (compiled, request) in enumerate(items):
            spans = spans_list[i]
            out.append(
                (
                    self._heads(compiled, request, hidden[i], spans),
                    {
                        q.question_id: self.evidence_logits(hidden[i], spans, q.question_id)
                        for q in compiled.schema.questions
                    },
                )
            )
        return out

    def _forward_batch(self, items):
        """Padded hidden states for several requests, and where each one's tokens are."""
        if not items:
            return torch.zeros(0), []

        embeddings, spans_list, masks = [], [], []
        for compiled, _ in items:
            embedded, spans = self._embed(compiled)
            embeddings.append(embedded[0])
            spans_list.append(spans)
            # The vectorized mask, not `materialize_mask`. The two are asserted
            # identical in `tests/test_independence.py`; this path was left on
            # the Python one when `logits` moved, and on HelpSteer2 it cost
            # 352 ms a request against 10 ms -- the training loop's whole
            # mask budget, on the one path the vectorization was written for.
            masks.append(self._cached_mask(compiled))

        device = self.device
        width = max(e.shape[0] for e in embeddings)
        batch = len(items)
        padded = torch.zeros(
            batch, width, self.config.d_model, dtype=embeddings[0].dtype, device=device
        )
        mask = torch.zeros(batch, width, width, dtype=torch.bool, device=device)
        positions = torch.zeros(batch, width, dtype=torch.long, device=device)
        segments = torch.zeros(batch, width, dtype=torch.long, device=device)

        for i, (embedded, spans, sample_mask) in enumerate(
            zip(embeddings, spans_list, masks, strict=True)
        ):
            length = embedded.shape[0]
            if sample_mask.shape[0] != length:
                raise ValueError(
                    f"mask is {sample_mask.shape[0]} tokens but the sequence is {length}; "
                    "the compiler's estimator must be the backend's tokenizer"
                )
            padded[i, :length] = embedded
            mask[i, :length, :length] = sample_mask
            positions[i, :length] = spans.positions(device)
            segments[i, :length] = spans.segment_types(device)
            # Padded rows attend to themselves and nothing else. They are
            # discarded below; this only keeps the softmax finite.
            pad = torch.arange(length, width, device=device)
            mask[i, pad, pad] = True

        return self.model(padded, mask, positions, segments), spans_list

    def infer(self, compiled: CompiledRequest, request: SystemOneRequest) -> BackendOutput:
        started = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        evidence: dict[str, tuple[tuple[int, int, float], ...]] = {}
        method = None
        try:
            if request.options.include_evidence:
                # Attribution needs gradients, so its pass is built with
                # autograd on; the span head does not, and stays on the
                # no-grad path the plain answer takes. Either way it is one
                # forward pass: the answer and its evidence share it.
                attributing = self._resolved_evidence_mode() == "gradient_x_input"
                with torch.enable_grad() if attributing else torch.no_grad():
                    embeddings, spans = self._embed(compiled)
                    if attributing:
                        embeddings = embeddings.detach().requires_grad_(True)
                    hidden = self._encode(compiled, embeddings, spans)
                    raw = self._heads(compiled, request, hidden, spans)
                    evidence, method = self._evidence(compiled, embeddings, hidden, spans, raw)
                length = int(embeddings.shape[1])
            else:
                with torch.no_grad():
                    raw, length = self.logits(compiled, request)
        finally:
            self.model.train(was_training)

        outputs = {
            q.question_id: QuestionOutput(
                question_id=q.question_id,
                kind=q.kind,
                logits=tuple(float(x) for x in raw[q.question_id].detach().tolist()),
                evidence=evidence.get(q.question_id),
                evidence_method=method,
            )
            for q in compiled.schema.questions
        }
        return BackendOutput(
            outputs=outputs,
            model_version=self._version,
            model_ms=(time.perf_counter() - started) * 1000.0,
            cached_schema_tokens=self._last_cached_tokens,
            diagnostics={"sequence_tokens": length},
        )

    def infer_many(
        self, batch: Sequence[tuple[CompiledRequest, SystemOneRequest]]
    ) -> list[BackendOutput]:
        """Several requests, one forward pass.

        `docs/next.md` B.2. Continuous batching over a prefill-only model is
        the easy case and this is why: there is no decode loop, no ragged
        generation and no per-step scheduling — every request is exactly one
        pass, so a batch is a pad and a stack.

        **The answers must not depend on what else was in the batch.** That is
        the same guarantee the block mask already gives *within* a request,
        extended across them: padding attends to nothing and nothing attends
        to padding, so sample `i`'s hidden states are a function of sample `i`
        alone. `tests/test_batching.py` asserts it against the one-at-a-time
        path, and it is the only property here worth testing, because a batcher
        that quietly mixes two callers' states produces well-formed answers to
        questions nobody asked.

        Returns one `BackendOutput` per input, in order. Falls back to the
        single path for a batch of one rather than paying to pad it.
        """
        if not batch:
            return []
        if len(batch) == 1:
            return [self.infer(*batch[0])]
        if any(request.options.include_evidence for _, request in batch):
            # Attribution needs a backward pass per question through its own
            # request's graph; a padded batch would share one graph between
            # callers. Evidence requests are answered one at a time, which is
            # the same arithmetic `test_batching.py` holds the batch to.
            return [self.infer(compiled, request) for compiled, request in batch]

        started = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        try:
            embedded = [self._embed(compiled) for compiled, _ in batch]
            lengths = [int(e.shape[1]) for e, _ in embedded]
            width = max(lengths)
            d_model = self.config.d_model
            device = self.device

            padded = torch.zeros(len(batch), width, d_model, device=device)
            positions = torch.zeros(len(batch), width, dtype=torch.long, device=device)
            segments = torch.zeros(len(batch), width, dtype=torch.long, device=device)
            masks = torch.zeros(len(batch), width, width, dtype=torch.bool, device=device)

            for i, ((compiled, _), (embeddings, spans)) in enumerate(
                zip(batch, embedded, strict=True)
            ):
                n = lengths[i]
                padded[i, :n] = embeddings[0]
                positions[i, :n] = spans.positions(device)
                segments[i, :n] = spans.segment_types(device)
                one = self._mask_tensor(compiled)
                if one.shape[0] != n:
                    raise ValueError(
                        f"mask is {one.shape[0]} tokens but the sequence is {n}; "
                        "the compiler's estimator must be the backend's tokenizer"
                    )
                masks[i, :n, :n] = one
                # A padding row that may attend to nothing is a softmax over an
                # empty set, which is NaN, and one NaN in a batched attention
                # poisons every sample sharing the tensor. Letting padding
                # attend to itself keeps it finite and keeps it isolated: no
                # real token attends *to* padding, so the value is never read.
                pad = torch.arange(n, width, device=device)
                masks[i, pad, pad] = True

            with torch.no_grad():
                hidden = self.model(padded, masks, positions, segments)

            results = []
            for i, ((compiled, request), (_, spans)) in enumerate(
                zip(batch, embedded, strict=True)
            ):
                raw = self._heads(compiled, request, hidden[i, : lengths[i]], spans)
                results.append(
                    BackendOutput(
                        outputs={
                            q.question_id: QuestionOutput(
                                question_id=q.question_id,
                                kind=q.kind,
                                logits=tuple(float(x) for x in raw[q.question_id].tolist()),
                            )
                            for q in compiled.schema.questions
                        },
                        model_version=self._version,
                        # The pass is shared, so its cost is too. Charging each
                        # request the whole batch's wall clock would make a
                        # batch of eight look eight times more expensive than
                        # the same work unbatched.
                        model_ms=(time.perf_counter() - started) * 1000.0 / len(batch),
                        # The schema prefix cache is not used on this path: a
                        # batch's samples can have different schemas and the
                        # prefix is per-schema, so mixing them would need a
                        # per-sample gather this does not do yet.
                        cached_schema_tokens=0,
                        diagnostics={"sequence_tokens": lengths[i], "batch": len(batch)},
                    )
                )
            return results
        finally:
            self.model.train(was_training)

    def _mask_tensor(self, compiled: CompiledRequest) -> torch.Tensor:
        """The attention mask, built with tensor ops rather than Python loops.

        `trigon.schema.compiler.materialize_mask` is the specification and
        stays pure Python, because the compiler is imported by the gateway and
        the drift tests without torch. It is also O(n^2) in the interpreter,
        and it was costing **208 ms per request** on HelpSteer2 -- over half
        the total forward time -- against 399 ms for everything.

        Banking77 hid that completely. Its requests are 77 options and a short
        state, so they land on a handful of distinct lengths and the shape
        cache hits almost every time. HelpSteer2's state is an LLM response:
        every request is a different length, every request misses, and every
        miss rebuilt a 2,246 x 2,246 list of Python bools. Four training seeds
        were on course for 45 hours.

        This builds the same mask by indexing a (groups x groups) table with
        the per-token group ids, which is the same computation the double loop
        does one cell at a time. `tests/test_independence.py` asserts the two
        agree exactly -- they must, because the Python one is what the
        isolation tests read.
        """
        owners: list[str] = []
        for segment in compiled.segments:
            owners.extend([segment.group] * segment.tokens)
        plan = compiled.attention
        groups = sorted(set(owners))
        index = {group: i for i, group in enumerate(groups)}
        allowed = torch.tensor(
            [[plan.can_attend(a, b) for b in groups] for a in groups], dtype=torch.bool
        )
        ids = torch.tensor([index[o] for o in owners], dtype=torch.long)
        mask = allowed[ids[:, None], ids[None, :]]
        if not plan.bidirectional:
            # The causal fallback applies within a group when the model was
            # not converted to prefix-LM attention.
            mask = mask & torch.ones_like(mask).tril()
        # Built on the CPU from Python lists, then moved once.
        return mask.to(self.device)

    # -- sequence construction -------------------------------------------

    def _embed(self, compiled: CompiledRequest) -> tuple[torch.Tensor, _Spans]:
        """Build the input embeddings and record where everything landed."""
        rows: list[torch.Tensor] = []
        spans = _Spans()
        cursor = 0
        table = self.model.embed
        member_embeddings: dict[tuple[str, int], torch.Tensor] = {}
        # Position counters restart per isolation group -- see the module
        # docstring for why this is load-bearing rather than cosmetic.
        group_position: dict[str, int] = {}

        for segment in compiled.segments:
            group = segment.group
            start = group_position.get(group, 0)
            if segment.kind is SegmentKind.READOUT:
                assert segment.question_id is not None
                # A pretrained backbone's vocabulary has no readout token -- id 1
                # is an ordinary word there -- so a model may carry its own
                # learned readout vector instead.
                own = getattr(self.model, "readout", None)
                slot = own if own is not None else table.weight[READOUT_ID]
                if segment.member_index is not None:
                    # Seed a per-option slot with that option's own content, so
                    # the slot carries which option it is answering for.
                    seed = member_embeddings.get((segment.question_id, segment.member_index))
                    if seed is not None:
                        slot = slot + seed
                rows.append(slot.unsqueeze(0))
                spans.readout.setdefault(segment.question_id, []).append(cursor)
                spans.record(start, _SEGMENT_TYPE[segment.kind])
                group_position[group] = start + 1
                cursor += 1
                continue

            ids = self.tokenizer.encode(segment.text)
            if len(ids) != segment.tokens:
                raise ValueError(
                    f"segment {segment.kind.value} compiled to {segment.tokens} tokens "
                    f"but tokenizes to {len(ids)}"
                )
            if segment.kind is SegmentKind.STATE:
                spans.state.extend(range(cursor, cursor + len(ids)))
            for offset in range(len(ids)):
                spans.record(start + offset, _SEGMENT_TYPE[segment.kind])
            group_position[group] = start + len(ids)
            if ids:
                vectors = table(torch.tensor(ids, dtype=torch.long, device=table.weight.device))
                rows.append(vectors)
                if segment.member_index is not None and segment.question_id is not None:
                    pooled = vectors.mean(dim=0)
                    member_embeddings[(segment.question_id, segment.member_index)] = pooled.detach()
                    spans.member_seed_rows.setdefault(segment.question_id, []).append(pooled)
                    spans.members.setdefault(segment.question_id, []).append(
                        list(range(cursor, cursor + len(ids)))
                    )
            cursor += len(ids)

        if not rows:
            raise ValueError("compiled request produced an empty sequence")
        return torch.cat(rows, dim=0).unsqueeze(0), spans


_SEGMENT_TYPE = {
    SegmentKind.SCHEMA_QUESTION: 0,
    SegmentKind.SCHEMA_OPTION: 0,
    SegmentKind.SCHEMA_LEVEL: 0,
    SegmentKind.STATE: 1,
    SegmentKind.READOUT: 2,
}


class _Spans:
    """Where everything landed, so the heads can find their states."""

    def __init__(self) -> None:
        self.readout: dict[str, list[int]] = {}
        # Sequence positions of the state's tokens, in order: what evidence scores.
        self.state: list[int] = []
        self.members: dict[str, list[list[int]]] = {}
        # Per-option input embeddings, kept differentiable for match_residual.
        self.member_seed_rows: dict[str, list] = {}
        self._positions: list[int] = []
        self._types: list[int] = []

    def record(self, position: int, segment_type: int) -> None:
        self._positions.append(position)
        self._types.append(segment_type)

    def positions(self, device=None) -> torch.Tensor:
        return torch.tensor(self._positions, dtype=torch.long, device=device)

    def segment_types(self, device=None) -> torch.Tensor:
        return torch.tensor(self._types, dtype=torch.long, device=device)
