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
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

try:  # pragma: no cover - exercised by the import error path only
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "the torch readout backend needs the 'train' extra: pip install 'trigon[train]'"
    ) from exc

from ..schema import (
    CompiledRequest,
    OptionScoring,
    SchemaCompiler,
    SegmentKind,
    mask_shape_key,
    materialize_mask,
)
from ..schema.tokens import CallableEstimator
from ..types import SystemOneRequest
from .base import BackendOutput, QuestionOutput
from .tokenizer import READOUT_ID, Tokenizer, build_tokenizer, default_tokenizer, describe

__all__ = ["PrefillOnlyModel", "ReadoutConfig", "TorchReadoutBackend"]

TRAINED_VERSION_PREFIX = "trigon-reference-0.1.0"
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
        max_levels: int = 32,
        match_normalize: bool = False,
        match_residual: bool = True,
    ) -> None:
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must divide by n_heads {n_heads}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_levels = max_levels
        # ``match_residual`` defaults ON: without it the dot-product head
        # collapses to the marginal and answers 0.254 against a 0.253 marginal
        # predictor; with it, 0.847. ``match_normalize`` defaults OFF: it does
        # nothing alone and cancels most of the residual's gain. Both measured
        # once, at spike scale -- see docs/decisions.md.
        self.match_normalize = match_normalize
        self.match_residual = match_residual


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

    def forward(
        self,
        token_embeddings: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        segment_types: torch.Tensor,
    ) -> torch.Tensor:
        """One forward pass.

        ``token_embeddings`` (1, T, d); ``mask`` (T, T) boolean "may attend";
        ``positions`` (T,) group-local indices; ``segment_types`` (T,) in
        {0: schema, 1: state, 2: readout}.
        """
        table = _sinusoidal(
            int(positions.max().item()) + 1,
            self.config.d_model,
            token_embeddings.device,
            token_embeddings.dtype,
        )
        hidden = (
            token_embeddings
            + table[positions].unsqueeze(0)
            + self.segment_embed(segment_types).unsqueeze(0)
        )
        # nn.TransformerEncoder takes True as "block this pair".
        hidden = self.encoder(hidden, mask=~mask)
        return self.norm(hidden)


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
        self._mask_cache: dict[object, torch.Tensor] = {}

    @property
    def model_version(self) -> str:
        return self._version

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
        key = mask_shape_key(compiled)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.tensor(materialize_mask(compiled), dtype=torch.bool)
            if len(self._mask_cache) >= 256:
                self._mask_cache.clear()
            self._mask_cache[key] = mask
        if mask.shape[0] != embeddings.shape[1]:
            raise ValueError(
                f"mask is {mask.shape[0]} tokens but the sequence is "
                f"{embeddings.shape[1]}; the compiler's estimator must be the "
                f"backend's tokenizer"
            )
        hidden = self.model(embeddings, mask, spans.positions(), spans.segment_types())[0]

        out: dict[str, torch.Tensor] = {}
        for compiled_q in compiled.schema.questions:
            qid = compiled_q.question_id
            readouts = hidden[spans.readout[qid]]
            if compiled_q.kind == "noul":
                out[qid] = self.model.noul_head(readouts[0]).reshape(1)
            elif (
                compiled_q.kind == "choice"
                and compiled_q.option_scoring is OptionScoring.READOUT_PER_OPTION
            ):
                out[qid] = self.model.choice_head(readouts).squeeze(-1)
            else:
                members = torch.stack([hidden[idx].mean(dim=0) for idx in spans.members[qid]])
                if self.config.match_residual:
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
                    # towards their shared mean -- which is the cheapest route
                    # to the marginal and destroys the option signal on the
                    # way. Normalising removes that route: scale buys nothing,
                    # so the only way to flatten the answer is to make the
                    # option directions identical, which the data does not
                    # reward.
                    query = query / query.norm().clamp_min(1e-6)
                    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    out[qid] = (keys @ query) * self.model.match_log_scale.exp()
                else:
                    out[qid] = keys @ query / math.sqrt(self.config.d_model)
        return out, int(embeddings.shape[1])

    @torch.no_grad()
    def infer(self, compiled: CompiledRequest, request: SystemOneRequest) -> BackendOutput:
        started = time.perf_counter()
        was_training = self.model.training
        self.model.eval()
        try:
            raw, length = self.logits(compiled, request)
        finally:
            self.model.train(was_training)

        outputs = {
            q.question_id: QuestionOutput(
                question_id=q.question_id,
                kind=q.kind,
                logits=tuple(float(x) for x in raw[q.question_id].tolist()),
            )
            for q in compiled.schema.questions
        }
        return BackendOutput(
            outputs=outputs,
            model_version=self._version,
            model_ms=(time.perf_counter() - started) * 1000.0,
            diagnostics={"sequence_tokens": length},
        )

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
                slot = table.weight[READOUT_ID]
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
            for offset in range(len(ids)):
                spans.record(start + offset, _SEGMENT_TYPE[segment.kind])
            group_position[group] = start + len(ids)
            if ids:
                vectors = table(torch.tensor(ids, dtype=torch.long))
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
        self.members: dict[str, list[list[int]]] = {}
        # Per-option input embeddings, kept differentiable for match_residual.
        self.member_seed_rows: dict[str, list] = {}
        self._positions: list[int] = []
        self._types: list[int] = []

    def record(self, position: int, segment_type: int) -> None:
        self._positions.append(position)
        self._types.append(segment_type)

    def positions(self) -> torch.Tensor:
        return torch.tensor(self._positions, dtype=torch.long)

    def segment_types(self) -> torch.Tensor:
        return torch.tensor(self._types, dtype=torch.long)
