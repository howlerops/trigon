"""Score each declared answer by the backbone's own language-model probability.

The readout heads in `torch_readout` and `qwen_readout` learn every answer from
scratch, through a bidirectional mask the pretrained model never saw, and
discard the backbone's language-model head. Public evidence says that is the
wrong trade: readouts of the form ``w * lm_logprob + residual`` with the
residual initialised at zero keep ``w`` near 1.0 after training -- the model
wants its pretrained token probabilities -- and an open 4B model built that way
matches the incumbent's macro accuracy on a 22-task benchmark. The incumbent's
own behaviour points the same way: options counted as output tokens, a
first-option position bias, a state-plus-longest-question budget.

This backend is the zero-shot half of that design: no trained parameters at
all. Each question becomes one causal prompt -- state, question, its options --
and each answer's logit is the log-probability of its text as the continuation.
Questions are separate sequences, so one question's answer cannot depend on
another's: the independence the block mask was built to guarantee holds by
construction. The prompt is run once and its cache reused for every candidate.

A raw LM prior leans: on a bare yes/no prompt a small model says "yes" more
often than the passage warrants, and in a list it favours the first option.
`content_free=True` subtracts each candidate's score under the same question
with the state replaced by "N/A" (contextual calibration, Zhao et al. 2021).
That score depends only on the question, so it is computed once per question
shape and cached -- the same class as the schema prefix.

`transformers` is an optional dependency, imported here only.
"""

from __future__ import annotations

import time

from ..schema import CompiledRequest, render_state
from ..types import ChoiceQuestion, DecisionRequest, NoulQuestion, ScoreQuestion
from .base import BackendOutput, QuestionOutput

TEMPLATE_VERSION = "lm-score-v1"
#: Below this many shared tokens a second forward pass costs more than it saves.
MIN_SHARED_PREFIX = 32
CONTENT_FREE_STATE = "N/A"


def _members(question) -> list[tuple[str, str | None]]:
    if isinstance(question, ChoiceQuestion):
        return [(o.name, o.criteria) for o in question.options]
    if isinstance(question, ScoreQuestion):
        return [(lv.name, lv.criteria) for lv in question.levels]
    return []


def prompt_for(state: str, question) -> tuple[str, list[str]]:
    """The prompt and the candidate continuations, in the question's label order.

    A Noul's candidates are ``no, yes`` -- trigon's aligned order -- and the
    backend turns their two log-probabilities into one log-odds logit.
    """
    lines = [state.strip(), "", f"Question: {question.instructions.strip()}"]
    if isinstance(question, NoulQuestion):
        lines.append("Answer yes or no.")
        lines.append("Answer:")
        return "\n".join(lines), [" no", " yes"]
    members = _members(question)
    lines.append("Options:")
    for name, criteria in members:
        lines.append(f"- {name}" + (f": {criteria}" if criteria and criteria != name else ""))
    lines.append("Answer with one option.")
    lines.append("Answer:")
    return "\n".join(lines), [f" {name}" for name, _ in members]


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def add_lora(model, rank: int = 16, alpha: float = 32.0, targets=LORA_TARGETS):
    """Freeze ``model`` and wrap each target linear with a rank-``rank`` update.

    ``B`` starts at zero, so the wrapped model is the pretrained one until it
    trains. Returns the LoRA parameters, the only ones that will.
    """
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.lora_a = nn.Parameter(
                torch.randn(rank, base.in_features, dtype=torch.float32) / base.in_features**0.5
            )
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

        def forward(self, x):
            if self.lora_a.dtype == x.dtype:
                # Serving in the model's own precision: no casts around it.
                return self.base(x) + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale
            update = (x.float() @ self.lora_a.T) @ self.lora_b.T
            return self.base(x) + (update * self.scale).to(x.dtype)

    for p in model.parameters():
        p.requires_grad_(False)
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child))
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def merge_lora(model) -> int:
    """Fold each LoRA update into its frozen weight and drop the wrapper: serving only.

    ``W + scale * B @ A`` is computed in float32 and rounded once to the
    weight's dtype. On a GPU that is bfloat16, so the merged model is not the
    unmerged one to the last bit; `scripts/serving_parity.py` is what says the
    difference is rounding. The wrapper's float32 matmuls and casts were ~30%
    of an agent step's GPU time on an L4.
    """
    import torch

    merged = 0
    with torch.no_grad():
        for module in list(model.modules()):
            for name, child in list(module.named_children()):
                if hasattr(child, "lora_a") and hasattr(child, "base"):
                    base = child.base
                    update = child.scale * (child.lora_b.float() @ child.lora_a.float())
                    base.weight.copy_(
                        (base.weight.float() + update.to(base.weight.device)).to(base.weight.dtype)
                    )
                    setattr(module, name, base)
                    merged += 1
    return merged


KIND_CHOICE, KIND_NOUL = 0, 1


class Readout:
    """``score = w * lm_logprob + residual(hidden)``, residual zero at init.

    At initialisation this is exactly the zero-shot scorer: training starts
    from the backbone's own answer and moves away only as far as the data
    asks. ``w`` is reported because where it settles says how much of the
    pretrained probability the trained model kept.

    One ``w`` per kind of question -- ``KIND_CHOICE`` (Choice and Score) and
    ``KIND_NOUL`` -- not one for all: with a single shared ``w``, a corpus
    whose yes/no questions the backbone reads backwards (agent transcripts)
    drove it from 0.40 to 0.01 and took the multiple-choice prior with it,
    CLINC150's ECE rising from 0.105 to 0.181-0.258. A scalar ``w`` in an
    older adapter loads into both.
    """

    def __init__(self, hidden: int, device):
        import torch

        self.w = torch.ones(2, device=device, requires_grad=True)
        self.residual = torch.nn.Linear(hidden, 1).to(device)
        torch.nn.init.zeros_(self.residual.weight)
        torch.nn.init.zeros_(self.residual.bias)

    def parameters(self):
        return [self.w, *self.residual.parameters()]

    def __call__(self, sums, hidden, kind: int = 0):
        return self.w[kind] * sums + self.residual(hidden.float())[:, 0]

    def state_dict(self):
        return {"w": self.w.detach().cpu(), "residual": self.residual.state_dict()}

    def load_state_dict(self, state):
        import torch

        with torch.no_grad():
            self.w.copy_(torch.as_tensor(state["w"], dtype=self.w.dtype).expand_as(self.w))
        self.residual.load_state_dict(state["residual"])


def shared_prefix(sequences: list[list[int]]) -> int:
    """The longest common prefix of every sequence, leaving each at least one token."""
    n = min(len(s) for s in sequences) - 1
    first = sequences[0]
    for i in range(max(n, 0)):
        if any(s[i] != first[i] for s in sequences[1:]):
            return i
    return max(n, 0)


def pack(prompt_ids: list[int], candidates: list[list[int]], offset: int = 0):
    """One sequence holding the prompt and every candidate after it.

    Each candidate attends to the whole prompt and to its own earlier tokens,
    never to another candidate, and its positions continue from the prompt as
    if it were the only continuation -- so one forward pass returns what one
    pass per candidate would, and the options cost what they cost the
    incumbent: their own tokens. Returns ``(ids, positions, allowed, pred,
    target, owner, last)``: ``allowed[q, k]`` is the attention mask, and token
    ``target[j]`` of candidate ``owner[j]`` is predicted at position ``pred[j]``;
    ``last[i]`` is candidate ``i``'s final position.

    ``offset`` is the length of a prefix already in the KV cache: positions
    start after it and every query may attend to all of it.
    """
    import torch

    n_prompt = len(prompt_ids)
    total = n_prompt + sum(len(c) for c in candidates)
    ids = list(prompt_ids)
    positions = list(range(offset, offset + n_prompt))
    allowed = torch.zeros((total, total), dtype=torch.bool)
    allowed[:n_prompt, :n_prompt] = torch.tril(torch.ones(n_prompt, n_prompt, dtype=torch.bool))
    pred, target, owner, last = [], [], [], []
    start = n_prompt
    for i, cand in enumerate(candidates):
        end = start + len(cand)
        ids += cand
        positions += range(offset + n_prompt, offset + n_prompt + len(cand))
        allowed[start:end, :n_prompt] = True
        allowed[start:end, start:end] = torch.tril(
            torch.ones(len(cand), len(cand), dtype=torch.bool)
        )
        for t, token in enumerate(cand):
            pred.append(n_prompt - 1 if t == 0 else start + t - 1)
            target.append(token)
            owner.append(i)
        last.append(end - 1)
        start = end
    if offset:
        allowed = torch.cat([torch.ones((total, offset), dtype=torch.bool), allowed], dim=1)
    return (
        torch.tensor(ids),
        torch.tensor(positions),
        allowed,
        torch.tensor(pred),
        torch.tensor(target),
        torch.tensor(owner),
        torch.tensor(last),
    )


def packed_scores(
    model, prompt_ids: list[int], candidates: list[list[int]], device, past=None, offset: int = 0
):
    """Each candidate's summed log-probability, and its last hidden state.

    Differentiable: training calls this with gradients on, serving under
    ``no_grad``. With ``past``, a KV cache holding the first ``offset`` tokens
    of the prompt, ``prompt_ids`` is the rest of it; the cache is cropped back
    to ``offset`` afterwards, so the next question can reuse it.
    """
    import torch

    ids, positions, allowed, pred, target, owner, last = pack(prompt_ids, candidates, offset)
    dtype = next(model.parameters()).dtype
    mask = torch.zeros(allowed.shape, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    # The decoder alone, then the vocabulary head only where a candidate's
    # token is predicted: over every position of an agent's page the head
    # was the largest single matmul in the request.
    out = model.get_decoder()(
        ids[None].to(device),
        position_ids=positions[None].to(device),
        attention_mask=mask[None, None].to(device),
        past_key_values=past,
        use_cache=past is not None,
    )
    if past is not None:
        past.crop(offset - past.get_seq_length())  # drop what this call appended
    hidden = out.last_hidden_state[0]
    logits = model.get_output_embeddings()(hidden[pred.to(device)]).float()
    token_logp = torch.log_softmax(logits, dim=-1).gather(1, target.to(device)[:, None])[:, 0]
    sums = torch.zeros(len(candidates), device=token_logp.device).index_add(
        0, owner.to(device), token_logp
    )
    return sums, hidden[last.to(device)]


class LMScoreBackend:
    """A pretrained causal LM as a typed-decision backend, with no trained head."""

    kind = "lm-score"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-0.6B",
        *,
        revision: str | None = None,
        device: str | None = None,
        max_prompt_tokens: int = 8192,
        content_free: bool = True,
        adapter: str | None = None,
        merge: bool = False,
        lora_in_model_dtype: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype).to(
            self.device
        )
        self.readout: Readout | None = None
        name = model.split("/")[-1].lower()
        if adapter is not None:
            # A trained adapter: LoRA on the frozen backbone and the readout.
            state = torch.load(adapter, map_location="cpu", weights_only=True)
            lora = add_lora(self.model, state["rank"], state["alpha"])
            self.model.load_state_dict(state["lora"], strict=False)
            missing = set(lora) - set(state["lora"])
            if missing:
                raise ValueError(f"{adapter} has no LoRA weights for {sorted(missing)[:3]}")
            if merge:
                merge_lora(self.model)
            elif lora_in_model_dtype:
                # Keeps the update separate from the weight -- no rounding of
                # it into bfloat16 -- but drops the float32 casts around it.
                for name, param in self.model.named_parameters():
                    if "lora_" in name:
                        param.data = param.data.to(dtype)
            self.model.to(self.device)
            self.readout = Readout(self.model.config.hidden_size, self.device)
            self.readout.load_state_dict(state["readout"])
            name = f"{name}-{state['name']}"
            content_free = False  # trained through, the prior is in the weights
        self.model.eval()
        self.max_prompt_tokens = max_prompt_tokens
        self.content_free = content_free
        self._prior: dict[tuple[str, tuple[str, ...]], list[float]] = {}
        suffix = "-cf" if content_free else ""
        self._version = f"trigon-lm-score-{name}+{TEMPLATE_VERSION}{suffix}"

    @property
    def model_version(self) -> str:
        return self._version

    def _encode(self, prompt: str, candidates: list[str]) -> tuple[list[int], list[list[int]]]:
        prompt_ids = self.tokenizer(prompt).input_ids[-self.max_prompt_tokens :]
        encoded = [self.tokenizer(c, add_special_tokens=False).input_ids for c in candidates]
        return prompt_ids, encoded

    def _prefix(self, ids: list[int]):
        """The KV cache of a prompt prefix several questions share."""
        import torch

        with torch.no_grad():
            decoder = self.model.get_decoder()
            out = decoder(torch.tensor([ids], device=self.device), use_cache=True)
        return out.past_key_values

    def _score_ids(
        self, prompt_ids, encoded, past=None, offset: int = 0, kind: int = KIND_CHOICE
    ) -> list[float]:
        import torch

        with torch.no_grad():
            sums, hidden = packed_scores(
                self.model, prompt_ids[offset:], encoded, self.device, past, offset
            )
            scores = sums if self.readout is None else self.readout(sums, hidden, kind)
            return [float(x) for x in scores]

    def _logprobs(self, prompt: str, candidates: list[str]) -> list[float]:
        return self._score_ids(*self._encode(prompt, candidates))

    def _content_free(self, question, candidates: list[str]) -> list[float]:
        empty, _ = prompt_for(CONTENT_FREE_STATE, question)
        key = (empty, tuple(candidates))
        if key not in self._prior:
            if len(self._prior) >= 4096:
                self._prior.clear()
            self._prior[key] = self._logprobs(empty, candidates)
        return self._prior[key]

    def infer(self, compiled: CompiledRequest, request: DecisionRequest) -> BackendOutput:
        started = time.perf_counter()
        state = render_state(request.state)
        outputs: dict[str, QuestionOutput] = {}
        from torch.profiler import record_function

        items = []
        with record_function("lm.encode"):
            for compiled_q in compiled.schema.questions:
                question = request.questions[compiled_q.question_id]
                prompt, candidates = prompt_for(state, question)
                items.append((compiled_q, question, candidates, *self._encode(prompt, candidates)))
        # Every question's prompt opens with the same state, and an agent's
        # state is most of its tokens: run the shared token prefix once and
        # score each question's remainder against its cache. Measured on the
        # token ids, so it is exact whatever the tokenizer does at the seam.
        offset = shared_prefix([ids for *_, ids, _ in items]) if len(items) > 1 else 0
        with record_function("lm.prefix"):
            past = self._prefix(items[0][3][:offset]) if offset >= MIN_SHARED_PREFIX else None
        offset = offset if past is not None else 0
        for compiled_q, question, candidates, prompt_ids, encoded in items:
            with record_function("lm.question"):
                kind = KIND_NOUL if isinstance(question, NoulQuestion) else KIND_CHOICE
                scores = self._score_ids(prompt_ids, encoded, past, offset, kind)
            if self.content_free:
                prior = self._content_free(question, candidates)
                scores = [s - p for s, p in zip(scores, prior, strict=True)]
            if isinstance(question, NoulQuestion):
                logits: tuple[float, ...] = (scores[1] - scores[0],)
            else:
                logits = tuple(scores)
            outputs[compiled_q.question_id] = QuestionOutput(
                question_id=compiled_q.question_id, kind=compiled_q.kind, logits=logits
            )
        return BackendOutput(
            outputs=outputs,
            model_version=self.model_version,
            model_ms=(time.perf_counter() - started) * 1000,
        )
