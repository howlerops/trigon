"""The LM-scoring backend's prompt and its content-free correction, without a model."""

from __future__ import annotations

import pytest

from trigon.backends.lm_score import CONTENT_FREE_STATE, LMScoreBackend, prompt_for
from trigon.types import ChoiceQuestion, NoulQuestion


def _choice() -> ChoiceQuestion:
    return ChoiceQuestion(
        instructions="Which team should take this ticket?",
        options=[
            {"name": "billing", "criteria": "charges and refunds"},
            {"name": "login", "criteria": "login"},
        ],
    )


def test_choice_prompt_lists_options_and_candidates_in_declared_order():
    prompt, candidates = prompt_for("I was charged twice.", _choice())
    assert candidates == [" billing", " login"]
    assert "- billing: charges and refunds" in prompt
    # A criterion that only repeats the name adds nothing.
    assert "- login\n" in prompt
    assert prompt.endswith("Answer:")


def test_noul_candidates_are_no_then_yes():
    question = NoulQuestion(instructions="Is it urgent?")
    _, candidates = prompt_for("Server is down.", question)
    assert candidates == [" no", " yes"]


def _fake(content_free: bool) -> LMScoreBackend:
    backend = LMScoreBackend.__new__(LMScoreBackend)
    backend.content_free = content_free
    backend._prior = {}
    backend.calls = []

    def logprobs(prompt, candidates):
        backend.calls.append(prompt)
        bias = 1.0 if prompt.startswith(CONTENT_FREE_STATE) else 0.0
        return [-1.0 - bias * i for i in range(len(candidates))]

    backend._logprobs = logprobs
    return backend


def test_content_free_prior_is_computed_once_per_question():
    backend = _fake(content_free=True)
    question = _choice()
    _, candidates = prompt_for("x", question)
    first = backend._content_free(question, candidates)
    second = backend._content_free(question, candidates)
    assert first == second == [-1.0, -2.0]
    assert sum(p.startswith(CONTENT_FREE_STATE) for p in backend.calls) == 1


def test_a_bundle_names_its_own_backend(tmp_path):
    """`TRIGON_BACKEND=torch` with an LM-score adapter serves it as one."""
    torch = __import__("pytest").importorskip("torch")
    from trigon.server.app import _is_lm_score_adapter

    lm = tmp_path / "lm.pt"
    torch.save({"kind": "lm-score", "base": "x", "lora": {}}, lm)
    readout = tmp_path / "readout.pt"
    torch.save({"config": {"d_model": 8}, "state": {"w": torch.zeros(2)}}, readout)
    assert _is_lm_score_adapter(str(lm))
    assert not _is_lm_score_adapter(str(readout))
    assert not _is_lm_score_adapter("Qwen/Qwen3-0.6B")


def test_shared_prefix_leaves_every_question_a_token():
    from trigon.backends.lm_score import shared_prefix

    assert shared_prefix([[1, 2, 3, 4], [1, 2, 3, 9], [1, 2, 7]]) == 2
    assert shared_prefix([[1, 2, 3], [1, 2, 3]]) == 2  # identical: one token each to score
    assert shared_prefix([[5], [5, 6]]) == 0


def test_the_readout_keeps_a_weight_per_kind_and_loads_an_old_scalar():
    torch = pytest.importorskip("torch")
    from trigon.backends.lm_score import KIND_CHOICE, KIND_NOUL, Readout

    readout = Readout(hidden=4, device="cpu")
    with torch.no_grad():
        readout.w.copy_(torch.tensor([1.0, 0.0]))
    sums, hidden = torch.tensor([-1.0, -3.0]), torch.zeros(2, 4)
    assert readout(sums, hidden, KIND_CHOICE).tolist() == [-1.0, -3.0]
    assert readout(sums, hidden, KIND_NOUL).tolist() == [0.0, 0.0]
    readout.load_state_dict({"w": torch.tensor(0.4), "residual": readout.residual.state_dict()})
    assert readout.w.tolist() == pytest.approx([0.4, 0.4])
