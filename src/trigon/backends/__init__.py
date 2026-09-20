"""Sources of logits. The torch backend is imported lazily -- the gateway and
the eval harness must stay importable without a GPU stack installed."""

from .base import Backend, BackendOutput, QuestionOutput, validate_output
from .lexical import LexicalBackend

__all__ = [
    "Backend",
    "BackendOutput",
    "LexicalBackend",
    "QuestionOutput",
    "validate_output",
]


def __getattr__(name: str):
    if name in {"TorchReadoutBackend", "PrefillOnlyModel", "ReadoutConfig"}:
        from . import torch_readout

        return getattr(torch_readout, name)
    if name == "LLMBaselineBackend":
        from . import llm

        return getattr(llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
