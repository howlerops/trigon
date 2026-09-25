"""Outcome-grounded training for the reference readout model.

Imported lazily everywhere else: this is the only part of the package that
needs torch.
"""

from .losses import OrdinalConfig, question_loss, rationale_loss, squared_emd
from .trainer import TrainingConfig, TrainingReport, train

__all__ = [
    "OrdinalConfig",
    "TrainingConfig",
    "TrainingReport",
    "question_loss",
    "rationale_loss",
    "squared_emd",
    "train",
]
