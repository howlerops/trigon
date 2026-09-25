"""Training objectives. Proper scoring rules first, everything else optional.

The build plan is explicit that distillation and calibration are different
objectives and that conflating them is the mistake to avoid. This module holds
only the calibration half -- outcome-grounded training against real labels --
because that is the half that can be implemented and tested without a teacher
ensemble, and it is the half the product is sold on.

Why the default is cross-entropy everywhere: it is a **strictly proper**
scoring rule. The loss-minimising prediction is the true conditional
distribution, so honest probabilities are the optimum rather than something a
regulariser has to encourage. Every alternative here is measured against that
property, not against accuracy.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["OrdinalConfig", "question_loss", "squared_emd"]


class OrdinalConfig:
    """Settings for the phase-2 ordinal-loss ablation (decision D2).

    ``weight`` of 0 is the shipped default: plain cross-entropy. Raising it
    mixes in squared earth-mover's distance over the level CDF, which respects
    the ordering of Score levels but is **not** a proper scoring rule -- it can
    prefer a distribution smoother than the truth, because smoothing is cheap
    under a transport cost. Adopt it only if the ablation shows ECE improving
    and Brier not regressing.
    """

    def __init__(self, weight: float = 0.0) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"ordinal weight must be in [0, 1], got {weight}")
        self.weight = weight

    @property
    def enabled(self) -> bool:
        return self.weight > 0.0


def squared_emd(logits: torch.Tensor, label: int, distribution=None) -> torch.Tensor:
    """Squared earth-mover's distance between the predicted and true CDFs.

    Ordinal-aware: moving mass one level costs less than moving it four. Used
    only as an auxiliary term, and only behind ``OrdinalConfig``. Against an
    annotator distribution when one is given, a point mass otherwise.
    """
    probs = torch.softmax(logits, dim=-1)
    if distribution is not None:
        target = torch.tensor(distribution, dtype=probs.dtype, device=probs.device)
    else:
        target = torch.zeros_like(probs)
        target[label] = 1.0
    return torch.sum((torch.cumsum(probs, dim=-1) - torch.cumsum(target, dim=-1)) ** 2)


def question_loss(
    logits: torch.Tensor,
    kind: str,
    label: int,
    ordinal: OrdinalConfig | None = None,
    distribution=None,
) -> torch.Tensor:
    """Loss for one question's head.

    Choice and Score use categorical cross-entropy over the declared labels;
    Noul uses binary cross-entropy on its single logit. Both are the log
    scoring rule, so a model that reports honest probabilities minimises them.

    With an annotator ``distribution`` the target is that distribution rather
    than one label: cross-entropy against it is minimised by reporting it, so
    a model trained this way learns how much people disagree -- where the hard
    label teaches it to be as sure as the majority, which is the confident
    wrong answer the product exists to avoid.
    """
    if distribution is not None and kind != "noul":
        target = torch.tensor(distribution, dtype=logits.dtype, device=logits.device)
        loss = -(target * torch.log_softmax(logits, dim=-1)).sum()
        if kind == "score" and ordinal is not None and ordinal.enabled:
            loss = (1.0 - ordinal.weight) * loss + ordinal.weight * squared_emd(
                logits, label, distribution
            )
        return loss
    if kind == "noul":
        # With a distribution, the target is the share of annotators who said
        # yes -- (no, yes) in the aligned order -- for the same reason as above:
        # binary cross-entropy against it is minimised by reporting it.
        yes = float(distribution[-1]) if distribution is not None else float(label)
        target = torch.tensor([yes], dtype=logits.dtype, device=logits.device)
        return nn.functional.binary_cross_entropy_with_logits(logits, target)

    loss = nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([label], device=logits.device)
    )
    if kind == "score" and ordinal is not None and ordinal.enabled:
        loss = (1.0 - ordinal.weight) * loss + ordinal.weight * squared_emd(logits, label)
    return loss
