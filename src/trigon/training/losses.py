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

__all__ = [
    "OrdinalConfig",
    "consistency_loss",
    "question_loss",
    "rationale_loss",
    "squared_emd",
]


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


def rationale_loss(span_logits: torch.Tensor, labels: list[int]) -> torch.Tensor:
    """Binary cross-entropy of the evidence head against a human rationale.

    One Bernoulli per state token -- did the annotators highlight it -- and the
    mean over tokens, so a long state does not outweigh a short one. The log
    scoring rule again, and for the same reason as everywhere else here: the
    loss-minimising score is the true probability that a person highlights the
    token, so the head's 0.5 threshold means what it says. No positive weight
    for the rarer class: that would buy recall by making the probabilities lie.
    """
    target = torch.tensor(labels, dtype=span_logits.dtype, device=span_logits.device)
    return nn.functional.binary_cross_entropy_with_logits(span_logits, target)


def _log_distribution(logits: torch.Tensor, kind: str) -> torch.Tensor:
    """Log-probabilities over a head's labels; a Noul's single logit as (no, yes)."""
    if kind == "noul":
        return torch.cat([nn.functional.logsigmoid(-logits), nn.functional.logsigmoid(logits)])
    return torch.log_softmax(logits, dim=-1)


def consistency_loss(
    variant: torch.Tensor, anchor: torch.Tensor, kind: str, relation: str = "same"
) -> torch.Tensor:
    """How far two answers are from what their pairing says they must be.

    The adversarial + paired stream (`trigon.evals.paired`). For a pair whose
    label is **held fixed** -- an injection, padding, a paraphrase -- the two
    answers should be one distribution, and the term is their symmetric KL
    divergence. For a **negation** pair it is ``(P(yes) + P(yes on the
    complement) - 1) ** 2``: coherence, not agreement.

    Not a scoring rule and not a substitute for one. Two answers can agree
    perfectly and both be wrong, and a model that ignores its input minimises
    this exactly; the cross-entropy beside it is what keeps the answers
    honest, which is why this is an auxiliary term with its weight at 0 by
    default. Zero for identical answers (or exactly complementary ones),
    positive otherwise, and gradients flow through both sides.
    """
    if relation == "complement":
        if kind != "noul":
            raise ValueError(f"a complement pair is two Nouls, not {kind!r}")
        return (torch.sigmoid(variant) + torch.sigmoid(anchor) - 1.0).pow(2).sum()
    if relation != "same":
        raise ValueError(f"relation must be 'same' or 'complement', got {relation!r}")
    log_p = _log_distribution(variant, kind)
    log_q = _log_distribution(anchor, kind)
    p, q = log_p.exp(), log_q.exp()
    return 0.5 * ((p * (log_p - log_q)).sum() + (q * (log_q - log_p)).sum())
