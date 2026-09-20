"""Fit the reference readout model on outcome-grounded data.

This exists so the calibration gates are passed by a model rather than only
defined by one. Until something has actually been trained and measured, a
calibration suite is a claim about a suite; once a run goes from failing the
gate to passing it, the loop is closed and every later change has a baseline
to regress against.

It is deliberately small: one request per forward pass, gradient accumulation
instead of padded batching, CPU-friendly defaults. The reference model is a
spike, not the workhorse, and the thing under test is the *pipeline* --
compile, forward, score, calibrate, gate -- not a training throughput number.
Padded batching is the first thing to add when this moves to a real backbone.

The one rule that matters here: training and serving share a single forward
path (``TorchReadoutBackend.logits``). A trainer with its own copy of the
forward pass is the standard way to end up with a model that scores well
offline and is miscalibrated in production.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ..evals.harness import Case
from ..schema import SchemaCompiler
from .losses import OrdinalConfig, question_loss

__all__ = ["TrainingConfig", "TrainingReport", "train"]


@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    # Requests per optimiser step. Accumulation, not padded batching.
    accumulate: int = 16
    grad_clip: float = 1.0
    seed: int = 0
    ordinal: OrdinalConfig = field(default_factory=OrdinalConfig)
    #: Torch intra-op threads. One is not a typo: a request compiles to a few
    #: hundred tokens at spike width, so every matmul is small enough that
    #: thread-launch overhead exceeds the work it parallelises. Measured on
    #: four cores, four threads ran this loop several times slower than one.
    #: Raise it only when the model is big enough for the parallelism to pay,
    #: and measure rather than assume.
    torch_threads: int | None = 1
    # Fraction of a warmup, as a share of total steps.
    warmup: float = 0.05
    log_every: int = 0


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    mean_loss: float
    seconds: float


@dataclass(frozen=True)
class TrainingReport:
    epochs: tuple[EpochReport, ...]
    n_cases: int
    n_questions: int
    seconds: float

    @property
    def first_loss(self) -> float:
        return self.epochs[0].mean_loss

    @property
    def final_loss(self) -> float:
        return self.epochs[-1].mean_loss

    def to_dict(self) -> dict:
        return {
            "epochs": [
                {"epoch": e.epoch, "mean_loss": e.mean_loss, "seconds": e.seconds}
                for e in self.epochs
            ],
            "n_cases": self.n_cases,
            "n_questions": self.n_questions,
            "seconds": self.seconds,
            "first_loss": self.first_loss,
            "final_loss": self.final_loss,
        }


def train(
    backend,
    cases: Sequence[Case],
    config: TrainingConfig | None = None,
    compiler: SchemaCompiler | None = None,
) -> TrainingReport:
    """Train ``backend``'s model in place on labelled cases.

    ``cases`` carry ground truth in the same ``Expectation`` shape the eval
    harness uses, so the training set and the eval set cannot disagree about
    what a label means.
    """
    config = config or TrainingConfig()
    if not cases:
        raise ValueError("nothing to train on")
    compiler = compiler or backend.make_compiler()
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    if config.torch_threads is not None:
        torch.set_num_threads(config.torch_threads)

    model = backend.model
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    order = list(range(len(cases)))
    steps_per_epoch = max(1, math.ceil(len(order) / config.accumulate))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(total_steps * config.warmup))
    step = 0

    started = time.perf_counter()
    epoch_reports: list[EpochReport] = []
    counted_questions = 0

    for epoch in range(config.epochs):
        rng.shuffle(order)
        epoch_started = time.perf_counter()
        total, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for position, index in enumerate(order, start=1):
            case = cases[index]
            compiled = compiler.compile_request(case.request)
            raw, _ = backend.logits(compiled, case.request)

            losses = []
            for compiled_q in compiled.schema.questions:
                qid = compiled_q.question_id
                expected = case.expected.get(qid)
                if expected is None:
                    continue
                label = expected.hard_label
                if label is None:
                    continue
                losses.append(question_loss(raw[qid], compiled_q.kind, label, config.ordinal))
            if not losses:
                continue
            if epoch == 0:
                counted_questions += len(losses)

            loss = torch.stack(losses).mean()
            (loss / config.accumulate).backward()
            total += float(loss.detach())
            seen += 1

            at_boundary = position % config.accumulate == 0
            if at_boundary or position == len(order):
                step += 1
                _set_lr(optimizer, config, step, total_steps, warmup_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if config.log_every and step % config.log_every == 0:
                    print(
                        f"  epoch {epoch + 1} step {step}/{total_steps} "
                        f"loss {total / max(seen, 1):.4f}",
                        flush=True,
                    )

        record = EpochReport(
            epoch=epoch + 1,
            mean_loss=total / max(seen, 1),
            seconds=time.perf_counter() - epoch_started,
        )
        epoch_reports.append(record)
        # Always on stderr, regardless of log_every: a training run with no
        # visible progress is indistinguishable from a hung one, and the first
        # epoch's loss is the cheapest signal that anything is learning at all.
        print(
            f"  epoch {record.epoch}/{config.epochs}: loss {record.mean_loss:.4f} "
            f"({record.seconds:.0f}s, {seen / max(record.seconds, 1e-9):.1f} cases/s)",
            file=sys.stderr,
            flush=True,
        )

    model.eval()
    return TrainingReport(
        epochs=tuple(epoch_reports),
        n_cases=len(cases),
        n_questions=counted_questions,
        seconds=time.perf_counter() - started,
    )


def _set_lr(optimizer, config: TrainingConfig, step: int, total: int, warmup: int) -> None:
    """Linear warmup, then cosine decay.

    Warmup matters more than usual here: the readout heads start from noise and
    an early large step pushes the whole distribution to one option, which is
    exactly the confidently-wrong behaviour the calibration objective exists to
    prevent.
    """
    if step <= warmup:
        scale = step / warmup
    else:
        progress = (step - warmup) / max(1, total - warmup)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate * scale
