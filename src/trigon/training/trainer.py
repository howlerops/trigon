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
    #: Chunks to draw from when bucketing by length. Each optimizer step pads
    #: its cases to the longest one, and attention is quadratic, so a chunk of
    #: mixed lengths wastes the difference. On the synthetic corpus that waste
    #: is 1.02x and invisible; on HelpSteer2, whose requests compile to between
    #: 253 and 3,647 tokens, it is **2.82x** at the default chunk.
    #:
    #: The epoch is still shuffled. This draws a window of
    #: `accumulate * bucket_window` cases from that shuffled order, sorts the
    #: window by compiled length, and cuts it into chunks -- so which cases
    #: share a gradient step is no longer independent of their length, which is
    #: a real change to the optimization and why this is a knob rather than the
    #: only behaviour. 1 disables it.
    bucket_window: int = 8
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
    #: Share of ``cases`` held out to pick which epoch to keep. The reference
    #: run's loss bottomed at epoch 4 and rose for the next four, so a run that
    #: keeps its last epoch ships weights it had already beaten. 0 disables the
    #: holdout and keeps the final epoch, which is the old behaviour.
    validation_fraction: float = 0.1


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    mean_loss: float
    seconds: float
    #: Loss on the held-out slice, when one was taken. This is what selects
    #: the kept epoch; ``mean_loss`` is the training loss and will keep
    #: falling after this one stops.
    validation_loss: float | None = None


@dataclass(frozen=True)
class TrainingReport:
    epochs: tuple[EpochReport, ...]
    n_cases: int
    n_questions: int
    seconds: float
    #: Which epoch's weights the model ended up holding. Not always the last.
    kept_epoch: int = 0

    @property
    def first_loss(self) -> float:
        return self.epochs[0].mean_loss

    @property
    def final_loss(self) -> float:
        return self.epochs[-1].mean_loss

    def to_dict(self) -> dict:
        return {
            "epochs": [
                {
                    "epoch": e.epoch,
                    "mean_loss": e.mean_loss,
                    "seconds": e.seconds,
                    "validation_loss": e.validation_loss,
                }
                for e in self.epochs
            ],
            "n_cases": self.n_cases,
            "n_questions": self.n_questions,
            "seconds": self.seconds,
            "first_loss": self.first_loss,
            "final_loss": self.final_loss,
            "kept_epoch": self.kept_epoch,
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

    # Held-out slice for choosing which epoch to keep. Taken deterministically
    # from a shuffle of the same seed, so a rerun holds out the same cases.
    holdout: list[Case] = []
    if config.validation_fraction > 0 and len(cases) >= 2 / config.validation_fraction:
        shuffled = list(cases)
        random.Random(config.seed + 7919).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * config.validation_fraction))
        holdout, cases = shuffled[:cut], shuffled[cut:]

    order = list(range(len(cases)))
    steps_per_epoch = max(1, math.ceil(len(order) / config.accumulate))
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, int(total_steps * config.warmup))
    step = 0

    started = time.perf_counter()
    # Compiled length per case, once. Used to bucket each optimizer step's
    # chunk by length so it pads less; see `_chunks`.
    lengths = (
        {i: compiler.compile_request(case.request).total_tokens for i, case in enumerate(cases)}
        if config.bucket_window > 1
        else {}
    )
    epoch_reports: list[EpochReport] = []
    counted_questions = 0
    best: tuple[float, int, dict] | None = None

    for epoch in range(config.epochs):
        rng.shuffle(order)
        epoch_started = time.perf_counter()
        total, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        # One batched forward per optimizer step rather than `accumulate`
        # sequential ones. The arithmetic is unchanged -- the step's loss is
        # still the mean over cases of the mean over that case's questions, and
        # `logits_batch` is asserted equal to `logits` to floating-point
        # equality -- but there is one backward pass instead of `accumulate` of
        # them, which is where the time goes.
        for chunk in _chunks(order, lengths, config):
            items = [
                (compiler.compile_request(cases[index].request), cases[index].request)
                for index in chunk
            ]
            batched = backend.logits_batch(items)

            case_losses = []
            for index, (compiled, _), raw in zip(chunk, items, batched, strict=True):
                case = cases[index]
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
                case_losses.append(torch.stack(losses).mean())

            if not case_losses:
                continue
            stacked = torch.stack(case_losses)
            stacked.mean().backward()
            total += float(stacked.detach().sum())
            seen += len(case_losses)

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

        validation = _validation_loss(backend, compiler, holdout, config) if holdout else None
        if validation is not None and (best is None or validation < best[0]):
            best = (
                validation,
                epoch + 1,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
            )

        record = EpochReport(
            epoch=epoch + 1,
            mean_loss=total / max(seen, 1),
            seconds=time.perf_counter() - epoch_started,
            validation_loss=validation,
        )
        epoch_reports.append(record)
        # Always on stderr, regardless of log_every: a training run with no
        # visible progress is indistinguishable from a hung one, and the first
        # epoch's loss is the cheapest signal that anything is learning at all.
        print(
            f"  epoch {record.epoch}/{config.epochs}: loss {record.mean_loss:.4f}"
            + (f" val {validation:.4f}" if validation is not None else "")
            + f" ({record.seconds:.0f}s, {seen / max(record.seconds, 1e-9):.1f} cases/s)",
            file=sys.stderr,
            flush=True,
        )

    kept = len(epoch_reports)
    if best is not None and best[1] != kept:
        # The last epoch is not the best one. Keeping it anyway ships weights
        # the run had already beaten -- which is not hypothetical: the 8,000
        # case run bottomed at epoch 4 and rose for the next four.
        model.load_state_dict(best[2])
        kept = best[1]
        print(
            f"  keeping epoch {kept} (validation {best[0]:.4f}), not the last",
            file=sys.stderr,
            flush=True,
        )

    model.eval()
    # The weights are final, so the build can be named after them -- before the
    # eval report is rendered, not only when a checkpoint is written. Otherwise
    # a report about a trained model names it "untrained".
    if hasattr(backend, "stamp_version"):
        backend.stamp_version()
    return TrainingReport(
        kept_epoch=kept,
        epochs=tuple(epoch_reports),
        n_cases=len(cases),
        n_questions=counted_questions,
        seconds=time.perf_counter() - started,
    )


@torch.no_grad()
def _chunks(order, lengths, config):
    """Optimizer-step chunks, bucketed by compiled length within a window.

    Each step pads its cases to the longest one and attention is quadratic, so
    a chunk of mixed lengths pays for the difference. Sorting a window of the
    already-shuffled order before cutting it into chunks keeps the epoch
    random at the scale of the window while making each chunk nearly uniform:
    measured on HelpSteer2, 2.82x of wasted attention becomes 1.04x.

    The window matters. Sorting the *whole* epoch would make every chunk
    perfectly uniform and would also mean the model sees all its short cases
    before any long one, which is a curriculum nobody chose. A window of
    `accumulate * bucket_window` is a few hundred cases: long enough to bucket
    well, short enough that the order stays shuffled where it counts.
    """
    step = config.accumulate
    if config.bucket_window <= 1:
        for start in range(0, len(order), step):
            yield order[start : start + step]
        return

    window = step * config.bucket_window
    for start in range(0, len(order), window):
        block = order[start : start + window]
        # `lengths` is computed once for the whole run rather than per epoch:
        # a case's compiled length does not change, and compiling twice to
        # sort by the result would cost more than the padding it saves.
        block.sort(key=lengths.__getitem__)
        for inner in range(0, len(block), step):
            yield block[inner : inner + step]


def _validation_loss(backend, compiler, holdout: Sequence[Case], config: TrainingConfig) -> float:
    """Mean loss on the held-out slice, in eval mode.

    Selection has to happen on data the gradients never saw, or it selects the
    epoch that memorised hardest rather than the one that generalised best.
    """
    was_training = backend.model.training
    backend.model.eval()
    try:
        total, seen = 0.0, 0
        for start in range(0, len(holdout), config.accumulate):
            chunk = holdout[start : start + config.accumulate]
            items = [(compiler.compile_request(c.request), c.request) for c in chunk]
            for case, (compiled, _), raw in zip(
                chunk, items, backend.logits_batch(items), strict=True
            ):
                losses = [
                    question_loss(
                        raw[q.question_id],
                        q.kind,
                        case.expected[q.question_id].hard_label,
                        config.ordinal,
                    )
                    for q in compiled.schema.questions
                    if case.expected.get(q.question_id) is not None
                    and case.expected[q.question_id].hard_label is not None
                ]
                if losses:
                    total += float(torch.stack(losses).mean())
                    seen += 1
        return total / max(seen, 1)
    finally:
        backend.model.train(was_training)


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
