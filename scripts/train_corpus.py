"""Train and calibrate on a real corpus, and publish the calibration per corpus.

Every calibration number this project has published came from synthetic data:
generated state, decidable predicates, no subjectivity. That exercises the
machinery honestly and it cannot tell you whether the machinery transfers,
which is Stage 2 of `docs/plan.md` and the reason this script exists.

`trigon train` is not reused as-is on purpose. Its report computes a
Bayes-optimal loss, which it can do because it knows the generator and its
noise rate; on a real corpus there is no such number, and a report that
printed one would be inventing a floor. What is reused is everything below
that: the trainer, the per-primitive calibrator selection, the suite, and the
gates.

**The baseline is what makes the number mean anything.** On Banking77 the
marginal predictor scores about 1.3%, so `accuracy_over_baseline` is a real
test here in a way it is not on a four-option synthetic question. A model that
ignores its input cannot pass it by accident.

Splits are disjoint by construction and the calibration split is carved out of
*train*, never out of the evaluation split. Fitting a calibrator on the data
the gates then read is how a calibration number stops meaning anything, and
this repo has made that mistake once already.

    python scripts/train_corpus.py banking77 --out reports/banking77/run.md \
        -n 4000 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.cli import _fit_calibration  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals import (  # noqa: E402
    blocking,
    check_gates,
    render_json,
    render_markdown,
    run_calibration_suite,
)
from trigon.evals.corpora import corpus, load  # noqa: E402
from trigon.limits import MIN_CALIBRATION_SAMPLES  # noqa: E402
from trigon.schema import OptionScoring  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="a corpus name from trigon.evals.corpora")
    parser.add_argument("--out", default=None, help="write the report here")
    parser.add_argument("--save-model", default=None)
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "re-calibrate and re-gate an existing checkpoint instead of training. "
            "The corpus analogue of scripts/regate.py: the splits are derived from "
            "--seed and the corpus, so a re-run with the same flags reads the same "
            "data the training run held out"
        ),
    )
    parser.add_argument("-n", type=int, default=4000, help="training cases (0 = all)")
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument(
        "--eval-n",
        type=int,
        default=0,
        help=(
            "evaluation cases; 0 (the default) means as many as "
            "MIN_CALIBRATION_SAMPLES requires, supplementing from unseen train "
            "rows when the corpus's own test split is smaller than the floor"
        ),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--floor-trials", type=int, default=200)
    parser.add_argument("--option-scoring", default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "torch intra-op threads. One by default so several seeds in parallel "
            "are cores rather than oversubscription, and so the latency the report "
            "publishes is measured under a known thread count"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--backbone",
        default=None,
        help=(
            "a pinned pretrained backbone from trigon.backends.hub (e.g. qwen2.5-1.5b) "
            "instead of the spike; --d-model and --layers are then the backbone's"
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--resume-path",
        default=None,
        help="write a resume file each epoch and continue from it if present",
    )
    parser.add_argument(
        "--max-batch-cells",
        type=int,
        default=0,
        help=(
            "cap batch x sequence^2 per forward pass, splitting a step into "
            "sub-batches that accumulate into the same update; 0 is no cap. "
            "A GPU needs one on long-tailed corpora -- see TrainingConfig"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "cpu, cuda, or auto (cuda when present). Naming cuda on a machine "
            "without one is an error, not a fallback: a run that asked for a GPU "
            "and quietly trained on the CPU publishes a report about hardware it "
            "never used"
        ),
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> tuple[str, str]:
    """The device to train on, and the name of the hardware behind it."""
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {requested} was asked for and no CUDA device is present")
        return requested, torch.cuda.get_device_name(torch.device(requested))
    import platform

    return requested, platform.processor() or platform.machine()


def splits(spec, args) -> tuple[list, list, list, int]:
    """Train and calibration out of the train split, evaluation out of test.

    The calibration split is carved from train rather than from test for the
    reason the whole repo is built around: a calibrator fitted on the data the
    gates read reports its own fit, not the model's calibration. Shuffled with
    a seeded RNG so the carve is reproducible and so it is not the corpus's own
    ordering, which on several of these is grouped by label.

    **The evaluation split is topped up from train when the corpus's own test
    split is below `MIN_CALIBRATION_SAMPLES`, and this is the interesting
    part.** Banking77's test split is 3,080 rows and the floor is 5,000, so
    the first run of this script failed `sample_size` and `gate_is_testable`
    on every seed: a perfectly calibrated model scores ECE 0.0221 on 1,500
    cases, which is most of the 0.05 gate, so the gate was not a test. That is
    not a threshold to widen -- `CLAUDE.md` is explicit that the run gets
    fixed rather than the gate -- and it is a real defect in the plan's Stage 2
    done-condition, which asks for per-corpus ECE without noticing that a
    corpus can be too small to carry one.

    The top-up rows are disjoint from both training and calibration and were
    never seen by either, so they are held-out data by the only definition
    that matters. They come from the train *distribution* rather than the
    test one, which is a real difference and is why the report states how many
    of each it used instead of printing one number.
    """
    train = load(spec.name, "train", purpose="train")
    evaluation = load(spec.name, "test", purpose="eval")
    # Both splits, and this is not defensive tidying. Banking77's test.csv is
    # ordered by label, so the first version of this took `evaluation[:1000]`
    # and got an evaluation set of one intent: the marginal predictor scored
    # 1.0000 and `accuracy_over_baseline` read -1.0000. A gate cannot catch a
    # subset that is not the distribution, because nothing about the number
    # looks wrong -- the model really did lose to that baseline.
    rng = random.Random(f"corpus:{spec.name}:{args.seed}")
    rng.shuffle(train)
    rng.shuffle(evaluation)

    calibration_n = min(args.calibration_n, len(train) // 4)
    calibration = train[:calibration_n]
    remaining = train[calibration_n:]

    if args.eval_n:
        evaluation = evaluation[: args.eval_n]
        topped_up = 0
    else:
        # As many as the floor requires, taken from rows nothing else will see
        # -- but never more than half the pool, because the top-up must not
        # eat the training set. The first version capped it at `len(remaining)
        # - 1` and, on a corpus with a 4,000-row train split, reached the floor
        # by leaving **one** training case. Every count in the report was
        # correct and the report was worthless.
        #
        # When half the pool is not enough to reach the floor, the floor is
        # not reached and `sample_size` fails. That is the right answer: a
        # corpus too small to supply both a trainable set and a floor-sized
        # evaluation is a corpus this gate cannot certify, and cannibalising
        # training to make the gate pass is the same move as widening it.
        wanted = max(0, MIN_CALIBRATION_SAMPLES - len(evaluation))
        topped_up = min(wanted, len(remaining) // 2)
        evaluation = evaluation + remaining[:topped_up]
        remaining = remaining[topped_up:]

    training = remaining if args.n == 0 else remaining[: args.n]
    return training, calibration, evaluation, topped_up


def baseline_accuracy(train: list, evaluation: list) -> dict[str, float]:
    """What a model that ignores its input scores, per question.

    The modal label is computed from the *training* split and applied to
    evaluation, because a baseline fitted on the evaluation split is not a
    baseline, it is an oracle with one degree of freedom.

    Per question rather than pooled. A corpus like HelpSteer2 asks five Score
    questions over the same state and their marginals differ -- pooling them
    into one number would let a well-predicted question carry a badly
    predicted one, which is the same cancellation that made pooled ECE
    misleading here before.
    """
    counts: dict[str, dict[int, int]] = {}
    for case in train:
        for qid, expectation in case.expected.items():
            counts.setdefault(qid, {})
            label = expectation.label
            counts[qid][label] = counts[qid].get(label, 0) + 1
    out: dict[str, float] = {}
    for qid, tally in counts.items():
        modal = max(tally, key=lambda k: tally[k])
        seen = [c for c in evaluation if qid in c.expected]
        hits = sum(1 for c in seen if c.expected[qid].label == modal)
        out[qid] = hits / len(seen) if seen else 0.0
    return out


def header(
    spec, args, train, calibration, evaluation, marginal, topped_up: int, hardware: str = ""
) -> str:
    questions = train[0].request.questions
    first = next(iter(questions.values()))
    labels = getattr(first, "options", None) or getattr(first, "levels", [])
    return "\n".join(
        [
            f"# {spec.name}",
            "",
            f"**{spec.attribution}**",
            "",
            "Real labelled data, not the synthetic generator. There is no",
            "Bayes-optimal loss to quote here -- the labels are human and the",
            "corpus does not come with a noise rate -- so the floor reported",
            "below is the marginal predictor, which is the floor that matters",
            "for the `accuracy_over_baseline` gate.",
            "",
            "| | |",
            "| --- | ---: |",
            f"| Training cases | {len(train):,} |",
            f"| Calibration cases (held out of train) | {len(calibration):,} |",
            f"| Evaluation cases | {len(evaluation):,} |",
            f"| — from the corpus's own test split | {len(evaluation) - topped_up:,} |",
            f"| — held out of train to reach the floor | {topped_up:,} |",
            f"| Questions per request | {len(questions)} |",
            f"| Labels per question | {len(labels)} |",
            "",
            "| Question | Marginal predictor |",
            "| --- | ---: |",
            *(f"| `{qid}` | {value:.4f} |" for qid, value in sorted(marginal.items())),
            f"| Epochs | {args.epochs} |",
            f"| Seed | {args.seed} |",
            *([f"| Device | {hardware} |"] if hardware else []),
            (
                f"| Model | {args.backbone}, LoRA rank {args.lora_rank} |"
                if args.backbone
                else f"| Model | reference spike, d_model {args.d_model}, {args.layers} layers |"
            ),
            "",
            "```",
            f"python scripts/train_corpus.py {spec.name} -n {args.n} "
            f"--calibration-n {args.calibration_n} --eval-n {args.eval_n} "
            f"--epochs {args.epochs} --lr {args.lr} --accumulate {args.accumulate} "
            f"--d-model {args.d_model} --layers {args.layers} --seed {args.seed} "
            f"--device {args.device}"
            + (f" --backbone {args.backbone} --lora-rank {args.lora_rank}" if args.backbone else "")
            + (f" --max-batch-cells {args.max_batch_cells}" if args.max_batch_cells else ""),
            "```",
            "",
            *(
                [
                    f"`MIN_CALIBRATION_SAMPLES` is {MIN_CALIBRATION_SAMPLES:,} and this",
                    "corpus's test split is smaller, so the evaluation set is topped up",
                    "from rows held out of train that neither training nor calibration",
                    "saw. They are held-out data by the only definition that matters and",
                    "they come from the train distribution, which is why the split is",
                    "reported above rather than summed into one number.",
                    "",
                ]
                if topped_up
                else []
            ),
            "**One seed is one sample from a distribution nobody measured.**",
            "See `CLAUDE.md`: certify on the median and the spread, never on a",
            "single draw. This report is a measurement, not a certification.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Set here rather than inherited from the trainer, which is where it used
    # to come from: `TrainingConfig.torch_threads` is 1 and `run_training`
    # applies it, so every eval after a training run was single-threaded by
    # accident. Re-gating a checkpoint skips training, so four seeds in
    # parallel each took four threads on four cores -- 3.5x slower, and worse
    # than slow, because the suite publishes p50 and p99 latency and those
    # numbers are meaningless under oversubscription. The accuracy and the ECE
    # would have been right and the latency quietly wrong.
    import torch

    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.training import TrainingConfig
    from trigon.training import train as run_training

    torch.set_num_threads(args.torch_threads)
    device, hardware = resolve_device(args.device)

    spec = corpus(args.corpus)
    if args.weights:
        backend = TorchReadoutBackend.load(args.weights)
    elif args.backbone:
        from trigon.backends.qwen_readout import QwenReadoutBackend

        # Loaded straight onto the device: a 1.5B model materialised on the
        # CPU first and moved costs a second copy of it in a container's RAM.
        backend = QwenReadoutBackend.from_backbone(
            args.backbone, device=device, seed=args.seed, lora_rank=args.lora_rank
        )
        backend.model.checkpointing = True
    else:
        backend = TorchReadoutBackend(
            config=ReadoutConfig(d_model=args.d_model, n_layers=args.layers), seed=args.seed
        )
    backend.to(device)
    hardware = f"{device} ({hardware})"
    print(f"{args.corpus}: training on {hardware}", file=sys.stderr)
    compiler = backend.make_compiler(option_scoring=OptionScoring(args.option_scoring))
    train, calibration, evaluation, topped_up = splits(spec, args)
    marginal = baseline_accuracy(train, evaluation)
    summary = " ".join(f"{qid}={value:.4f}" for qid, value in sorted(marginal.items()))
    print(
        f"{spec.name}: {len(train)} train, {len(calibration)} calibration, "
        f"{len(evaluation)} eval; marginals {summary}",
        file=sys.stderr,
    )
    report = None
    if not args.weights:
        report = run_training(
            backend,
            train,
            TrainingConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                accumulate=args.accumulate,
                seed=args.seed,
                log_every=args.log_every,
                validation_fraction=args.validation_fraction,
                max_batch_cells=args.max_batch_cells or None,
                resume_path=args.resume_path,
            ),
            compiler=compiler,
        )
    if device.startswith("cuda"):
        # Sizing the GPU for the next run needs this, and nothing else records it.
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{args.corpus}: peak GPU memory in training {peak:.1f} GiB", file=sys.stderr)
    # **Evaluate the path we serve.** The gateway defaults the schema prefix
    # cache on (reports/cache/README.md: 6x at 77 options), so a report
    # measured with it off describes a deployment nobody runs. Safe here for
    # the reason it is unsafe in training: the weights are fixed now, and
    # `_prefix_for` refuses to cache while the model is in training mode
    # anyway. It moves answers by ~5e-08, which is the same cross-shape
    # float32 rounding already in the ledger and below any gate.
    backend.cache_prefixes = True

    engine = Engine(backend, compiler=compiler)
    before, _ = run_calibration_suite(
        engine, evaluation, suite=f"{spec.name}/uncalibrated", floor_trials=args.floor_trials
    )
    scaler, isotonic = _fit_calibration(engine, calibration)
    calibrated = Engine(backend, compiler=compiler, scaler=scaler, isotonic=isotonic)
    after, slices = run_calibration_suite(
        calibrated, evaluation, suite=f"{spec.name}/calibrated", floor_trials=args.floor_trials
    )
    gates = check_gates(after, slices=slices)
    markdown = header(
        spec, args, train, calibration, evaluation, marginal, topped_up, hardware
    ) + render_markdown([before, after], gates, slices, gated=after)
    print(markdown)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        out.with_suffix(".json").write_text(
            render_json([before, after], gates, slices, gated=after)
        )
        stem = out.with_suffix("")
        scaler.save(pathlib.Path(f"{stem}-temperatures.json"))
        if isotonic.knots:
            isotonic.save(pathlib.Path(f"{stem}-isotonic.json"))
        if report is not None:
            pathlib.Path(f"{stem}-training.json").write_text(json.dumps(report.to_dict(), indent=2))
        if args.save_model:
            backend.save(pathlib.Path(args.save_model))
        print(f"\nwrote {out}", file=sys.stderr)
    return 0 if all(g.passed for g in blocking(gates)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
