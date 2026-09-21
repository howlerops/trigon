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
from trigon.schema import OptionScoring  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="a corpus name from trigon.evals.corpora")
    parser.add_argument("--out", default=None, help="write the report here")
    parser.add_argument("--save-model", default=None)
    parser.add_argument("-n", type=int, default=4000, help="training cases (0 = all)")
    parser.add_argument("--calibration-n", type=int, default=1000)
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--floor-trials", type=int, default=200)
    parser.add_argument("--option-scoring", default="auto")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def splits(spec, args) -> tuple[list, list, list]:
    """Train and calibration out of the train split, evaluation out of test.

    The calibration split is carved from train rather than from test for the
    reason the whole repo is built around: a calibrator fitted on the data the
    gates read reports its own fit, not the model's calibration. Shuffled with
    a seeded RNG so the carve is reproducible and so it is not the corpus's own
    ordering, which on several of these is grouped by label.
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
    training = remaining if args.n == 0 else remaining[: args.n]
    if args.eval_n:
        evaluation = evaluation[: args.eval_n]
    return training, calibration, evaluation


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


def header(spec, args, train, calibration, evaluation, marginal: dict[str, float]) -> str:
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
            f"| Evaluation cases (the corpus's own test split) | {len(evaluation):,} |",
            f"| Questions per request | {len(questions)} |",
            f"| Labels per question | {len(labels)} |",
            "",
            "| Question | Marginal predictor |",
            "| --- | ---: |",
            *(f"| `{qid}` | {value:.4f} |" for qid, value in sorted(marginal.items())),
            f"| Epochs | {args.epochs} |",
            f"| Seed | {args.seed} |",
            "",
            "```",
            f"python scripts/train_corpus.py {spec.name} -n {args.n} "
            f"--calibration-n {args.calibration_n} --eval-n {args.eval_n} "
            f"--epochs {args.epochs} --lr {args.lr} --accumulate {args.accumulate} "
            f"--d-model {args.d_model} --layers {args.layers} --seed {args.seed}",
            "```",
            "",
            "**One seed is one sample from a distribution nobody measured.**",
            "See `CLAUDE.md`: certify on the median and the spread, never on a",
            "single draw. This report is a measurement, not a certification.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from trigon.backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from trigon.training import TrainingConfig
    from trigon.training import train as run_training

    spec = corpus(args.corpus)
    backend = TorchReadoutBackend(
        config=ReadoutConfig(d_model=args.d_model, n_layers=args.layers), seed=args.seed
    )
    compiler = backend.make_compiler(option_scoring=OptionScoring(args.option_scoring))
    train, calibration, evaluation = splits(spec, args)
    marginal = baseline_accuracy(train, evaluation)
    summary = " ".join(f"{qid}={value:.4f}" for qid, value in sorted(marginal.items()))
    print(
        f"{spec.name}: {len(train)} train, {len(calibration)} calibration, "
        f"{len(evaluation)} eval; marginals {summary}",
        file=sys.stderr,
    )
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
        ),
        compiler=compiler,
    )
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
    markdown = header(spec, args, train, calibration, evaluation, marginal) + render_markdown(
        [before, after], gates, slices
    )
    print(markdown)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        out.with_suffix(".json").write_text(render_json([before, after], gates, slices))
        stem = out.with_suffix("")
        scaler.save(pathlib.Path(f"{stem}-temperatures.json"))
        if isotonic.knots:
            isotonic.save(pathlib.Path(f"{stem}-isotonic.json"))
        pathlib.Path(f"{stem}-training.json").write_text(json.dumps(report.to_dict(), indent=2))
        if args.save_model:
            backend.save(pathlib.Path(args.save_model))
        print(f"\nwrote {out}", file=sys.stderr)
    return 0 if all(g.passed for g in blocking(gates)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
