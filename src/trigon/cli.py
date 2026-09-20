"""``trigon`` -- run the stack without writing any Python.

Every command works on a fresh clone with no weights and no GPU, because the
lexical floor is a real backend. That is deliberate: a reproducible-evals story
that requires a checkpoint before anyone can see it run is not reproducible.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence

from . import __version__

__all__ = ["main"]


def _engine(
    backend: str,
    domain: str | None = None,
    temperature_path: str | None = None,
    weights: str | None = None,
):
    from .calibration.temperature import TemperatureScaler
    from .engine import Engine, EngineConfig

    if backend == "lexical":
        from .backends.lexical import LexicalBackend

        impl = LexicalBackend()
        compiler = None
    elif backend == "torch":
        from .backends.torch_readout import TorchReadoutBackend

        impl = TorchReadoutBackend.load(weights) if weights else TorchReadoutBackend()
        compiler = impl.make_compiler()
    else:
        raise SystemExit(f"unknown backend {backend!r}; try 'lexical' or 'torch'")

    scaler = TemperatureScaler.load(temperature_path) if temperature_path else None
    return Engine(impl, compiler=compiler, scaler=scaler, config=EngineConfig(domain=domain))


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer one request read from a JSON file or stdin."""
    from .types import SystemOneRequest

    # Build the engine first: a bad --backend should fail before we consume
    # stdin, which the caller cannot rewind.
    engine = _engine(args.backend, args.domain, args.temperatures, args.weights)
    raw = sys.stdin.read() if args.request == "-" else pathlib.Path(args.request).read_text()
    response = engine.answer(SystemOneRequest.model_validate_json(raw))
    print(response.model_dump_json(indent=2, exclude_none=True))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError:
        raise SystemExit("serving needs the 'server' extra: pip install 'trigon[server]'") from None
    from .server.app import build_app
    from .server.config import ServerConfig

    config = ServerConfig.from_env()
    config.backend = args.backend
    if args.weights:
        config.weights = args.weights
    if config.backend == "torch" and not config.weights:
        print(
            "warning: serving torch with no --weights means randomly initialised "
            "weights; /healthz will report trained=false",
            file=sys.stderr,
        )
    uvicorn.run(build_app(config), host=args.host, port=args.port)
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    from .server.app import build_app

    print(json.dumps(build_app().openapi(), indent=2, sort_keys=True))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals import (
        all_workflows,
        blocking,
        check_gates,
        render_json,
        render_markdown,
        run_calibration_suite,
        run_cardinality_gate,
        run_jaggedness,
        run_workflow,
        synthetic_outcome_cases,
    )

    engine = _engine(args.backend, args.domain, args.temperatures)
    results, gates, slices = [], [], {}

    if args.suite in {"calibration", "all"}:
        cases = synthetic_outcome_cases(n=args.n, seed=args.seed, noise=args.noise)
        result, slices = run_calibration_suite(engine, cases)
        results.append(result)
        gates = check_gates(result, tier=args.tier)

    if args.suite in {"jaggedness", "all"}:
        results.extend(run_jaggedness(engine, n=args.n, seed=args.seed))

    cardinality: list = []
    if args.suite in {"cardinality", "all"}:
        # Decision D1's falsifier. Capped by default because the top of the
        # range is minutes, not seconds; --cardinality-max opens it up.
        counts = tuple(c for c in (256, 1_024, 4_096, 10_000) if c <= args.cardinality_max)
        cardinality = run_cardinality_gate(option_counts=counts, n_queries=args.n, seed=args.seed)

    workflows: list = []
    if args.suite in {"workflow", "all"}:
        workflows = [
            run_workflow(engine, wf, cases) for wf, cases in all_workflows(n=args.n, seed=args.seed)
        ]

    if not results and not cardinality and not workflows:
        raise SystemExit(f"unknown suite {args.suite!r}")

    markdown = render_markdown(results, gates, slices, cardinality, workflows)
    print(markdown)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        out.with_suffix(".json").write_text(
            render_json(results, gates, slices, cardinality, workflows)
        )
        print(f"\nwrote {out} and {out.with_suffix('.json')}", file=sys.stderr)

    # Non-zero exit on any failed gate, so CI can depend on this directly.
    passed = all(g.passed for g in blocking(gates)) and all(c.passed for c in cardinality)
    return 0 if passed else 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train the reference model, then measure and calibrate it.

    This is the loop the whole repo exists to support, run end to end on one
    machine with no weights to download: generate outcome-grounded data, fit
    the readout heads against proper scoring rules, measure calibration on a
    held-out split, fit a temperature, and put the result through the release
    gates. Its value is that the gates are passed (or failed) by a model
    rather than asserted about one.
    """
    import json as _json

    from .backends.torch_readout import ReadoutConfig, TorchReadoutBackend
    from .engine import Engine
    from .evals import (
        blocking,
        check_gates,
        render_markdown,
        run_calibration_suite,
        synthetic_outcome_cases,
    )
    from .schema import OptionScoring
    from .training import TrainingConfig
    from .training import train as run_training

    backend = TorchReadoutBackend(
        ReadoutConfig(
            d_model=args.d_model,
            n_layers=args.layers,
            match_normalize=args.match_normalize,
            match_residual=args.match_residual,
        ),
        seed=args.seed,
    )
    compiler = backend.make_compiler(option_scoring=OptionScoring(args.option_scoring))

    train_cases = synthetic_outcome_cases(n=args.n, seed=args.seed, noise=args.noise)
    # A separate seed, so the held-out split is genuinely unseen rather than a
    # reshuffle of the same generated records.
    eval_cases = synthetic_outcome_cases(n=args.eval_n, seed=args.seed + 1000, noise=args.noise)

    print(f"training on {len(train_cases)} cases, {args.epochs} epochs", file=sys.stderr)
    report = run_training(
        backend,
        train_cases,
        TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            accumulate=args.accumulate,
            seed=args.seed,
            log_every=args.log_every,
        ),
        compiler=compiler,
    )
    engine = Engine(backend, compiler=compiler)
    before, _ = run_calibration_suite(
        engine, eval_cases, suite="calibration/uncalibrated", floor_trials=args.floor_trials
    )

    # Fit the temperature on the training split, never on the split the gates
    # are read from -- fitting and reporting on the same data is how a
    # calibration number stops meaning anything.
    scaler = _fit_temperatures(engine, train_cases)
    calibrated = Engine(backend, compiler=compiler, scaler=scaler)
    after, slices = run_calibration_suite(
        calibrated,
        eval_cases,
        suite="calibration/temperature-scaled",
        floor_trials=args.floor_trials,
    )
    gates = check_gates(after)
    markdown = _training_section(report, args) + render_markdown([before, after], gates, slices)

    print(markdown)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        # Sidecars are named after the report, not fixed: two runs writing into
        # one directory (an ablation, say) must not clobber each other's
        # temperatures and loss curves.
        stem = out.with_suffix("")
        temperatures = pathlib.Path(f"{stem}-temperatures.json")
        training = pathlib.Path(f"{stem}-training.json")
        scaler.save(temperatures)
        training.write_text(_json.dumps(report.to_dict(), indent=2))
        written = [out.name, temperatures.name, training.name]
        if args.save_model:
            weights = pathlib.Path(args.save_model)
            backend.save(weights)
            written.append(weights.name)
        print(f"\nwrote {', '.join(written)}", file=sys.stderr)
    return 0 if all(g.passed for g in blocking(gates)) else 1


def _training_section(report, args) -> str:
    """Put the run's own settings and loss curve above the eval report.

    A calibration number is not interpretable without the run that produced it,
    and the Bayes-optimal loss is computable for this generator -- so the
    report states how far from optimal the model actually got rather than
    leaving the reader to guess whether the loss is good.
    """
    import math

    correct = 1.0 - args.noise + args.noise / 4
    wrong = args.noise / 4
    floor_categorical = -(correct * math.log(correct) + 3 * wrong * math.log(wrong))
    floor_binary = -(
        (1 - args.noise) * math.log(1 - args.noise) + args.noise * math.log(args.noise)
    )
    # Two categorical questions and one Noul per case.
    bayes = (2 * floor_categorical + floor_binary) / 3
    chance = (2 * math.log(4) + math.log(2)) / 3

    # Every setting that moves a number in this report, as a command that can
    # be pasted back. The prose version left out --eval-n and --floor-trials,
    # so a report could not in fact be reproduced from what it printed.
    command = (
        f"trigon train -n {args.n} --eval-n {args.eval_n} --epochs {args.epochs} "
        f"--lr {args.lr} --accumulate {args.accumulate} --d-model {args.d_model} "
        f"--layers {args.layers} --noise {args.noise} --seed {args.seed} "
        f"--floor-trials {args.floor_trials} --option-scoring {args.option_scoring}"
        + (" --match-normalize" if args.match_normalize else "")
        + (" --match-residual" if args.match_residual else "")
    )
    lines = [
        "# Reference run",
        "",
        "Everything below is reproducible from this command — the run is seeded "
        "end to end, so it returns the same weights and the same gate verdicts:",
        "",
        "```bash",
        command,
        "```",
        "",
        "| Epoch | Mean loss | Seconds |",
        "| ---: | ---: | ---: |",
    ]
    for epoch in report.epochs:
        lines.append(f"| {epoch.epoch} | {epoch.mean_loss:.4f} | {epoch.seconds:.0f} |")
    achieved = (chance - report.final_loss) / max(chance - bayes, 1e-9)
    lines += [
        "",
        f"Chance is `{chance:.4f}` and the Bayes-optimal loss for this generator "
        f"is `{bayes:.4f}` — the label noise puts a floor under how well anything "
        f"can do. The run closed **{achieved:.0%}** of that gap.",
        "",
    ]
    return "\n".join(lines)


def _fit_temperatures(engine, cases):
    """Fit one temperature per primitive on the training split."""
    import math
    import warnings

    from .calibration.temperature import CalibrationWarning, TemperatureScaler
    from .evals import run_cases

    scaler = TemperatureScaler()
    rows: dict[str, list[tuple[list[float], int]]] = {}
    for outcome in run_cases(engine, cases):
        for question in outcome.questions.values():
            if question.expected is None or question.expected.hard_label is None:
                continue
            logits = [math.log(max(p, 1e-12)) for p in question.probabilities]
            rows.setdefault(question.primitive, []).append((logits, question.expected.hard_label))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        for primitive, data in sorted(rows.items()):
            if primitive == "noul":
                scaler.fit_binary([r[0][1] - r[0][0] for r in data], [y for _, y in data])
            else:
                scaler.fit(primitive, [x for x, _ in data], [y for _, y in data])
    print(f"  temperatures: {scaler.primitive}", file=sys.stderr)
    return scaler


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the post-hoc calibration layer on held-out labelled data.

    This is the tool the build plan's biggest risk depends on: outcome-grounded
    calibration fitted on public and synthetic data may not transfer to a
    user's domain, and the mitigation is that they can fit a conformal wrapper
    on a few hundred of their own labels. A mitigation that exists only in a
    design document is not a mitigation, so it ships as a command.
    """
    import math

    from .calibration.conformal import ConformalMethod, fit_conformal
    from .calibration.temperature import TemperatureScaler
    from .evals import run_cases, synthetic_outcome_cases

    engine = _engine(args.backend, args.domain, weights=args.weights)
    outcomes = run_cases(
        engine, synthetic_outcome_cases(n=args.n, seed=args.seed, noise=args.noise)
    )
    scaler = TemperatureScaler()
    by_primitive: dict[str, list[tuple[list[float], int]]] = {}
    for outcome in outcomes:
        for question in outcome.questions.values():
            if question.expected is None or question.expected.hard_label is None:
                continue
            # Recover logits from probabilities: temperature scaling is
            # invariant to an additive constant, so log p is sufficient.
            logits = [math.log(max(p, 1e-12)) for p in question.probabilities]
            by_primitive.setdefault(question.primitive, []).append(
                (logits, question.expected.hard_label)
            )

    for primitive, rows in sorted(by_primitive.items()):
        if primitive == "noul":
            value = scaler.fit_binary(
                [row[0][1] - row[0][0] for row in rows], [y for _, y in rows], args.domain
            )
        else:
            value = scaler.fit(primitive, [x for x, _ in rows], [y for _, y in rows], args.domain)
        print(f"{primitive}: T={value:.4f} on {len(rows)} examples", file=sys.stderr)

    scaler.save(args.out)
    print(json.dumps(scaler.to_dict(), indent=2, sort_keys=True))

    if args.conformal_out:
        # Fitted on a split the temperatures never saw, and reported on a
        # third: a coverage number measured where the threshold was fitted is
        # not a coverage number.
        calib = synthetic_outcome_cases(n=args.n, seed=args.seed + 500, noise=args.noise)
        test = synthetic_outcome_cases(n=args.n, seed=args.seed + 900, noise=args.noise)
        rows = _choice_rows(engine, calib)
        held = _choice_rows(engine, test)
        predictor = fit_conformal(
            [p for p, _ in rows],
            [y for _, y in rows],
            alpha=args.alpha,
            method=ConformalMethod(args.conformal_method),
        )
        out = pathlib.Path(args.conformal_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        predictor.save(out)

        from .calibration import coverage, mean_set_size

        sets = [predictor.predict(p).indices for p, _ in held]
        achieved = coverage(sets, [y for _, y in held])
        print(
            f"\nconformal '{out.stem}': target {predictor.target_coverage:.2f}, "
            f"achieved {achieved:.4f} on {len(held)} held-out answers, "
            f"mean set {mean_set_size(sets):.2f} of 4 options",
            file=sys.stderr,
        )
    return 0


def _choice_rows(engine, cases) -> list[tuple[list[float], int]]:
    """Choice distributions and their true labels, aligned."""
    from .evals import run_cases

    rows = []
    for outcome in run_cases(engine, cases):
        for question in outcome.questions.values():
            if question.primitive != "choice" or question.expected is None:
                continue
            label = question.expected.hard_label
            if label is not None:
                rows.append((list(question.probabilities), label))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trigon", description=__doc__)
    parser.add_argument("--version", action="version", version=f"trigon {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--backend", default="lexical", help="lexical (default) or torch")
        p.add_argument("--domain", default=None, help="domain for per-domain calibration")

    ask = sub.add_parser("ask", help="answer one request from a JSON file or '-'")
    shared(ask)
    ask.add_argument("request")
    ask.add_argument("--temperatures", default=None, help="path to fitted temperatures")
    ask.add_argument("--weights", default=None, help="a checkpoint written by 'trigon train'")
    ask.set_defaults(func=cmd_ask)

    serve = sub.add_parser("serve", help="run the reference gateway")
    shared(serve)
    serve.add_argument("--weights", default=None, help="a checkpoint written by 'trigon train'")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    spec = sub.add_parser("spec", help="print the OpenAPI spec")
    spec.set_defaults(func=cmd_spec)

    ev = sub.add_parser("eval", help="run an eval suite; exits non-zero on a failed gate")
    shared(ev)
    ev.add_argument(
        "suite", choices=["calibration", "jaggedness", "cardinality", "workflow", "all"]
    )
    ev.add_argument(
        "--cardinality-max",
        type=int,
        default=4_096,
        help="largest option count in the cardinality gate (10000 is minutes, not seconds)",
    )
    ev.add_argument("-n", type=int, default=100, help="cases per benchmark")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--noise", type=float, default=0.1, help="label noise in the synthetic set")
    ev.add_argument("--tier", default="workhorse", choices=["workhorse", "premium"])
    ev.add_argument("--temperatures", default=None)
    ev.add_argument("--out", default=None, help="write the report here (plus a .json sibling)")
    ev.set_defaults(func=cmd_eval)

    fit = sub.add_parser("fit", help="fit temperatures on the synthetic outcome set")
    shared(fit)
    fit.add_argument("--out", default="temperatures.json")
    fit.add_argument("-n", type=int, default=400)
    fit.add_argument("--seed", type=int, default=1)
    fit.add_argument("--noise", type=float, default=0.1)
    fit.add_argument("--weights", default=None, help="a checkpoint written by 'trigon train'")
    fit.add_argument(
        "--conformal-out",
        default=None,
        help="also fit a conformal profile and write it here",
    )
    fit.add_argument("--alpha", type=float, default=0.1, help="1 - target coverage")
    fit.add_argument(
        "--conformal-method",
        default="lac",
        choices=["lac", "aps"],
        help="lac gives the smallest sets; aps tracks difficulty",
    )
    fit.set_defaults(func=cmd_fit)

    tr = sub.add_parser("train", help="train the reference model, calibrate it, and run the gates")
    tr.add_argument("-n", type=int, default=3000, help="training cases")
    tr.add_argument("--eval-n", type=int, default=6000, help="held-out cases for the gates")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-2)
    tr.add_argument("--accumulate", type=int, default=16)
    tr.add_argument(
        "--noise",
        type=float,
        default=0.2,
        help="irreducible label noise; 0 makes calibration untestable",
    )
    tr.add_argument("--d-model", type=int, default=192)
    tr.add_argument("--layers", type=int, default=3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--floor-trials", type=int, default=100)
    tr.add_argument(
        "--match-normalize",
        action="store_true",
        help="dot-product head: cosine similarity with a learnable temperature",
    )
    tr.add_argument(
        "--match-residual",
        action="store_true",
        help="dot-product head: add each option's input embedding to its key",
    )
    tr.add_argument("--log-every", type=int, default=25)
    tr.add_argument(
        "--option-scoring",
        default="auto",
        choices=["auto", "readout_per_option", "dot_product"],
        help="phase-1 ablation: a readout slot per option, or one slot dotted "
        "with pooled option states",
    )
    tr.add_argument("--out", default=None, help="write the report here")
    tr.add_argument(
        "--save-model",
        default=None,
        help="write the trained weights here, so the run can be served",
    )
    tr.set_defaults(func=cmd_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
