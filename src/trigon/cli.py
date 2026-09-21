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
    # The isotonic map is written beside the temperatures by `trigon train`,
    # so it is found beside them here. Loading one without the other would
    # serve any isotonic-calibrated primitive raw -- the same shape of failure
    # as compiling with one tokenizer and running tensors built by another.
    isotonic = None
    if temperature_path:
        sibling = pathlib.Path(
            str(temperature_path).replace("-temperatures.json", "-isotonic.json")
        )
        if sibling != pathlib.Path(temperature_path) and sibling.exists():
            from .calibration.isotonic import IsotonicCalibrator

            isotonic = IsotonicCalibrator.load(sibling)
    return Engine(
        impl,
        compiler=compiler,
        scaler=scaler,
        isotonic=isotonic,
        config=EngineConfig(domain=domain),
    )


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


#: Every split any command derives from the synthetic generator, in one place.
#:
#: Offsets rather than derived seeds, spaced 1,000 apart, so that no two runs
#: within 1,000 seeds of each other share a split. Two of these were 500 and
#: 900 apart, which made that guarantee false for `--seed 500`: a conformal
#: split of the run at seed 0 was the *training* split of the run at seed 500.
#: Nothing had gone wrong yet, and nothing would have announced it when it
#: did -- a conformal threshold fitted on another run's training data reports
#: a coverage number that is simply too good.
#:
#: `train` is 0 so that `--seed N` still means "the model initialised and
#: trained at N", which is what every committed report's header says.
SPLIT_SEED_OFFSETS = {
    "train": 0,
    "eval": 1_000,
    "calibration": 2_000,
    "conformal_fit": 3_000,
    "conformal_test": 4_000,
}


def training_splits(
    *, n: int, calibration_n: int, eval_n: int, seed: int, noise: float
) -> tuple[list, list, list]:
    """The train, calibration and eval splits for one run, in that order.

    Three splits, three seeds, none of them a reshuffle of another.

    **The calibration split is the one that was missing.** A temperature is
    fitted to close the gap between a model's confidence and its accuracy, and
    on data the model trained on that gap is the *memorised* one -- so the fit
    systematically under-corrects, by however much this particular draw
    overfit. It was fitted on the training split until the first four-seed
    sweep at 8,000 cases made the cost visible: temperature scaling *raised*
    ECE on two seeds of four, on one of them from 0.0431 to 0.0677 and past
    the gate, while lowering it on the other two. A calibration step that
    makes calibration worse on half its draws is not a calibration step.

    Separate from `cmd_train` so the property can be tested without training:
    what has to hold is that no split is the same data as another, and that is
    a fact about seeds, not about models. See `docs/decisions.md`, "The
    temperature was fitted on the split the model trained on".
    """
    from .evals import synthetic_outcome_cases

    return (
        synthetic_outcome_cases(n=n, seed=seed + SPLIT_SEED_OFFSETS["train"], noise=noise),
        synthetic_outcome_cases(
            n=calibration_n, seed=seed + SPLIT_SEED_OFFSETS["calibration"], noise=noise
        ),
        synthetic_outcome_cases(n=eval_n, seed=seed + SPLIT_SEED_OFFSETS["eval"], noise=noise),
    )


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
        render_json,
        render_markdown,
        run_calibration_suite,
    )
    from .schema import OptionScoring
    from .training import TrainingConfig
    from .training import train as run_training

    tokenizer = None
    if args.tokenizer == "hashing":
        from .backends.tokenizer import HashingTokenizer

        tokenizer = HashingTokenizer()
    backend = TorchReadoutBackend(
        tokenizer=tokenizer,
        config=ReadoutConfig(
            d_model=args.d_model,
            n_layers=args.layers,
            match_normalize=args.match_normalize,
            match_residual=args.match_residual,
        ),
        seed=args.seed,
    )
    compiler = backend.make_compiler(option_scoring=OptionScoring(args.option_scoring))

    train_cases, calibration_cases, eval_cases = training_splits(
        n=args.n,
        calibration_n=args.calibration_n,
        eval_n=args.eval_n,
        seed=args.seed,
        noise=args.noise,
    )

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
            validation_fraction=args.validation_fraction,
        ),
        compiler=compiler,
    )
    engine = Engine(backend, compiler=compiler)
    before, _ = run_calibration_suite(
        engine, eval_cases, suite="calibration/uncalibrated", floor_trials=args.floor_trials
    )

    # Fit on a split the model has never seen and the gates never read. Both
    # halves matter and only the second was observed before: fitting and
    # reporting on the same data is how a calibration number stops meaning
    # anything, and fitting on the training split is how it stops being a
    # calibration.
    scaler, isotonic = _fit_calibration(engine, calibration_cases)
    calibrated = Engine(backend, compiler=compiler, scaler=scaler, isotonic=isotonic)
    after, slices = run_calibration_suite(
        calibrated,
        eval_cases,
        suite="calibration/temperature-scaled",
        floor_trials=args.floor_trials,
    )
    # The same calibrated serving path, with int8 weights. `check_gates` has
    # always had a slot for this and nothing ever filled it, so
    # `quantization_ece_delta` was a gate over a run that did not exist. The
    # quantized twin is built from the trained weights and read through the
    # same temperature, because the question is whether the numerics move
    # calibration, not whether they move it enough to need refitting.
    quantized = None
    if not args.no_quantization_gate:
        twin = backend.quantized()
        quantized, _ = run_calibration_suite(
            Engine(twin, compiler=compiler, scaler=scaler, isotonic=isotonic),
            eval_cases,
            suite="calibration/int8",
            floor_trials=args.floor_trials,
        )
    gates = check_gates(after, quantized=quantized, slices=slices)
    # The int8 run is published as a row of its own, not folded into a delta.
    # A delta says how far two numbers are apart and hides which one is which;
    # the row says what the quantized path actually scores, which is the thing
    # a reader deciding whether to deploy it needs.
    suites = [before, after] + ([quantized] if quantized is not None else [])
    markdown = _training_section(report, args) + render_markdown(suites, gates, slices)

    print(markdown)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        # Sidecars are named after the report, not fixed: two runs writing into
        # one directory (an ablation, say) must not clobber each other's
        # temperatures and loss curves.
        # A machine-readable sibling, the same as `trigon eval` writes. Without
        # it a training run can only be read by a person, and anything that
        # compares runs -- scripts/seed_sweep.py, a CI trend -- has to parse
        # markdown.
        out.with_suffix(".json").write_text(render_json(suites, gates, slices))
        stem = out.with_suffix("")
        temperatures = pathlib.Path(f"{stem}-temperatures.json")
        training = pathlib.Path(f"{stem}-training.json")
        scaler.save(temperatures)
        # Beside the temperatures, not inside them: they are different objects
        # with different fit requirements, and a reader checking whether a
        # primitive was calibrated at all should see which tool was used.
        if isotonic.knots:
            isotonic.save(pathlib.Path(f"{stem}-isotonic.json"))
        training.write_text(_json.dumps(report.to_dict(), indent=2))
        written = [out.name, out.with_suffix(".json").name, temperatures.name, training.name]
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
        f"--floor-trials {args.floor_trials} --calibration-n {args.calibration_n} "
        f"--option-scoring {args.option_scoring}"
        + (" --match-normalize" if args.match_normalize else "")
        + ("" if args.match_residual else " --no-match-residual")
        + (
            ""
            if args.validation_fraction == 0.1
            else f" --validation-fraction {args.validation_fraction}"
        )
        + (f" --tokenizer {args.tokenizer}" if args.tokenizer != "bpe" else "")
        # Omitting it changes which gates the report carries, so a header
        # without it describes a different run from the one printed below.
        + (" --no-quantization-gate" if args.no_quantization_gate else "")
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


def _scaled(logits: list[float], temperature: float) -> list[float]:
    """Softmax of ``logits / temperature``, without importing a tensor library."""
    import math

    top = max(x / temperature for x in logits)
    exponentiated = [math.exp(x / temperature - top) for x in logits]
    total = sum(exponentiated)
    return [x / total for x in exponentiated]


#: How sure the check has to be that a temperature *helps* before applying it.
#: A paired bootstrap at this level: the fit is discarded unless it lowers ECE
#: in at least this share of resamples of the check split.
#:
#: The burden of proof sits on accepting, and that direction is a measurement
#: rather than a preference -- `scripts/decline_rule.py` scores three rules
#: against heads whose true calibration is known. See `docs/decisions.md`,
#: "A temperature is a proposal, not a result".
ACCEPT_CONFIDENCE = 0.95


def _harm_is_real(unscaled, rescaled, labels, *, resamples: int = 200, seed: int = 4919) -> bool:
    """Does this temperature raise ECE by more than sampling noise?

    **Not used by the pipeline.** Retained as the rejected alternative that
    `scripts/decline_rule.py` scores against, so the comparison that rejected
    it stays runnable. `_help_is_real` is the rule that ships.

    The reasoning below is preserved because it was wrong in an instructive
    way: it argues from two real data points for putting the burden of proof
    on *declining*, and seven constructed heads with known calibration say the
    opposite. This rule declines 6 times in 40 on an already-calibrated head
    where the shipped one declines 40 times in 40, and its worst-case ECE on
    that shape is 0.0460 against 0.0188.

    A paired bootstrap over the check split. The two ECEs are computed on the
    *same* points, so the quantity with a meaningful spread is their
    difference; resampling the points together preserves that pairing and
    asks, directly, how often the scaled version is worse.

    **This exists because the first version of the decision was a bare
    comparison** -- decline if `after >= before` -- which is a threshold on a
    noisy estimate with no noise floor under it, the one error this project
    refuses everywhere else. It cost exactly what that error costs: on seed 0
    of the 8,000-case sweep it declined a temperature that was genuinely
    helping, and that seed's ECE went from 0.0128 to 0.0219 while seed 1's
    real problem was fixed. A rule that is right about a large effect and
    random about a small one has to be told which it is looking at.

    The asymmetry is deliberate. A false decline costs a temperature that
    would have helped a little; a false accept costs a head served through a
    scalar that makes it much worse -- measured, 0.0251 against 0.0928 on one
    head of one seed. So the burden of proof is on declining, and a tie keeps
    the fit.
    """
    import random

    from .calibration.metrics import report as calibration_report

    rng = random.Random(seed)
    n = len(labels)
    if n < 2:
        return False
    worse = 0
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        pick_labels = [labels[i] for i in picks]
        if len(set(pick_labels)) < 2:
            # A resample with one label has no calibration to speak of;
            # counting it either way would be noise about noise.
            continue
        a = calibration_report([unscaled[i] for i in picks], pick_labels, simulate_floor=False).ece
        b = calibration_report([rescaled[i] for i in picks], pick_labels, simulate_floor=False).ece
        worse += b > a
    return worse >= ACCEPT_CONFIDENCE * resamples


def _calibration_error(probs, labels) -> float:
    """The statistic the calibrator selection minimises: the worse estimator.

    ECE bins confidence at equal *width*; adaptive ECE bins at equal *mass*.
    They disagree, and not by a little: on one seed's Noul head they read
    0.0251 and 0.1625 on the same 6,000 answers, because a binary head's
    confidences cluster tightly and equal-width bins average that cluster into
    one number while equal-mass bins resolve it.

    Both are release gates, so a run must pass both, and the selection used to
    score candidates on the plain one alone. That is optimising the estimator
    that cannot see the problem: it declined every calibrator for that head at
    a plain ECE of 0.0184 while the adaptive gate failed the run at 0.0604.
    Taking the worse of the two makes what the selection optimises the same
    thing the gates measure.
    """
    from .calibration.metrics import report as calibration_report

    measured = calibration_report(probs, labels, simulate_floor=False)
    return max(measured.ece, measured.adaptive_ece)


def _help_is_real(unscaled, rescaled, labels, *, resamples: int = 200, seed: int = 4919) -> bool:
    """Does this temperature *lower* ECE by more than sampling noise?

    The rule that ships: a fit is applied only where it demonstrably helps,
    and anything short of that serves unscaled.

    Which way the burden of proof points is an empirical question, and
    `scripts/decline_rule.py` answers it against heads whose true calibration
    is known. Worst-case ECE on a third draw none of the rules ever saw,
    40 trials per shape:

    ====================== ======= ========= ======== ============
    head                   bare    bootstrap strict   never scale
    ====================== ======= ========= ======== ============
    already calibrated     0.0290  0.0460    0.0188   0.0188
    slightly overconfident 0.0460  0.0382    0.0460   0.0460
    clearly overconfident  0.0528  0.0528    0.0528   0.2145
    clearly underconfident 0.0463  0.0463    0.0463   0.2173
    spread, calibrated     0.0343  0.0466    0.0208   0.0208
    spread, tilted         0.0708  0.0708    0.0624   0.0624
    spread, tilted hard    0.1246  0.1246    0.1218   0.1218
    ====================== ======= ========= ======== ============

    This rule matches or beats every alternative on six of seven shapes. What
    it is doing is visible in the decline counts rather than the ECEs: it
    declines 40 of 40 on every shape a temperature cannot fix, and 0 of 40 on
    the two where scaling is the difference between 0.05 and 0.21. It behaves
    like "never scale" where scaling is useless and like "always scale" where
    it is essential, which is the rule one would write by hand knowing the
    answers in advance.

    It loses on one shape -- a head overconfident by three points, where
    scaling helps a little and this refuses it, 0.0460 against the bootstrap's
    0.0382. That is the price of the direction, paid where the stake is
    smallest.
    """
    import random

    rng = random.Random(seed)
    n = len(labels)
    if n < 2:
        return False
    better = 0
    counted = 0
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        pick_labels = [labels[i] for i in picks]
        if len(set(pick_labels)) < 2:
            continue
        a = _calibration_error([unscaled[i] for i in picks], pick_labels)
        b = _calibration_error([rescaled[i] for i in picks], pick_labels)
        better += b < a
        counted += 1
    return counted > 0 and better >= ACCEPT_CONFIDENCE * counted


def _fit_calibration(engine, cases):
    """Fit both calibrators per primitive and apply whichever demonstrably helps.

    **A calibrator is a proposal, not a result**, and there are two proposals.
    A temperature is fitted by minimising NLL, and NLL is not ECE. It is also
    one-parameter: it sharpens or flattens everywhere at once, so a *tilted*
    head -- overconfident where it is confident, underconfident where it is
    not -- has no correct temperature and the fitter returns its best one
    anyway. An isotonic map fits any monotone shape and handles exactly that
    case, and overfits a head that was already calibrated.

    Neither wins everywhere, so neither is chosen in advance. Worst-case ECE
    against heads whose true calibration is known, at 4,000 answers
    (`scripts/calibrator_choice.py`):

    ====================== ======= ============ =========
    head                   none    temperature  isotonic
    ====================== ======= ============ =========
    already calibrated     0.0182  0.0182       0.0298
    clearly overconfident  0.2135  0.0247       0.0294
    tilted                 0.0607  0.0607       0.0286
    tilted hard            0.1232  0.1232       0.0227
    ====================== ======= ============ =========

    So: fit both on half the calibration split, score both on the half neither
    was fitted on, and apply the one that lowers ECE by more than sampling
    noise -- or neither. The check is held out because any calibrator improves
    the split it was fitted on, and scoring it there would accept every fit by
    construction.

    Which way the burden of proof points is itself measured rather than
    assumed; see `docs/decisions.md`, "A temperature is a proposal, not a
    result".
    """
    import math
    import warnings

    from .calibration.isotonic import MIN_ISOTONIC_SAMPLES, IsotonicCalibrator
    from .calibration.temperature import CalibrationWarning, TemperatureScaler
    from .evals import run_cases

    scaler = TemperatureScaler()
    isotonic = IsotonicCalibrator()
    rows: dict[str, list[tuple[list[float], int]]] = {}
    for outcome in run_cases(engine, cases):
        for question in outcome.questions.values():
            if question.expected is None or question.expected.hard_label is None:
                continue
            logits = [math.log(max(p, 1e-12)) for p in question.probabilities]
            rows.setdefault(question.primitive, []).append((logits, question.expected.hard_label))

    chosen: dict[str, str] = {}
    # Recorded rather than ignored. A fit pinned at a bound carries
    # information the verdict line does not -- that the head's logits may
    # carry no signal and the "improvement" is a flattening toward uniform --
    # and `CLAUDE.md` requires a degenerate fit to warn rather than return a
    # quiet number. Suppressing it here silently applied a T=20 fit on the
    # lexical floor. They are re-raised after the loop so the message names
    # the primitive and says whether the fit was kept.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", CalibrationWarning)
        for primitive, data in sorted(rows.items()):
            cut = max(1, len(data) // 2)
            fit_rows, check_rows = data[:cut], data[cut:]
            if not check_rows:
                # Nothing to check on. Serve unscaled rather than apply an
                # unverified calibrator: the failure being prevented is
                # precisely a fit nobody checked.
                chosen[primitive] = "none -- too few answers to check a fit on"
                continue

            labels = [y for _, y in check_rows]
            unscaled = [_scaled(x, 1.0) for x, _ in check_rows]
            baseline = _calibration_error(unscaled, labels)

            # Candidate 1: a temperature.
            if primitive == "noul":
                value = scaler.fit_binary(
                    [r[0][1] - r[0][0] for r in fit_rows], [y for _, y in fit_rows]
                )
            else:
                value = scaler.fit(primitive, [x for x, _ in fit_rows], [y for _, y in fit_rows])
            warmed = [_scaled(x, value) for x, _ in check_rows]
            warm_ece = _calibration_error(warmed, labels)
            warm_ok = _help_is_real(unscaled, warmed, labels)

            # Candidate 2: an isotonic map, where there is enough data for one.
            # `fit` refuses below its minimum rather than memorising the split.
            iso_ok, iso_ece, mapped = False, float("nan"), None
            if len(fit_rows) >= MIN_ISOTONIC_SAMPLES:
                candidate = IsotonicCalibrator()
                fit_probs = [_scaled(x, 1.0) for x, _ in fit_rows]
                if primitive == "noul":
                    # A Noul is calibrated on P(yes) against whether yes
                    # happened -- the standard binary calibration, and the
                    # quantity the engine actually maps at serving time.
                    #
                    # It was fitted on max(p) against correctness, which lives
                    # in [0.5, 1], and applied to P(yes), which lives in
                    # [0, 1]. Every answer below even odds fell off the left
                    # end of the fitted range and took the leftmost knot. That
                    # is the same shape of defect as compiling with one
                    # tokenizer and running tensors built by another, and it
                    # cost a factor of thirteen: this head's ECE read 0.3313
                    # against 0.0251 unscaled.
                    candidate.fit(
                        primitive,
                        [p[1] for p in fit_probs],
                        [int(y == 1) for _, y in fit_rows],
                    )
                else:
                    candidate.fit(
                        primitive,
                        [max(p) for p in fit_probs],
                        [
                            int(max(range(len(p)), key=lambda i: p[i]) == y)
                            for p, (_, y) in zip(fit_probs, fit_rows, strict=True)
                        ],
                    )
                mapped = (
                    [
                        [
                            1.0 - candidate.confidence(primitive, p[1]),
                            candidate.confidence(primitive, p[1]),
                        ]
                        for p in unscaled
                    ]
                    if primitive == "noul"
                    else [candidate.apply(primitive, p) for p in unscaled]
                )
                iso_ece = _calibration_error(mapped, labels)
                iso_ok = _help_is_real(unscaled, mapped, labels)

            # Prefer whichever helps more, among those that demonstrably help.
            if iso_ok and (not warm_ok or iso_ece < warm_ece):
                isotonic.knots[primitive] = candidate.knots[primitive]
                isotonic.fitted_on[primitive] = candidate.fitted_on[primitive]
                scaler.primitive.pop(primitive, None)
                scaler.fitted_on.pop(primitive, None)
                chosen[primitive] = f"isotonic ({baseline:.4f} -> {iso_ece:.4f})"
            elif warm_ok:
                chosen[primitive] = f"temperature T={value:.4f} ({baseline:.4f} -> {warm_ece:.4f})"
            else:
                # `fitted_on` is what a reader checks to see whether a
                # primitive was calibrated at all, so a declined fit must not
                # leave a count behind claiming it was.
                scaler.primitive.pop(primitive, None)
                scaler.fitted_on.pop(primitive, None)
                best = warm_ece if math.isnan(iso_ece) else min(warm_ece, iso_ece)
                chosen[primitive] = f"none -- unscaled is {baseline:.4f}, the best fit {best:.4f}"

    for primitive, verdict in sorted(chosen.items()):
        print(f"  {primitive}: {verdict}", file=sys.stderr)
    for entry in caught:
        if issubclass(entry.category, CalibrationWarning):
            applied = [p for p, v in chosen.items() if not v.startswith("none")]
            warnings.warn(
                f"{entry.message} (primitives calibrated: {', '.join(sorted(applied)) or 'none'})",
                CalibrationWarning,
                stacklevel=2,
            )
    return scaler, isotonic


def _fit_domain_temperatures(engine, cases, domain: str):
    """Per-domain temperatures, the one path the calibrator selection skips.

    `TemperatureScaler` carries a per-domain override and `IsotonicCalibrator`
    has no dimension for one, so a domain fit cannot choose between them. It
    fits temperatures unconditionally, which is what this command did for
    every fit before the selection existed.
    """
    import math

    from .calibration.isotonic import IsotonicCalibrator
    from .calibration.temperature import TemperatureScaler
    from .evals import run_cases

    scaler = TemperatureScaler()
    rows: dict[str, list[tuple[list[float], int]]] = {}
    for outcome in run_cases(engine, cases):
        for question in outcome.questions.values():
            if question.expected is None or question.expected.hard_label is None:
                continue
            # Recover logits from probabilities: temperature scaling is
            # invariant to an additive constant, so log p is sufficient.
            logits = [math.log(max(p, 1e-12)) for p in question.probabilities]
            rows.setdefault(question.primitive, []).append((logits, question.expected.hard_label))

    for primitive, data in sorted(rows.items()):
        if primitive == "noul":
            value = scaler.fit_binary(
                [r[0][1] - r[0][0] for r in data], [y for _, y in data], domain
            )
        else:
            value = scaler.fit(primitive, [x for x, _ in data], [y for _, y in data], domain)
        print(f"  {primitive}: T={value:.4f} on {len(data)} examples", file=sys.stderr)
    return scaler, IsotonicCalibrator()


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the post-hoc calibration layer on held-out labelled data.

    This is the tool the build plan's biggest risk depends on: outcome-grounded
    calibration fitted on public and synthetic data may not transfer to a
    user's domain, and the mitigation is that they can fit a conformal wrapper
    on a few hundred of their own labels. A mitigation that exists only in a
    design document is not a mitigation, so it ships as a command.
    """

    from .calibration.conformal import ConformalMethod, fit_conformal
    from .evals import synthetic_outcome_cases
    from .limits import CONFORMAL_COVERAGE_SIGMAS, conformal_coverage_floor

    engine = _engine(args.backend, args.domain, weights=args.weights)
    # The same seed offset `trigon train` reserves for calibration, so that
    # `trigon fit --weights` on a checkpoint from `trigon train --seed N` fits
    # on data that model has never seen. It used `args.seed` unshifted, which
    # is exactly the training split -- see `docs/decisions.md`, "The
    # temperature was fitted on the split the model trained on". With a
    # `--weights` checkpoint in play this command is the user-facing half of
    # that same defect.
    cases = synthetic_outcome_cases(
        n=args.n, seed=args.seed + SPLIT_SEED_OFFSETS["calibration"], noise=args.noise
    )

    # The same selection `trigon train` runs, rather than a second opinion
    # about how to calibrate. This command exists so a calibration layer can
    # be refitted without retraining -- minutes rather than twenty of them --
    # and a refit that used a different rule from the training run would be
    # producing a differently calibrated deployment under the same weights.
    if args.domain:
        # Per-domain temperatures are a `TemperatureScaler` feature that the
        # isotonic map has no dimension for. Rather than silently drop the
        # domain or silently drop the selection, fit only temperatures and
        # say so.
        print(
            f"  fitting temperatures only: --domain {args.domain!r} has no isotonic "
            "equivalent, so the calibrator selection is skipped",
            file=sys.stderr,
        )
        scaler, isotonic = _fit_domain_temperatures(engine, cases, args.domain)
    else:
        scaler, isotonic = _fit_calibration(engine, cases)

    scaler.save(args.out)
    if isotonic.knots:
        # Beside the temperatures, under the name `_engine` looks for, so
        # `trigon ask --temperatures` and `trigon serve` pick it up without a
        # second flag. A calibrator written where nothing reads it is the same
        # as not having fitted one.
        sibling = pathlib.Path(str(args.out).replace("-temperatures.json", "-isotonic.json"))
        if sibling == pathlib.Path(args.out):
            sibling = pathlib.Path(args.out).with_suffix(".isotonic.json")
        isotonic.save(sibling)
        print(f"  isotonic map written to {sibling}", file=sys.stderr)
    print(json.dumps(scaler.to_dict(), indent=2, sort_keys=True))

    if args.conformal_out:
        # Fitted on a split the temperatures never saw, and reported on a
        # third: a coverage number measured where the threshold was fitted is
        # not a coverage number.
        calib = synthetic_outcome_cases(
            n=args.n, seed=args.seed + SPLIT_SEED_OFFSETS["conformal_fit"], noise=args.noise
        )
        test = synthetic_outcome_cases(
            n=args.n, seed=args.seed + SPLIT_SEED_OFFSETS["conformal_test"], noise=args.noise
        )
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

        # Gate it. Coverage is the only thing a conformal wrapper promises,
        # and it was printed rather than checked -- a number a reader was
        # trusted to notice. The floor is derived from the target and the
        # held-out count, not configured, because empirical coverage on n
        # points fluctuates even for a perfect predictor and a fixed
        # tolerance is either vacuous at small n or spuriously red at large n.
        floor = conformal_coverage_floor(predictor.target_coverage, len(held))
        print(
            f"{'PASS' if achieved >= floor else 'FAIL'} conformal_coverage: "
            f"{achieved:.4f} (floor {floor:.4f} = target "
            f"{predictor.target_coverage:.2f} less {CONFORMAL_COVERAGE_SIGMAS:g} sigma "
            f"of sampling noise at n={len(held)})",
            file=sys.stderr,
        )
        if achieved < floor:
            print(
                "\nThe profile was written anyway, because a wrapper that "
                "under-covers is\nevidence about the model and you will want to "
                "look at it. Do not serve it:\nit makes a guarantee it does not "
                "keep, which is worse than making none.",
                file=sys.stderr,
            )
            return 1
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
        "--calibration-n",
        type=int,
        default=1000,
        help="cases held out to fit the temperature; seen by neither training nor the gates",
    )
    tr.add_argument(
        "--no-quantization-gate",
        action="store_true",
        help="skip the int8 twin; it is a second full eval pass over the same cases",
    )
    tr.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="share of cases held out to pick which epoch to keep; 0 keeps the last",
    )
    tr.add_argument(
        "--tokenizer",
        default="bpe",
        choices=["bpe", "hashing"],
        help="ablation: the shipped BPE vocabulary, or the hashing fallback",
    )
    tr.add_argument(
        "--match-normalize",
        action="store_true",
        help="dot-product head: cosine similarity with a learnable temperature",
    )
    tr.add_argument(
        "--match-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dot-product head: add each option's input embedding to its key (default on)",
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
