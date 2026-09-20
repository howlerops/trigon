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


def _engine(backend: str, domain: str | None = None, temperature_path: str | None = None):
    from .calibration.temperature import TemperatureScaler
    from .engine import Engine, EngineConfig

    if backend == "lexical":
        from .backends.lexical import LexicalBackend

        impl = LexicalBackend()
        compiler = None
    elif backend == "torch":
        from .backends.torch_readout import TorchReadoutBackend

        impl = TorchReadoutBackend()
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
    engine = _engine(args.backend, args.domain, args.temperatures)
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
    uvicorn.run(build_app(config), host=args.host, port=args.port)
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    from .server.app import build_app

    print(json.dumps(build_app().openapi(), indent=2, sort_keys=True))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals import (
        check_gates,
        render_json,
        render_markdown,
        run_calibration_suite,
        run_jaggedness,
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

    if not results:
        raise SystemExit(f"unknown suite {args.suite!r}")

    markdown = render_markdown(results, gates, slices)
    print(markdown)
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        out.with_suffix(".json").write_text(render_json(results, gates, slices))
        print(f"\nwrote {out} and {out.with_suffix('.json')}", file=sys.stderr)

    # Non-zero exit on a failed gate, so CI can depend on this directly.
    return 0 if all(g.passed for g in gates) else 1


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit temperatures on the synthetic outcome set and save them."""
    import math

    from .calibration.temperature import TemperatureScaler
    from .evals import run_cases, synthetic_outcome_cases

    engine = _engine(args.backend, args.domain)
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
    return 0


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
    ask.set_defaults(func=cmd_ask)

    serve = sub.add_parser("serve", help="run the reference gateway")
    shared(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    spec = sub.add_parser("spec", help="print the OpenAPI spec")
    spec.set_defaults(func=cmd_spec)

    ev = sub.add_parser("eval", help="run an eval suite; exits non-zero on a failed gate")
    shared(ev)
    ev.add_argument("suite", choices=["calibration", "jaggedness", "all"])
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
    fit.set_defaults(func=cmd_fit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
