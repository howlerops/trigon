"""The lexical floor against the paired robustness benchmarks.

`CLAUDE.md`: run the floor against a benchmark before believing it. The
paired benchmarks (`trigon.evals.paired`) are built on a corpus's own schema
and state, so what the floor scores on them says which of their numbers a
model can get without reading anything -- and one of them it gets for free:
the floor scores a Choice by the overlap between each option's name and the
state and never reads the question, so no paraphrase of the question can
move it. Paraphrase agreement is therefore evidence only beside accuracy.

    python scripts/paired_floor.py --n 1000 --out reports/paired/floor.json

Banking77's test split and the verifiable synthetic generator, with the
evaluation pool of templates the benchmarks use in every training report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.backends.lexical import LexicalBackend  # noqa: E402
from trigon.engine import Engine  # noqa: E402
from trigon.evals import run_jaggedness, synthetic_outcome_cases  # noqa: E402
from trigon.evals.paired import paired_benchmarks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="bases per benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-banking77", action="store_true")
    args = parser.parse_args(argv)

    sources = {"synthetic": synthetic_outcome_cases(max(args.n, 200), seed=10_000 + args.seed)}
    if not args.skip_banking77:
        from trigon.evals.corpora import load

        sources["banking77"] = load("banking77", "test", purpose="eval")
    engine = Engine(LexicalBackend())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, base in sources.items():
        results = run_jaggedness(engine, paired_benchmarks(base, n=args.n, seed=args.seed))
        out[name] = {r.suite: dict(r.extra) for r in results}
        print(f"## {name} ({min(args.n, len(base))} bases)")
        for suite, extra in out[name].items():
            print(f"  {suite}")
            for metric, value in sorted(extra.items()):
                print(f"    {metric:24s} {value:.4f}")
    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
