#!/usr/bin/env python
"""Refit a published LM-score bundle's calibrators, and say per source what they do.

    python scripts/refit_calibration.py --bundle runs/hf/qwen3-4b-lms-agents-s0 \\
        --exclude-texts bench-texts.jsonl --device mps --out runs/refit-4b-agents

The bundle's `mix.json` records the arguments it was trained with, and
`train_lm_score.split` is a function of those alone, so this rebuilds the
exact calibration split the shipped calibrators were fitted on, answers it
once, and keeps the answers (`rows.jsonl`). The fit is then
`trigon.cli._choose_calibrator`, the rule that ships, over one answer per case
and primitive as training keeps them, so a refit here is the calibrator the
trainer would now write.

The report breaks the calibration answers down by source, because a pooled
ECE cancels: one source overconfident and another underconfident read as
calibrated together, and the rule declines a calibrator that each would want.

``--exclude-texts`` must be the benchmark's texts (`decision_bench.py
--dump-texts`), as it was in training; `mix.json` records only that a file
was given.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from train_lm_score import split  # noqa: E402
from train_mix import MIX  # noqa: E402


def run_args(argv: list[str], exclude_texts: pathlib.Path | None) -> argparse.Namespace:
    """The training arguments that decide the split, read back from `mix.json`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--calibration-per-corpus", type=int, default=600)
    parser.add_argument("--reshape-max", type=int, default=77)
    parser.add_argument("--noul-from-choice", type=float, default=0.0)
    parser.add_argument("--exclude-texts", default=None)
    args, _ = parser.parse_known_args(argv)
    if args.exclude_texts is not None and exclude_texts is None:
        raise SystemExit("the run excluded a benchmark's texts: pass the same --exclude-texts")
    args.exclude_texts = exclude_texts if args.exclude_texts is not None else None
    return args


def source(case) -> str:
    if "noul-from-choice" in case.tags:
        return "derived-noul"
    return case.domain or case.case_id.split("/", 1)[0]


def answer(bundle: pathlib.Path, cases: list, device: str) -> list[dict]:
    from trigon.backends.lm_score import LMScoreBackend
    from trigon.engine import Engine
    from trigon.evals import run_cases

    mix = json.loads((bundle / "mix.json").read_text())
    backend = LMScoreBackend(
        mix["base"], revision=mix["revision"], device=device, adapter=str(bundle / "adapter.pt")
    )
    rows = []
    engine = Engine(backend)
    for i, case in enumerate(cases):
        if i % 200 == 0:
            print(f"answered {i}/{len(cases)}", file=sys.stderr, flush=True)
        [outcome] = run_cases(engine, [case])
        for question in outcome.questions.values():
            if question.expected is None or question.expected.hard_label is None:
                continue
            rows.append(
                {
                    "case": outcome.case.case_id,
                    "source": source(outcome.case),
                    "primitive": question.primitive,
                    "logits": [math.log(max(p, 1e-12)) for p in question.probabilities],
                    "label": question.expected.hard_label,
                }
            )
    return rows


def ece(rows: list[dict], temperature: float = 1.0) -> tuple[float, float, float]:
    """(ECE, accuracy, mean confidence) of the top answer, ten equal-width bins."""
    from trigon.cli import _scaled

    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        p = _scaled(row["logits"], temperature)
        top = max(range(len(p)), key=p.__getitem__)
        bins[min(9, int(p[top] * 10))].append((p[top], int(top == row["label"])))
    n = sum(len(b) for b in bins.values())
    gap = sum(abs(sum(c for c, _ in b) - sum(y for _, y in b)) for b in bins.values()) / n
    acc = sum(y for b in bins.values() for _, y in b) / n
    conf = sum(c for b in bins.values() for c, _ in b) / n
    return gap, acc, conf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--exclude-texts", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    from trigon.calibration.isotonic import IsotonicCalibrator
    from trigon.calibration.temperature import TemperatureScaler
    from trigon.cli import _choose_calibrator

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "rows.jsonl"
    if cache.exists():
        rows = [json.loads(line) for line in cache.open()]
    else:
        mix_json = json.loads((args.bundle / "mix.json").read_text())
        run = run_args(mix_json["argv"], args.exclude_texts)
        mix = dict(MIX)
        for item in run.extra:
            name, _, count = item.partition("=")
            mix[name] = int(count)
        _, calibration, _ = split(mix, run)
        expected = mix_json["calibrated_on"]["cases"]
        if len(calibration) != expected:
            raise SystemExit(
                f"rebuilt {len(calibration)} calibration cases, the run had {expected}"
            )
        rows = answer(args.bundle, calibration, args.device)
        cache.write_text("".join(json.dumps(r) + "\n" for r in rows))

    if any("case" not in row for row in rows):
        raise SystemExit(f"{cache} predates case ids; delete it and answer again")
    # One answer per case and primitive, the way `trigon.cli._calibration_rows`
    # chooses it, so a refit here is the calibrator training would now write.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["primitive"])].append(row)
    by_primitive: dict[str, list[dict]] = defaultdict(list)
    for (case, primitive), group in grouped.items():
        by_primitive[primitive].append(random.Random(f"{case}:{primitive}").choice(group))

    scaler, isotonic = TemperatureScaler(), IsotonicCalibrator()
    lines = [
        "| primitive | source | n | accuracy | confidence | ECE |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for primitive, prows in sorted(by_primitive.items()):
        verdict = _choose_calibrator(
            primitive, [(r["logits"], r["label"]) for r in prows], scaler, isotonic
        )
        print(f"{primitive}: {verdict}", file=sys.stderr)
        sources: dict[str, list[dict]] = defaultdict(list)
        for row in prows:
            sources[row["source"]].append(row)
        for name, srows in sorted(sources.items()) + [("(pooled)", prows)]:
            gap, acc, conf = ece(srows)
            lines.append(
                f"| {primitive} | {name} | {len(srows)} | {acc:.3f} | {conf:.3f} | {gap:.3f} |"
            )
    scaler.save(args.out / "temperatures.json")
    # A map from an earlier refit into the same directory must not outlive a
    # verdict that declined one.
    (args.out / "isotonic.json").unlink(missing_ok=True)
    if isotonic.knots:
        isotonic.save(args.out / "isotonic.json")
    (args.out / "sources.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
