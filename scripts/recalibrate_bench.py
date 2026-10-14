#!/usr/bin/env python
"""Score a decision-benchmark run's saved answers under a different calibrator.

    python scripts/recalibrate_bench.py --cases <bench>/cases --run runs/bench-4b \\
        --served runs/hf/qwen3-4b-lms-agents-s0 --candidate runs/refit-4b-agents

`decision_bench.py --out` keeps every answer it was served (`raw/`). Those
answers already carry the calibrator the run was served with, so this undoes
it -- a temperature is invertible on a Choice's probabilities, and the run's
Nouls must have been served raw -- and applies the candidate's instead. No
model call: what changes between the two columns is the calibrator alone,
which is the question a refit has to answer before it ships.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.calibration.isotonic import IsotonicCalibrator  # noqa: E402
from trigon.calibration.temperature import TemperatureScaler  # noqa: E402
from trigon.cli import _scaled  # noqa: E402

SKIP = {"scienceqa_image", "vqa_rad", "sst5"}


def _load(directory: pathlib.Path) -> tuple[TemperatureScaler, IsotonicCalibrator]:
    scaler = (
        TemperatureScaler.load(directory / "temperatures.json")
        if (directory / "temperatures.json").exists()
        else TemperatureScaler()
    )
    isotonic = (
        IsotonicCalibrator.load(directory / "isotonic.json")
        if (directory / "isotonic.json").exists()
        else IsotonicCalibrator()
    )
    return scaler, isotonic


def raw_probabilities(answer: dict, primitive: str, served: TemperatureScaler) -> list[float]:
    """The backend's own probabilities, the served temperature taken back out."""
    if primitive == "noul":
        if "noul" in served.primitive:
            raise SystemExit("the run's yes/no answers were served scaled; undo is not built")
        p = float(answer["noul"])
        return [1 - p, p]
    probs = [float(v) for v in answer["probabilities"].values()]
    t = served.primitive.get("choice", 1.0)
    return _scaled([math.log(max(p, 1e-12)) * t for p in probs], 1.0)


def calibrate(probs, primitive, scaler, isotonic) -> list[float]:
    if primitive in isotonic.knots:
        if primitive == "noul":
            p = isotonic.confidence(primitive, probs[1])
            return [1 - p, p]
        return isotonic.apply(primitive, probs)
    t = scaler.primitive.get(primitive, 1.0)
    return _scaled([math.log(max(p, 1e-12)) for p in probs], t)


def ece(rows: list[tuple[list[float], int]]) -> tuple[float, float, float]:
    bins: dict[int, list[tuple[float, int]]] = {}
    for probs, gold in rows:
        top = max(range(len(probs)), key=probs.__getitem__)
        bins.setdefault(min(9, int(probs[top] * 10)), []).append((probs[top], int(top == gold)))
    n = len(rows)
    gap = sum(abs(sum(c for c, _ in b) - sum(y for _, y in b)) for b in bins.values()) / n
    acc = sum(y for b in bins.values() for _, y in b) / n
    conf = sum(c for b in bins.values() for c, _ in b) / n
    return gap, acc, conf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, required=True)
    parser.add_argument("--run", type=pathlib.Path, required=True, help="decision_bench --out")
    parser.add_argument("--served", type=pathlib.Path, required=True, help="the run's calibrators")
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    args = parser.parse_args()

    served = _load(args.served)
    candidate = _load(args.candidate)
    print("| slice | n | accuracy | served conf | served ECE | candidate conf | candidate ECE |")
    print("|---|--:|--:|--:|--:|--:|--:|")
    means = {"served": [], "candidate": []}
    for path in sorted(args.cases.glob("*.jsonl")):
        if path.stem in SKIP:
            continue
        before, after = [], []
        for line in path.open():
            if not line.strip():
                continue
            case = json.loads(line)
            cached = args.run / "raw" / f"{case['case_id']}.json"
            if not cached.exists():
                continue
            answer = json.loads(cached.read_text())
            primitive = case["task_type"]
            if primitive == "noul":
                gold = int(
                    bool(case["gold"])
                    if not isinstance(case["gold"], str)
                    else case["gold"].lower() in ("true", "yes", "1")
                )
            else:
                question = case["question"]
                question = ast.literal_eval(question) if isinstance(question, str) else question
                gold = list(question["criteria"]).index(case["gold"])
            raw = raw_probabilities(answer, primitive, served[0])
            before.append((calibrate(raw, primitive, *served), gold))
            after.append((calibrate(raw, primitive, *candidate), gold))
        if not before:
            continue
        e0, acc, c0 = ece(before)
        e1, _, c1 = ece(after)
        means["served"].append(e0)
        means["candidate"].append(e1)
        cells = [path.stem, len(before), f"{acc:.3f}", f"{c0:.3f}", f"{e0:.3f}", f"{c1:.3f}"]
        print("| " + " | ".join(map(str, cells)) + f" | {e1:.3f} |")
    s, c = (sum(v) / len(v) for v in (means["served"], means["candidate"]))
    print(f"| **mean ECE** | | | | **{s:.3f}** | | **{c:.3f}** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
