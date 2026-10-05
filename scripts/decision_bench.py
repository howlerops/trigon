#!/usr/bin/env python
"""Score a system on a public decision benchmark, beside the incumbent's own predictions.

    python scripts/decision_bench.py --cases <bench>/cases --bundle runs/broad --device mps \\
        --reference <bench>/predictions --out reports/decision-bench/broad

The benchmark (Leanmcp, 2026, on the Hugging Face Hub; cases under their
sources' licences, predictions and code MIT) publishes every request it sent
-- in the incumbent's request shape -- with the gold answer, and the full
probability vector each system returned.
That makes the comparison exact: the same 8,016 requests, scored the same way,
against the incumbent's real answers rather than a description of them.

The cases are sent through trigon's compatibility route, as a client would send
them. Conventions are the benchmark's: a Noul is right at 0.5, a Choice by argmax;
Brier is the multiclass sum; ECE uses ten bins over the top-label probability.
SST-5 ships without its text (its source states no licence) and is skipped, as
are the image slices: trigon reads text only.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.server.compat_path import COMPAT_PATH  # noqa: E402

#: SST-5 ships without its text; the image slices need pixels trigon cannot read.
SKIP = {"sst5", "scienceqa_image", "vqa_rad"}


def _parse(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return json.loads(value)
    return value


def _ece(confidence: list[float], correct: list[int], bins: int = 10) -> float:
    total, n = 0.0, len(confidence)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidence) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            total += (
                len(idx)
                / n
                * abs(
                    sum(correct[i] for i in idx) / len(idx)
                    - sum(confidence[i] for i in idx) / len(idx)
                )
            )
    return total


def score_slice(rows: list[dict]) -> dict:
    """rows: {task_type, gold, probs: {label: p} or p_true}."""
    correct, confidence, brier = [], [], []
    for r in rows:
        if r["task_type"] == "noul":
            p = float(r["p_true"])
            gold = r["gold"] in (True, "True", "true", 1)
            correct.append(int((p >= 0.5) == gold))
            confidence.append(max(p, 1 - p))
            brier.append((p - gold) ** 2 + ((1 - p) - (1 - gold)) ** 2)
        else:
            probs = r["probs"]
            top = max(probs, key=probs.get)
            correct.append(int(top == r["gold"]))
            confidence.append(probs[top])
            brier.append(sum((p - (k == r["gold"])) ** 2 for k, p in probs.items()))
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(correct) / n,
        "brier": sum(brier) / n,
        "ece_10": _ece(confidence, correct),
        "mean_confidence": sum(confidence) / n,
    }


def _reference(path: pathlib.Path) -> dict[str, dict]:
    out = {}
    for line in path.open():
        p = json.loads(line)
        if p.get("error"):
            continue
        row = {"task_type": p["task_type"], "gold": p["gold"]}
        if p["task_type"] == "noul":
            row["p_true"] = p["p_true"]
        else:
            row["probs"] = p.get("probabilities") or p.get("probs") or {}
            if not row["probs"]:
                continue
        out[p["case_id"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=pathlib.Path, required=True)
    parser.add_argument(
        "--dump-texts",
        type=pathlib.Path,
        default=None,
        help="write every text the benchmark shows a model, one JSON string a line, and exit: "
        "what a training run must not contain (train_lm_score.py --exclude-texts)",
    )
    who = parser.add_mutually_exclusive_group()
    who.add_argument("--bundle", type=pathlib.Path, help="a train_mix.py run directory")
    who.add_argument("--lm", help="a causal LM's Hugging Face id, scored zero-shot (lm-score)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reference", type=pathlib.Path, default=None)
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="DIR=NAME",
        help="report a reference system under NAME instead of its directory's name",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="DIR",
        help="compare against these reference systems only (default: every one)",
    )
    parser.add_argument("--limit", type=int, default=0, help="cases per slice; 0 = all")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.dump_texts:
        texts = set()
        for path in sorted(args.cases.glob("*.jsonl")):
            for line in path.open():
                row = json.loads(line) if line.strip() else {}
                if row.get("state") is not None:
                    state = _parse(row["state"])
                    values = state.values() if isinstance(state, dict) else [state]
                    texts.update(str(v).strip() for v in values if str(v).strip())
        args.dump_texts.write_text("".join(json.dumps(t) + "\n" for t in sorted(texts)))
        print(f"{len(texts)} texts -> {args.dump_texts}", file=sys.stderr)
        return 0
    if not (args.bundle or args.lm) or args.out is None:
        parser.error("--out and one of --bundle or --lm are required")

    from fastapi.testclient import TestClient

    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    if args.lm:
        # A Hugging Face id (zero-shot), or a train_lm_score.py adapter.pt
        # whose run directory holds its calibrators.
        env = {"TRIGON_BACKEND": "lm-score", "TRIGON_WEIGHTS": args.lm}
        run = pathlib.Path(args.lm).parent if args.lm.endswith(".pt") else None
    else:
        env = {"TRIGON_BACKEND": "torch", "TRIGON_WEIGHTS": str(args.bundle / "adapter.pt")}
        run = args.bundle
    for name, variable in (
        ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
        ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
    ):
        if run is not None and (run / name).exists():
            env[variable] = str(run / name)
    if args.device:
        env["TRIGON_DEVICE"] = args.device
    client = TestClient(build_app(ServerConfig.from_env(env)))
    print("healthz", client.get("/healthz").json(), file=sys.stderr, flush=True)

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ours: dict[str, dict] = {}
    for path in sorted(args.cases.glob("*.jsonl")):
        slice_name = path.stem
        if slice_name in SKIP:
            continue
        cases = [json.loads(line) for line in path.open() if line.strip()]
        if args.limit:
            # A seeded sample, never a prefix: at least one slice's file is
            # ordered by label, and its first 60 cases are nearly all one
            # answer -- a prefix scored a yes-biased model 0.917 there.
            cases = random.Random(f"limit:{slice_name}").sample(cases, min(args.limit, len(cases)))
        started, failed = time.perf_counter(), 0
        for case in cases:
            cached = raw / f"{case['case_id']}.json"
            if cached.exists():
                answer = json.loads(cached.read_text())
            else:
                question = _parse(case["question"])
                body = {
                    "state": _parse(case["state"]),
                    "model": "trigon",
                    "questions": {case.get("question_key") or "decision": question},
                }
                response = client.post(f"/compat{COMPAT_PATH}", json=body)
                if response.status_code != 200:
                    failed += 1
                    continue
                answer = next(iter(response.json()["answers"].values()))
                cached.write_text(json.dumps(answer))
            row = {"task_type": case["task_type"], "gold": case["gold"]}
            if case["task_type"] == "noul":
                row["p_true"] = answer["noul"]
            else:
                row["probs"] = answer.get("probabilities") or {}
            ours[case["case_id"]] = row
        print(
            f"{slice_name}: {len(cases)} cases, {failed} failed, "
            f"{time.perf_counter() - started:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    systems = {"trigon": ours}
    rename = dict(item.split("=", 1) for item in args.rename)
    if args.reference:
        for directory in sorted(p for p in args.reference.iterdir() if p.is_dir()):
            file = directory / "predictions.jsonl"
            if file.exists() and (not args.only or directory.name in args.only):
                systems[rename.get(directory.name, directory.name)] = _reference(file)

    def slice_of(case_id: str) -> str:
        return case_id.rsplit("-", 2)[0]

    # Per slice, scored on the cases every system that answered the slice
    # answered, so no two systems are judged on different subsets of it. A
    # reference system may cover only some slices (one covers only the image
    # ones); the macro is taken over the slices every system covers.
    by_slice: dict[str, dict[str, dict]] = {}
    for name, rows in systems.items():
        for case_id, row in rows.items():
            by_slice.setdefault(slice_of(case_id), {}).setdefault(name, {})[case_id] = row
    results: dict[str, dict[str, dict]] = {}
    for slice_name, per in sorted(by_slice.items()):
        if "trigon" not in per:
            continue
        common = set.intersection(*(set(rows) for rows in per.values()))
        if common:
            results[slice_name] = {
                name: score_slice([rows[c] for c in sorted(common)]) for name, rows in per.items()
            }
    names = list(systems)

    # A system's macro is over the slices it answered; one that skipped some
    # of ours is marked, because its macro is over a different set.
    def macro_of(name: str) -> tuple[float, int]:
        covered = [s for s in results if name in results[s]]
        if not covered:
            return math.nan, 0
        return sum(results[s][name]["accuracy"] for s in covered) / len(covered), len(covered)

    macro = {name: macro_of(name)[0] for name in names}
    partial = {name for name in names if macro_of(name)[1] < len(results)}
    full = [s for s in results if all(n in results[s] for n in names if n not in partial)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps({"macro_over": full, "macro_accuracy": macro, "slices": results}, indent=2)
    )
    print("| Slice | n | " + " | ".join(f"{n} acc / ECE" for n in names) + " |")
    print("| --- | ---: |" + " --- |" * len(names))
    for slice_name, per in results.items():
        n = per["trigon"]["n"]
        cells = [
            f"{per[k]['accuracy']:.3f} / {per[k]['ece_10']:.3f}" if k in per else "—" for k in names
        ]
        print(f"| {slice_name} | {n} | " + " | ".join(cells) + " |")
    cells = [
        "—"
        if math.isnan(macro[k])
        else f"**{macro[k]:.3f}**" + (f" ({macro_of(k)[1]} slices only)" if k in partial else "")
        for k in names
    ]
    print(f"| **macro** ({len(results)} slices) | | " + " | ".join(cells) + " |")
    return 0 if not math.isnan(macro["trigon"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
