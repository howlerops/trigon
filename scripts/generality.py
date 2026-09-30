#!/usr/bin/env python
"""Run the generality suite against any system that speaks the contract.

    # a trained bundle, in process (adapter.pt + temperatures.json + isotonic.json)
    python scripts/generality.py --bundle releases/local/v2 --out reports/generality/v2

    # anything on the incumbent's wire, e.g. a compatible hosted service
    TRIGON_INCUMBENT_KEY=... python scripts/generality.py \\
        --compat https://api.example.com/api --model their-model --out reports/generality/them

Every raw answer is written under ``--out/raw/<task>/<i>.json`` before it is
scored, so a run that is interrupted -- or rate-limited -- resumes where it
stopped, and a number in the report can always be traced to the response it
came from. The cases are `trigon.evals.generality.build`, drawn from a fixed
seed, so two systems run on the same ``-n`` answered the same questions.

The report prints ECE only beside its simulated noise floor: at the suite's
default of 1,000 cases per task the floor is not negligible, and an ECE that
cannot be told apart from a calibrated model's is reported as exactly that.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.calibration.metrics import report  # noqa: E402
from trigon.evals.generality import DEFAULT_N, TASKS, build, selected_names, task  # noqa: E402
from trigon.evals.harness import Case  # noqa: E402
from trigon.server.compat import (  # noqa: E402
    COMPAT_PATH,
    compat_answer_to_native,
    to_compat_request,
)
from trigon.types import ChoiceQuestion, NoulQuestion  # noqa: E402


def _in_process(bundle: pathlib.Path, int8: bool = False):
    from fastapi.testclient import TestClient

    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    env = {
        "TRIGON_BACKEND": "torch",
        "TRIGON_WEIGHTS": str(bundle / "adapter.pt"),
        "TRIGON_INT8": "1" if int8 else "0",
    }
    for key, name in (
        ("TRIGON_TEMPERATURE_PATH", "temperatures.json"),
        ("TRIGON_ISOTONIC_PATH", "isotonic.json"),
    ):
        if (bundle / name).exists():
            env[key] = str(bundle / name)
    client = TestClient(build_app(ServerConfig.from_env(env)))
    health = client.get("/healthz").json()
    print(f"in process: {bundle} {health}", file=sys.stderr)

    def answer(case: Case) -> dict:
        response = client.post("/v1/decide", json=case.request.model_dump(exclude_none=True))
        response.raise_for_status()
        return response.json()

    return answer, health, 1


def _compat(url: str, model: str, key: str | None, concurrency: int):
    def post(target: str, body: dict | None) -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            target, data=json.dumps(body).encode() if body else None, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def answer(case: Case) -> dict:
        body = to_compat_request(case.request, model=model)
        for attempt in range(8):
            code, data = post(url.rstrip("/") + COMPAT_PATH + "?wait=30", body)
            if code == 429 or code >= 500:
                time.sleep(min(60, 2**attempt))
                continue
            if code != 200:
                raise RuntimeError(f"HTTP {code}: {json.dumps(data)[:300]}")
            # A service that answers asynchronously wraps the result in a job.
            result = data.get("result", data) if isinstance(data.get("result"), dict) else data
            if data.get("status") not in (None, "completed"):
                raise RuntimeError(f"job {data.get('id')} is {data.get('status')}")
            return {
                "answers": {
                    qid: compat_answer_to_native(a, case.request.questions.get(qid))
                    for qid, a in (result.get("answers") or {}).items()
                },
                "raw": data,
            }
        raise RuntimeError("rate limited on every retry")

    return answer, {"compat": url, "model": model}, concurrency


def _run(t, cases: list[Case], answer, workers: int, raw_dir: pathlib.Path) -> list[dict | None]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    def one(index: int) -> dict | None:
        path = raw_dir / f"{index:05d}.json"
        if path.exists():
            return json.loads(path.read_text())
        started = time.perf_counter()
        try:
            data = answer(cases[index])
        except Exception as error:  # noqa: BLE001 - a failed call is a result
            print(f"  {t.name}/{index}: {type(error).__name__}: {error}", file=sys.stderr)
            return None
        data["_seconds"] = round(time.perf_counter() - started, 4)
        path.write_text(json.dumps(data))
        return data

    with cf.ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, range(len(cases))))


def _score(cases: list[Case], results: list[dict | None]) -> dict:
    probs, labels, chance, seconds, failed = [], [], [], [], 0
    for case, result in zip(cases, results, strict=True):
        qid, question = next(iter(case.request.questions.items()))
        answer = (result or {}).get("answers", {}).get(qid)
        expected = case.expected.get(qid)
        if answer is None or expected is None or expected.hard_label is None:
            failed += 1
            continue
        if isinstance(question, ChoiceQuestion):
            got = answer.get("probabilities") or {}
            row = [float(got.get(o.name, 0.0)) for o in question.options]
            chance.append(1 / len(row))
        elif isinstance(question, NoulQuestion):
            p = float(answer["probability"])
            row = [1 - p, p]
            chance.append(0.5)
        else:
            failed += 1
            continue
        total = sum(row)
        if total <= 0:
            failed += 1
            continue
        probs.append([v / total for v in row])
        labels.append(expected.hard_label)
        seconds.append((result or {}).get("_seconds", 0.0))
    if not probs:
        return {"n": 0, "failed": failed}
    r = report(probs, labels)
    majority = max(labels.count(v) for v in set(labels)) / len(labels)
    seconds.sort()
    return {
        "n": r.n,
        "failed": failed,
        "accuracy": r.accuracy,
        "chance": sum(chance) / len(chance),
        "majority": majority,
        "ece": r.ece,
        "adaptive_ece": r.adaptive_ece,
        "ece_floor_p95": r.floor.p95 if r.floor else None,
        "miscalibration_detectable": r.distinguishable,
        "mean_confidence": r.mean_confidence,
        "nll": r.nll,
        "brier": r.brier,
        "p50_seconds": seconds[len(seconds) // 2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--bundle", type=pathlib.Path, help="adapter.pt + calibrators, in process")
    who.add_argument("--compat", help="base URL of a service on the incumbent's wire")
    parser.add_argument("--model", default=None, help="model name, with --compat")
    parser.add_argument("--int8", action="store_true", help="with --bundle: int8 CPU twin")
    parser.add_argument("--threads", type=int, default=0, help="torch threads; 0 leaves default")
    parser.add_argument("--concurrency", type=int, default=4, help="with --compat")
    parser.add_argument("--tasks", nargs="*", default=[t.name for t in TASKS])
    parser.add_argument("-n", type=int, default=DEFAULT_N)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.bundle:
        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        answer, system, workers = _in_process(args.bundle, int8=args.int8)
    else:
        if not args.model:
            parser.error("--model is required with --compat")
        key = os.environ.get("TRIGON_INCUMBENT_KEY")
        answer, system, workers = _compat(args.compat, args.model, key, args.concurrency)

    results: dict[str, dict] = {}
    answered: dict[str, tuple[list[Case], list[dict | None]]] = {}
    for name in args.tasks:
        t = task(name)
        if args.compat and t.over_compat_cap:
            results[name] = {"skipped": "over the incumbent's per-question option cap"}
            continue
        cases = build(t, args.n)
        print(f"{name}: {len(cases)} cases", file=sys.stderr, flush=True)
        out = _run(t, cases, answer, workers, args.out / "raw" / name.replace("/", "__"))
        answered[name] = (cases, out)
        results[name] = {"kind": t.kind, "note": t.note, **_score(cases, out)}
        print(f"  {json.dumps(results[name])}", file=sys.stderr, flush=True)

    # Invariance: the order task against the shift task it reorders, by case.
    for name, (cases, out) in answered.items():
        t = task(name)
        if t.kind != "invariance":
            continue
        base = name.replace("/order", "/shift")
        if base not in answered:
            results[name]["agreement"] = None
            continue
        a = selected_names([(r or {}).get("answers", {}).get("intent") for r in out], cases)
        base_cases, base_out = answered[base]
        b = selected_names(
            [(r or {}).get("answers", {}).get("intent") for r in base_out], base_cases
        )
        pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
        results[name]["agreement"] = sum(x == y for x, y in pairs) / len(pairs) if pairs else None

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"system": system, "n": args.n, "tasks": results}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "| Task | n | Accuracy | Chance | ECE (floor p95) | NLL | Agreement | p50 s |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        if "skipped" in r or not r.get("n"):
            lines.append(f"| {name} | — | {r.get('skipped', 'no answers')} | | | | | |")
            continue
        verdict = "" if r["miscalibration_detectable"] else ", not separable"
        agreement = f"{r['agreement']:.3f}" if r.get("agreement") is not None else ""
        lines.append(
            f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['chance']:.3f} | "
            f"{r['ece']:.3f} ({r['ece_floor_p95']:.3f}{verdict}) | {r['nll']:.2f} | "
            f"{agreement} | {r['p50_seconds']:.2f} |"
        )
    (args.out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
