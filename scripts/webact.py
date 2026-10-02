#!/usr/bin/env python
"""Score a system on real web-agent steps: the operation, its target, and the time.

    python scripts/webact.py --bundle runs/mix-q3-06b-s0 --out reports/webact/q3
    TRIGON_INCUMBENT_KEY=... python scripts/webact.py --compat https://api.example.com/api \\
        --model their-model --out reports/webact/them

The cases are `trigon.evals.webact` over Mind2Web: one request per step, an
``operation`` Choice and one target Choice per operation over a 30-element
table, as a browser agent on a typed-decision API sends it. Three numbers per
system, the ones such an agent lives by:

* **operation accuracy** -- CLICK, TYPE_TEXT or SELECT;
* **target accuracy** -- the right element, in the head of the true operation;
* **step success** -- the operation right *and* the element its own head
  picked right, i.e. the action the agent would actually execute.

Plus calibration of the step's executed choice (beside its noise floor) and
latency per request, which is the whole reason the workload exists. The
system wrappers, raw-answer cache and resume are `scripts/generality.py`'s.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from generality import _compat, _in_process, _run  # noqa: E402

from trigon.calibration.metrics import report  # noqa: E402
from trigon.evals.corpora import load  # noqa: E402


class _Task:
    name = "webact"


def _argmax(answer: dict | None) -> str | None:
    probabilities = (answer or {}).get("probabilities") or {}
    return max(probabilities, key=lambda k: probabilities[k]) if probabilities else None


def score(cases, results) -> dict:
    op_ok = target_ok = step_ok = n = failed = 0
    step_probs, step_labels, seconds = [], [], []
    by_op: dict[str, list[int]] = {}
    for case, result in zip(cases, results, strict=True):
        answers = (result or {}).get("answers") or {}
        if "operation" not in answers:
            failed += 1
            continue
        n += 1
        ops = [o.name for o in case.request.questions["operation"].options]
        true_op = ops[case.expected["operation"].label]
        head = true_op.lower() + "_target"
        names = [o.name for o in case.request.questions[head].options]
        true_target = names[case.expected[head].label]
        op = _argmax(answers["operation"])
        target_here = _argmax(answers.get(head))
        op_ok += op == true_op
        target_ok += target_here == true_target
        # What the agent executes: its operation, then that operation's own head.
        executed = _argmax(answers.get((op or "").lower() + "_target")) if op else None
        success = op == true_op and executed == true_target
        step_ok += success
        by_op.setdefault(true_op, []).append(int(success))
        # Confidence in the executed action: P(op) x P(target | op).
        p_op = float((answers["operation"].get("probabilities") or {}).get(op, 0.0))
        p_target = float(
            ((answers.get((op or "").lower() + "_target") or {}).get("probabilities") or {}).get(
                executed, 1.0 if op not in ("CLICK", "TYPE_TEXT", "SELECT") else 0.0
            )
        )
        confidence = p_op * p_target
        step_probs.append([1 - confidence, confidence])
        step_labels.append(int(success))
        seconds.append(result.get("_seconds", 0.0))
    if not n:
        return {"n": 0, "failed": failed}
    calibration = report(step_probs, step_labels)
    seconds.sort()
    return {
        "n": n,
        "failed": failed,
        "operation_accuracy": op_ok / n,
        "target_accuracy": target_ok / n,
        "step_success": step_ok / n,
        "step_success_by_operation": {k: sum(v) / len(v) for k, v in sorted(by_op.items())},
        "step_counts_by_operation": {k: len(v) for k, v in sorted(by_op.items())},
        "executed_confidence_ece": calibration.ece,
        "executed_confidence_ece_floor_p95": calibration.floor.p95 if calibration.floor else None,
        "mean_executed_confidence": sum(p[1] for p in step_probs) / n,
        "p50_seconds": seconds[n // 2],
        "p95_seconds": seconds[min(n - 1, int(n * 0.95))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--bundle", type=pathlib.Path)
    who.add_argument("--compat")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("-n", type=int, default=600)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.bundle:
        if args.threads:
            import torch

            torch.set_num_threads(args.threads)
        answer, system, workers = _in_process(args.bundle)
    else:
        if not args.model:
            parser.error("--model is required with --compat")
        answer, system, workers = _compat(
            args.compat, args.model, os.environ.get("TRIGON_INCUMBENT_KEY"), args.concurrency
        )
    cases = load("mind2web", "test", purpose="eval", limit=args.n)
    print(f"webact: {len(cases)} steps", file=sys.stderr, flush=True)
    results = _run(_Task(), cases, answer, workers, args.out / "raw")
    scored = score(cases, results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps({"system": system, "n": args.n, "webact": scored}, indent=2)
    )
    print(json.dumps(scored, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
