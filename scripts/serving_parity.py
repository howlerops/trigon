#!/usr/bin/env python
"""Does the served model answer as the evaluated one did? Checked before a publish counts.

    TRIGON_PARITY_KEY_FILE=<file> python scripts/serving_parity.py \\
        --bundle runs/lms-4b --url https://<worker> --device cuda --out reports/parity/lms-4b

A published number describes the bundle as it was evaluated. What a caller
gets is that bundle loaded somewhere else -- another GPU, another precision,
another copy of the files -- behind a Worker. This sends the same requests to
both and compares every answer: the argmax must agree on nearly every
question and the probabilities must sit within a tolerance that allows for a
different GPU's rounding, not for a different model. A failure means the
deployment is not the model the card describes, and it is not published.

The key is read from a file, never from the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.evals.generality import build, task  # noqa: E402

TASKS = ("banking77/shift", "clinc150", "boolq")


def _probabilities(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        p = float(answer["probability"])
        return {"no": 1 - p, "yes": p}
    return {k: float(v) for k, v in answer["probabilities"].items()}


def compare(local: list[dict], served: list[dict]) -> dict:
    agree, total, worst = 0, 0, 0.0
    for a, b in zip(local, served, strict=True):
        for qid, answer in a["answers"].items():
            p, q = _probabilities(answer), _probabilities(b["answers"][qid])
            if set(p) != set(q):
                raise ValueError(f"{qid}: the served answer has different labels")
            total += 1
            agree += int(max(p, key=p.get) == max(q, key=q.get))
            worst = max(worst, max(abs(p[k] - q[k]) for k in p))
    return {"questions": total, "argmax_agreement": agree / total, "max_abs_diff": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--url", required=True, help="the served base URL (Worker or gateway)")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--model", default=None, help="the request's tier, e.g. trigon-large; default tier if unset"
    )
    parser.add_argument("-n", type=int, default=40, help="cases per task")
    parser.add_argument("--min-agreement", type=float, default=0.98)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    key_file = os.environ.get("TRIGON_PARITY_KEY_FILE")
    key = pathlib.Path(key_file).read_text().strip() if key_file else None

    from fastapi.testclient import TestClient

    from trigon.server.app import build_app
    from trigon.server.config import ServerConfig

    env = {"TRIGON_BACKEND": "torch", "TRIGON_WEIGHTS": str(args.bundle / "adapter.pt")}
    for name, variable in (
        ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
        ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
    ):
        if (args.bundle / name).exists():
            env[variable] = str(args.bundle / name)
    if args.device:
        env["TRIGON_DEVICE"] = args.device
    client = TestClient(build_app(ServerConfig.from_env(env)))
    local_version = client.get("/healthz").json()

    def served(body: dict) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "trigon-parity"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/decide", data=json.dumps(body).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
            return json.loads(response.read())

    local, remote, models = [], [], set()
    for name in TASKS:
        for case in build(task(name), n=args.n):
            body = case.request.model_dump(exclude_none=True)
            if args.model:
                body["model"] = args.model
            response = client.post("/v1/decide", json=body)
            response.raise_for_status()
            local.append(response.json())
            remote.append(served(body))
            models.add((local[-1]["model"], remote[-1]["model"]))
    result = compare(local, remote)
    result["models"] = sorted({m for pair in models for m in pair})
    result["same_build"] = all(a == b for a, b in models)
    result["local_health"] = local_version
    result["passed"] = (
        result["same_build"]
        and result["argmax_agreement"] >= args.min_agreement
        and result["max_abs_diff"] <= args.tolerance
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "local_health"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
