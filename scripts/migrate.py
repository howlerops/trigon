#!/usr/bin/env python
"""Compare this gateway against whatever you run today, on your own traffic.

    python scripts/migrate.py requests.jsonl --incumbent https://api.example.com
    python scripts/migrate.py requests.jsonl --incumbent-cmd 'my-baseline'
    python scripts/migrate.py requests.jsonl --labels outcomes.jsonl

**Nobody switches on a promise.** They switch on a diff over their own traffic,
which is why this exists and why it takes *your* requests rather than shipping
its own. Give it a JSONL file of `/v1/systemone` request bodies -- a day of
production traffic, replayed -- and it reports where the two systems agree,
where they diverge, and, if you have outcomes, which one was right.

**It does not assume this project wins.** The three things it reports are
agreement, calibration and disagreement-with-truth, and the third is the one
that matters: two systems agreeing 95% of the time tells you nothing about
whether the 5% is the important 5%. Where labels are supplied it prints which
system is right on exactly the cases they disagree about, which is the only
comparison a migration decision actually rests on.

An incumbent is reached either over HTTP (`--incumbent`) or by a command that
reads a request on stdin and writes a response on stdout (`--incumbent-cmd`),
so a baseline that is not an HTTP service -- a prompted LLM behind a script, a
rules engine -- can be compared without wrapping it in a server first.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))


def _selected(answer: dict) -> str | None:
    """The decision an answer carries, whatever primitive it came from.

    Choice says `selected`, Score says `level`, Noul says a probability that
    has to be thresholded. Comparing them needs one notion of "what did it
    decide", and this is it.
    """
    if "selected" in answer:
        return str(answer["selected"])
    if "level" in answer:
        return str(answer["level"])
    if "probability" in answer:
        return "yes" if float(answer["probability"]) >= 0.5 else "no"
    return None


def _confidence(answer: dict) -> float | None:
    """The confidence, where the primitive has one.

    A Noul deliberately has no confidence field -- the contract says so -- so
    its probability distance from even odds stands in. That is not the same
    quantity and is labelled as such wherever it is reported.
    """
    if "confidence" in answer:
        return float(answer["confidence"])
    if "probability" in answer:
        return abs(float(answer["probability"]) - 0.5) * 2.0
    return None


def _call_http(url: str, body: dict, timeout: float) -> dict:
    import urllib.request

    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _call_cmd(command: str, body: dict, timeout: float) -> dict:
    result = subprocess.run(
        command,
        shell=True,
        input=json.dumps(body),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", help="JSONL of /v1/systemone request bodies")
    parser.add_argument("--incumbent", default=None, help="base URL of what you run today")
    parser.add_argument("--incumbent-cmd", default=None, help="command; request on stdin")
    parser.add_argument("--challenger", default=None, help="base URL; default is in-process")
    parser.add_argument("--weights", default=None, help="checkpoint for the in-process challenger")
    parser.add_argument("--backend", default="lexical")
    parser.add_argument(
        "--labels",
        default=None,
        help="JSONL of {question_id: label} aligned with the requests; turns "
        "agreement into a verdict",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if not args.incumbent and not args.incumbent_cmd:
        raise SystemExit("give --incumbent or --incumbent-cmd; there is nothing to compare to")

    bodies = [
        json.loads(line)
        for line in pathlib.Path(args.requests).read_text().splitlines()
        if line.strip()
    ]
    labels: list[dict] = []
    if args.labels:
        labels = [
            json.loads(line)
            for line in pathlib.Path(args.labels).read_text().splitlines()
            if line.strip()
        ]
        if len(labels) != len(bodies):
            raise SystemExit(
                f"{len(bodies)} requests and {len(labels)} label rows; they have to align "
                "by position, because that is the only thing joining them"
            )
    if args.limit:
        bodies, labels = bodies[: args.limit], labels[: args.limit]

    if args.challenger:

        def challenge(body: dict) -> dict:
            return _call_http(args.challenger, body, args.timeout)
    else:
        from fastapi.testclient import TestClient

        from trigon.server.app import build_app
        from trigon.server.config import ServerConfig

        client = TestClient(build_app(ServerConfig(backend=args.backend, weights=args.weights)))

        def challenge(body: dict) -> dict:
            response = client.post("/v1/systemone", json=body)
            response.raise_for_status()
            return response.json()

    def incumbent(body: dict) -> dict:
        if args.incumbent:
            return _call_http(args.incumbent, body, args.timeout)
        return _call_cmd(args.incumbent_cmd, body, args.timeout)

    agree: dict[str, int] = {}
    total: dict[str, int] = {}
    disagreements: list[tuple[int, str, str, str]] = []
    conf_ours: dict[str, list[float]] = {}
    failures = 0

    for index, body in enumerate(bodies):
        try:
            theirs = incumbent(body)
            ours = challenge(body)
        except Exception as error:  # noqa: BLE001 - a failed call is a result
            failures += 1
            print(f"  request {index}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        for qid, ours_answer in ours.get("answers", {}).items():
            theirs_answer = theirs.get("answers", {}).get(qid)
            if theirs_answer is None:
                continue
            mine, yours = _selected(ours_answer), _selected(theirs_answer)
            total[qid] = total.get(qid, 0) + 1
            if mine == yours:
                agree[qid] = agree.get(qid, 0) + 1
            else:
                disagreements.append((index, qid, str(yours), str(mine)))
            confidence = _confidence(ours_answer)
            if confidence is not None:
                conf_ours.setdefault(qid, []).append(confidence)

    if not total:
        print("no comparable answers; do both systems answer the same question ids?")
        return 1

    print()
    print(
        f"{len(bodies) - failures} of {len(bodies)} requests compared"
        + (f", {failures} failed" if failures else "")
    )
    print()
    print(f"{'question':<20} {'n':>6} {'agree':>8} {'our mean conf':>14}")
    print("-" * 52)
    for qid in sorted(total):
        confidences = conf_ours.get(qid, [])
        mean = f"{statistics.mean(confidences):.3f}" if confidences else "—"
        print(f"{qid:<20} {total[qid]:>6} {agree.get(qid, 0) / total[qid]:>7.1%} {mean:>14}")

    if not labels:
        print()
        print("Agreement is not a verdict. Two systems agreeing 95% of the time")
        print("says nothing about whether the other 5% is the important 5%.")
        print("Pass --labels to find out which one is right where they differ.")
        return 0

    # The comparison a migration decision actually rests on.
    ours_right = theirs_right = both_wrong = 0
    for index, qid, yours, mine in disagreements:
        truth = labels[index].get(qid)
        if truth is None:
            continue
        if str(truth) == mine:
            ours_right += 1
        elif str(truth) == yours:
            theirs_right += 1
        else:
            both_wrong += 1

    judged = ours_right + theirs_right + both_wrong
    print()
    if not judged:
        print("None of the disagreements carried a label, so there is no verdict.")
        return 0
    print(f"On the {judged} disagreements with a label:")
    for caption, count in (
        ("this project right", ours_right),
        ("incumbent right", theirs_right),
        ("both wrong", both_wrong),
    ):
        print(f"  {caption:<20} {count:>6}  ({count / judged:.1%})")
    print()
    print("This is the number to migrate on. Where the two systems agree, the")
    print("choice between them is a question of cost and latency, not accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
