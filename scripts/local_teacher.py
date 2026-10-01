#!/usr/bin/env python
"""Build the teacher-labelled stream on this machine, with a local MoE teacher.

    ollama pull qwen3:30b            # Qwen3-30B-A3B, Apache-2.0, ~3B active per token
    python scripts/local_teacher.py --n 6000 --seed 1 --out "$TRIGON_CORPUS_CACHE/teacher-local"

`scripts/modal_teacher.py` builds the same stream on rented GPUs with vLLM.
This builds it through a local ollama server instead, and differs in exactly
two places, both recorded on every record:

* **The teacher** is the ollama model named by ``--model`` (default
  ``qwen3:30b``, the Qwen3-30B-A3B mixture of experts), pinned by the digest
  ollama reports for it at build time rather than by a Hugging Face revision.
  Thinking is suppressed with Qwen3's documented empty think block.
* **The label readout** is first-token: each option is given a letter, the
  reply is pre-filled with ``Answer:``, and the label distribution is the
  teacher's probability on each declared letter, renormalised. vLLM scores
  every option's whole continuation; ollama returns the top 20 next-token log
  probabilities and nothing it was not asked to sample, so a question is
  limited to 20 labels here, and a letter outside the top 20 is floored one
  nat below the least likely letter returned. ``declared_mass`` records how
  much of the reply fell on the letters at all.

Everything else is the existing pipeline: `trigon.evals.teacher`'s plans,
generation prompts and parser, and its record format, so `case_from_record`
and the loader read this build unchanged. Teacher labels buy coverage, never
calibration; the harness and the calibrator fit refuse them either way.

Resumable: a case already in ``--out/records.jsonl`` is skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import math
import pathlib
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.evals.teacher import (  # noqa: E402
    GENERATION_TEMPLATE_SHA256,
    Rejected,
    generation_messages,
    generation_plan,
    labels_of,
    parse_generated,
)
from trigon.schema import render_state  # noqa: E402
from trigon.types import ChoiceQuestion, NoulQuestion  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRST"

LABEL_SYSTEM = (
    "You label one decision about a piece of state. Read the state and the question, "
    "then answer with the single letter of the option that fits best."
)
LABEL_USER = "STATE:\n{state}\n\nQUESTION: {instructions}\n\nOPTIONS:\n{options}"
LABEL_TEMPLATE_SHA256 = hashlib.sha256(f"{LABEL_SYSTEM}\x1e{LABEL_USER}".encode()).hexdigest()


def _chat(messages: list[dict[str, str]], prefill: str) -> str:
    """Qwen's chat template, with an empty think block so it answers directly."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + prefill


def _generate(host: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
        return json.loads(response.read())


def _members(question) -> list[str]:
    """The declared labels as the teacher reads them, in `labels_of` order."""
    if isinstance(question, NoulQuestion):
        return ["no", "yes"]
    members = question.options if isinstance(question, ChoiceQuestion) else question.levels
    return [m.name + (f": {m.criteria}" if m.criteria else "") for m in members]


def label(host: str, model: str, state, question) -> dict:
    members = _members(question)
    if len(members) > len(LETTERS):
        raise Rejected(f"{len(members)} labels; the first-token readout takes {len(LETTERS)}")
    options = "\n".join(f"{LETTERS[i]}. {m}" for i, m in enumerate(members))
    prompt = _chat(
        [
            {"role": "system", "content": LABEL_SYSTEM},
            {
                "role": "user",
                "content": LABEL_USER.format(
                    state=render_state(state), instructions=question.instructions, options=options
                ),
            },
        ],
        prefill="Answer:",
    )
    out = _generate(
        host,
        {
            "model": model,
            "raw": True,
            "stream": False,
            "prompt": prompt,
            "logprobs": True,
            "top_logprobs": 20,
            "options": {"temperature": 0, "num_predict": 1},
        },
    )
    top = out["logprobs"][0]["top_logprobs"]
    found: dict[str, float] = {}
    for entry in top:
        letter = entry["token"].strip()
        if len(letter) == 1 and letter in LETTERS[: len(members)] and letter not in found:
            found[letter] = entry["logprob"]
    if not found:
        raise Rejected("no declared letter in the teacher's top 20")
    floor = min(found.values()) - 1.0
    logprobs = [found.get(LETTERS[i], floor) for i in range(len(members))]
    peak = max(logprobs)
    weights = [math.exp(v - peak) for v in logprobs]
    total = sum(weights)
    return {
        "options": labels_of(question),
        "logprobs": logprobs,
        "probabilities": [w / total for w in weights],
        "declared_mass": sum(math.exp(found[k]) for k in found),
        "readout": "first-token-letter",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1, help="the build's plan seed; tw0 used 0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--model", default="qwen3:30b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.host}/api/tags", timeout=30) as response:  # noqa: S310
        tags = {m["name"]: m for m in json.loads(response.read())["models"]}
    if args.model not in tags:
        raise SystemExit(f"{args.model} is not pulled; `ollama pull {args.model}`")
    meta = {
        "model": f"ollama:{args.model}",
        "digest": tags[args.model]["digest"],
        "details": tags[args.model].get("details", {}),
        "licence": "Apache-2.0",
        "generation_template_sha256": GENERATION_TEMPLATE_SHA256,
        "label_template_sha256": LABEL_TEMPLATE_SHA256,
        "readout": "first-token-letter",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    records_path, rejected_path = args.out / "records.jsonl", args.out / "rejected.jsonl"
    done = set()
    if records_path.exists():
        done = {json.loads(line)["case_id"] for line in records_path.open() if line.strip()}
    plans = [p for p in generation_plan(args.n, seed=args.seed, start=args.start)]
    plans = [p for p in plans if p.case_id not in done]
    print(f"teacher: {len(done)} done, {len(plans)} to go with {args.model}", file=sys.stderr)
    lock = threading.Lock()
    counts = {"kept": 0, "rejected": 0}
    started = time.time()

    def one(plan) -> None:
        text = ""
        try:
            out = _generate(
                args.host,
                {
                    "model": args.model,
                    "raw": True,
                    "stream": False,
                    "prompt": _chat(generation_messages(plan), prefill=""),
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "seed": plan.sample_seed,
                        "num_predict": 1600,
                    },
                },
            )
            text = out["response"]
            request = parse_generated(text, plan)
            labels = {
                qid: label(args.host, args.model, request.state, q)
                for qid, q in request.questions.items()
            }
            record = {
                "case_id": plan.case_id,
                "domain": plan.domain,
                "plan": plan.to_dict(),
                "state": request.state,
                "questions": {
                    qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()
                },
                "labels": labels,
                "teacher": meta,
                "raw_generation": text,
            }
            path, key = records_path, "kept"
        except (Rejected, KeyError, ValueError) as error:
            record = {"case_id": plan.case_id, "reason": str(error), "text": text}
            path, key = rejected_path, "rejected"
        with lock:
            with path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            counts[key] += 1
            total = counts["kept"] + counts["rejected"]
            if total % 25 == 0:
                rate = total / (time.time() - started) * 3600
                print(f"  {total}/{len(plans)} {counts} {rate:.0f}/h", file=sys.stderr, flush=True)

    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(one, plans))

    # The loader reads one gzipped file; write it whole from the records so far.
    with gzip.open(args.out / "cases.jsonl.gz", "wt") as out:
        for line in records_path.open():
            if line.strip():
                out.write(line)
    digest = hashlib.sha256((args.out / "cases.jsonl.gz").read_bytes()).hexdigest()
    (args.out / "build.json").write_text(
        json.dumps({**meta, "seed": args.seed, "n_planned": args.n, "sha256": digest}, indent=2)
    )
    print(f"teacher: {counts}, cases.jsonl.gz sha256 {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
