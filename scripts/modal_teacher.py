"""Generate and label the teacher-workflow stream on Modal, and bring it back.

    python scripts/modal_teacher.py launch --n 6000 --shards 4 --verifiable 1700  # exits at once
    python scripts/modal_teacher.py status tw0-n6000
    python scripts/modal_teacher.py merge tw0-n6000      # shards -> cases.jsonl.gz, prints SHA-256
    python scripts/modal_teacher.py fetch tw0-n6000      # into the ignored corpus cache
    python scripts/modal_teacher.py fixture tw0-n6000    # <= 20 cases into tests/fixtures/
    python scripts/modal_teacher.py cost tw0-n6000       # dollars per 1k labelled cases

**The teacher is Qwen2.5-7B-Instruct at a pinned revision** (Apache-2.0 on its
card and in its LICENSE at that revision; `trigon.evals.teacher`), served by
vLLM on one A10G per shard. 7B in bf16 is 15.2 GB of weights, which leaves a
24 GB A10G about 5 GB of KV cache at 4,096 tokens -- enough, since no prompt
here is longer than ~2,500. The A10G rather than an L4: the same memory at
1.4x the price, twice the memory bandwidth for the generation half, and the
GPU every other job in this repository runs on.

**Two calls per case, both to the same model.** The first writes a (state,
schema) pair to a plan drawn from `trigon.evals.teacher.generation_plan`
(domain, state shape, question primitives and counts, a borderline share),
seeded per case. The second answers each question **alone** and records the
teacher's full distribution: for every declared label, the log-likelihood of
the whole reply -- the label's tokens and then the end of the turn -- read off
`prompt_logprobs`, never an argmax and never only the first token. `docs/data.md`
says a teacher call that returns an argmax has thrown away most of what it
cost; a first-token readout throws away the rest whenever two options share a
first token.

**A verifiable holdout rides along.** `label_verifiable` has the same teacher
answer cases whose truth is computed (`synthetic_outcome_cases`, the
verifiable stream) and publishes the *teacher's* accuracy and calibration
against that truth, with its noise floor. That is the one place a teacher's
probabilities are scored as calibration, because there the labels are not the
teacher's.

**Durability is the same design as `modal_train.py`.** `launch` deploys a
per-commit app, spawns one call per shard, writes the call ids to the volume
and exits. Every shard writes its records to the `trigon-teacher` Volume
before it returns. Nothing is committed from here: the data stays on the
volume and in the ignored corpus cache, and only a <=20-case fixture is
written into the repository.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DATA = "/teacher"
WEIGHTS = "/weights"
APP_NAME = "trigon-teacher"
VLLM = "vllm==0.11.0"

#: Modal list prices on 2026-09-27 (modal.com/pricing), per second. Recorded
#: here so the cost per 1k cases is computed rather than estimated; the
#: container is billed for CPU and memory as well as the GPU.
PRICE_PER_SECOND = {"A10G": 0.000306, "L4": 0.000222, "A100": 0.000583, "H100": 0.001097}
CPU_CORES, MEMORY_GIB = 4, 32
CPU_PRICE, MEMORY_PRICE = 0.0000131, 0.00000222

image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers 5 dropped an attribute vLLM 0.11 reads when it loads a tokenizer.
    .pip_install(VLLM, "transformers==4.57.1", "pydantic>=2")
    .env(
        {
            "HF_HOME": f"{WEIGHTS}/hf",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "PYTHONPATH": "/opt/trigon/src",
        }
    )
    # Not under /root: the script lands in /root, and a /root/trigon directory
    # there is a namespace package that shadows the real one.
    .add_local_dir(REPO / "src", "/opt/trigon/src", ignore=["**/__pycache__/**"])
)

app = modal.App(APP_NAME)
data = modal.Volume.from_name("trigon-teacher", create_if_missing=True)
weights = modal.Volume.from_name("trigon-weights", create_if_missing=True)

GENERATION = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 1536}


def _engine():
    """The pinned teacher under vLLM, and its tokenizer."""
    from vllm import LLM

    from trigon.evals.teacher import TEACHER_MODEL, TEACHER_REVISION

    llm = LLM(
        model=TEACHER_MODEL,
        revision=TEACHER_REVISION,
        tokenizer_revision=TEACHER_REVISION,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        seed=0,
    )
    return llm, llm.get_tokenizer()


def _chat_ids(tokenizer, messages) -> list[int]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def _score(llm, tokenizer, items):
    """Full-reply log-likelihoods: items are (messages, [reply strings]).

    Each reply is scored as the teacher would emit it -- the reply's tokens
    and then `<|im_end|>` -- from `prompt_logprobs`, summed over the reply's
    positions. Returns, per item, a list of (log-likelihood, token count).
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    prompts, index = [], []
    for i, (messages, replies) in enumerate(items):
        prefix = _chat_ids(tokenizer, messages)
        for j, reply in enumerate(replies):
            ids = tokenizer.encode(reply, add_special_tokens=False) + [end]
            prompts.append(TokensPrompt(prompt_token_ids=prefix + ids))
            index.append((i, j, len(prefix), len(ids)))
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    scores = [[None] * len(replies) for _, replies in items]
    for output, (i, j, start, length) in zip(outputs, index, strict=True):
        ids = output.prompt_token_ids
        total = sum(output.prompt_logprobs[k][ids[k]].logprob for k in range(start, start + length))
        scores[i][j] = (total, length)
    return scores


def _write_gz(path: pathlib.Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _teacher_meta() -> dict:
    from trigon.evals.teacher import (
        GENERATION_TEMPLATE_SHA256,
        LABEL_TEMPLATE_SHA256,
        TEACHER_MODEL,
        TEACHER_REVISION,
    )

    return {
        "model": TEACHER_MODEL,
        "revision": TEACHER_REVISION,
        "engine": VLLM,
        "generation_template_sha256": GENERATION_TEMPLATE_SHA256,
        "label_template_sha256": LABEL_TEMPLATE_SHA256,
        "labeller": "full-reply log-likelihood (label tokens + <|im_end|>), softmax over labels",
        "generation": GENERATION,
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=CPU_CORES,
    memory=MEMORY_GIB * 1024,
    timeout=60 * 60 * 6,
    volumes={DATA: data, WEIGHTS: weights},
)
def build_shard(build: str, seed: int, start: int, n: int, run: dict) -> dict:
    """Generate and label cases ``start`` to ``start + n`` of one build."""
    import torch
    from vllm import SamplingParams

    from trigon.evals.teacher import (
        Rejected,
        generation_messages,
        generation_plan,
        label_messages,
        labels_of,
        option_continuations,
        parse_generated,
        softmax,
    )

    began = time.time()
    llm, tokenizer = _engine()
    weights.commit()
    loaded = time.time()

    plans = generation_plan(n, seed=seed, start=start)
    prompts = [
        {"prompt_token_ids": _chat_ids(tokenizer, generation_messages(plan))} for plan in plans
    ]
    params = [SamplingParams(seed=plan.sample_seed, **GENERATION) for plan in plans]
    generated = llm.generate(prompts, params, use_tqdm=False)
    generated_at = time.time()

    kept, rejected, dropped = [], {}, []
    output_tokens = 0
    for plan, output in zip(plans, generated, strict=True):
        text = output.outputs[0].text
        output_tokens += len(output.outputs[0].token_ids)
        try:
            request = parse_generated(text, plan)
        except Rejected as error:
            reason = " ".join(str(error).split(":")[0].split()[:4])
            rejected[reason] = rejected.get(reason, 0) + 1
            # Kept, so a rejection rate can be read rather than guessed at.
            dropped.append({"case_id": plan.case_id, "reason": str(error), "text": text})
            continue
        kept.append((plan, request, text))

    items, owners = [], []
    for k, (_, request, _) in enumerate(kept):
        for qid, question in request.questions.items():
            items.append((label_messages(request.state, question), option_continuations(question)))
            owners.append((k, qid))
    scores = _score(llm, tokenizer, items)
    labelled_at = time.time()

    meta = _teacher_meta()
    records = []
    labels_by_case: dict[int, dict] = {}
    for (k, qid), row in zip(owners, scores, strict=True):
        question = kept[k][1].questions[qid]
        logprobs = [lp for lp, _ in row]
        labels_by_case.setdefault(k, {})[qid] = {
            "options": labels_of(question),
            "logprobs": logprobs,
            "tokens": [t for _, t in row],
            "probabilities": softmax(logprobs),
            # How much of the teacher's reply mass fell on the declared
            # strings at all; the rest went to replies that are not labels.
            "declared_mass": sum(pow(2.718281828459045, lp) for lp in logprobs),
        }
    for k, (plan, request, text) in enumerate(kept):
        records.append(
            {
                "case_id": plan.case_id,
                "domain": plan.domain,
                "plan": plan.to_dict(),
                "state": request.state,
                "questions": {
                    qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()
                },
                "labels": labels_by_case[k],
                "teacher": meta,
                "raw_generation": text,
            }
        )

    folder = pathlib.Path(DATA) / build
    _write_gz(folder / f"shard-{start:06d}.jsonl.gz", records)
    _write_gz(folder / f"rejected-{start:06d}.jsonl.gz", dropped)
    finished = time.time()
    summary = {
        "build": build,
        "start": start,
        "n_planned": n,
        "n_kept": len(records),
        "rejected": rejected,
        "questions": len(items),
        "scored_replies": sum(len(r) for _, r in items),
        "generated_tokens": output_tokens,
        "seconds": {
            "load": round(loaded - began, 1),
            "generate": round(generated_at - loaded, 1),
            "label": round(labelled_at - generated_at, 1),
            "total": round(finished - began, 1),
        },
        "gpu_requested": run["gpu"],
        "gpu_actual": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        **{k: run[k] for k in ("commit", "dirty")},
        "teacher": meta,
    }
    (folder / f"shard-{start:06d}.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    return summary


@app.function(
    image=image,
    gpu="A10G",
    cpu=CPU_CORES,
    memory=MEMORY_GIB * 1024,
    timeout=60 * 60 * 2,
    volumes={DATA: data, WEIGHTS: weights},
)
def label_verifiable(build: str, n: int, run: dict) -> dict:
    """The teacher against computed truth: the one place its probabilities are calibration.

    Cases from `synthetic_outcome_cases` (noise 0, so the truth is the
    predicate) and the first step of the two committed workflows, whose
    outcomes the generator computed. The teacher answers them exactly as it
    answers the generated stream, and its accuracy and ECE are scored against
    the truth, with the simulated noise floor.
    """
    import torch

    from trigon.calibration.metrics import report
    from trigon.evals import moderation_queue, support_triage, synthetic_outcome_cases
    from trigon.evals.teacher import label_messages, labels_of, option_continuations, softmax
    from trigon.types import NoulQuestion

    began = time.time()
    llm, tokenizer = _engine()
    rows = []  # (suite, qid, question, state, truth)
    for case in synthetic_outcome_cases(n, seed=0):
        for qid, question in case.request.questions.items():
            rows.append(
                ("synthetic", qid, question, case.request.state, case.expected[qid].hard_label)
            )
    for name, (workflow, cases) in (
        ("support_triage", support_triage(600, seed=0)),
        ("moderation_queue", moderation_queue(600, seed=0)),
    ):
        first = workflow.steps[0]
        for case in cases:
            for qid, question in first.build(case.state, {}).items():
                labels = labels_of(question)
                truth = case.outcome[qid]
                rows.append((name, qid, question, case.state, labels.index(truth)))

    items = [(label_messages(s, q), option_continuations(q)) for _, _, q, s, _ in rows]
    scores = _score(llm, tokenizer, items)
    records, by_suite = [], {}
    for (suite, qid, question, _state, truth), row in zip(rows, scores, strict=True):
        probabilities = softmax([lp for lp, _ in row])
        by_suite.setdefault(suite, ([], []))
        by_suite[suite][0].append(probabilities)
        by_suite[suite][1].append(truth)
        by_suite.setdefault(f"{suite}/{qid}", ([], []))
        by_suite[f"{suite}/{qid}"][0].append(probabilities)
        by_suite[f"{suite}/{qid}"][1].append(truth)
        records.append(
            {
                "suite": suite,
                "question": qid,
                "kind": "noul" if isinstance(question, NoulQuestion) else question.type,
                "labels": labels_of(question),
                "truth": truth,
                "logprobs": [lp for lp, _ in row],
                "probabilities": probabilities,
            }
        )
    reports = {}
    for suite, (probs, labels) in by_suite.items():
        # The floor is simulated only for whole suites large enough to quote.
        scored = report(probs, labels, slice_name=suite, simulate_floor="/" not in suite)
        reports[suite] = {
            "n": scored.n,
            "accuracy": scored.accuracy,
            "mean_confidence": scored.mean_confidence,
            "overconfidence": scored.overconfidence,
            "ece": scored.ece,
            "adaptive_ece": scored.adaptive_ece,
            "brier": scored.brier,
            "floor_p95": scored.floor.p95 if scored.floor else None,
            "distinguishable": scored.distinguishable,
        }
    folder = pathlib.Path(DATA) / build
    _write_gz(folder / "verifiable.jsonl.gz", records)
    summary = {
        "build": build,
        "n_questions": len(records),
        "reports": reports,
        "seconds": {"total": round(time.time() - began, 1)},
        "gpu_requested": run["gpu"],
        "gpu_actual": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        **{k: run[k] for k in ("commit", "dirty")},
        "teacher": _teacher_meta(),
    }
    (folder / "verifiable.json").write_text(json.dumps(summary, indent=2))
    data.commit()
    return summary


# -- the local side ---------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def launch(args) -> None:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not args.allow_dirty:
        raise SystemExit("commit first, or pass --allow-dirty and it will be recorded")
    build = args.build or f"tw{args.seed}-n{args.n}"
    run = {"commit": commit, "dirty": dirty, "gpu": args.gpu}
    deployed = f"{APP_NAME}-{commit[:12]}" + ("-dirty" if dirty else "")
    with modal.enable_output():
        app.deploy(name=deployed)
    shard = modal.Function.from_name(deployed, "build_shard").with_options(gpu=args.gpu)
    # Every call is one GPU, and the workspace's cap is shared: --verifiable
    # with shards is shards + 1 GPUs, so it can also be launched alone with
    # --n 0 once the shards finish.
    size = max(1, -(-args.n // max(args.shards, 1)))
    calls = {}
    only = {int(x) for x in args.only.split(",") if x.strip()}
    for start in range(0, args.n, size):
        if only and start not in only:
            continue
        calls[f"shard-{start:06d}"] = shard.spawn(
            build, args.seed, start, min(size, args.n - start), run
        ).object_id
    if args.verifiable:
        verify = modal.Function.from_name(deployed, "label_verifiable").with_options(gpu=args.gpu)
        calls["verifiable"] = verify.spawn(build, args.verifiable, run).object_id
    try:  # a later launch into the same build adds its calls to the record
        calls = {**json.loads(_read(f"{build}/calls.json")), **calls}
    except Exception:  # noqa: BLE001 - no record yet
        pass
    with data.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(json.dumps(calls).encode()), f"{build}/calls.json")
    print(f"build {build}: {len(calls)} calls on {args.gpu}, app {deployed}")
    print(json.dumps(calls))


def _read(path: str) -> bytes:
    return b"".join(data.read_file(path))


def _entries(build: str) -> list[str]:
    try:
        return sorted(entry.path for entry in data.listdir(build))
    except Exception:  # noqa: BLE001 - the SDK raises a bare not-found
        return []


def status(args) -> None:
    calls = json.loads(_read(f"{args.build}/calls.json"))
    for name, call_id in calls.items():
        call = modal.FunctionCall.from_id(call_id)
        try:
            call.get(timeout=0)
            state = "done"
        except TimeoutError:
            state = "running"
        except Exception as error:  # noqa: BLE001 - report whatever failed
            state = f"FAILED: {type(error).__name__}: {str(error)[:200]}"
        print(f"  {name}: {state}")


def cancel(args) -> None:
    for name, call_id in json.loads(_read(f"{args.build}/calls.json")).items():
        modal.FunctionCall.from_id(call_id).cancel()
        print(f"cancelled {name}")


def merge(args) -> None:
    """Concatenate the shards in order into one file on the volume, and pin it."""
    shards = [e for e in _entries(args.build) if e.endswith(".jsonl.gz") and "/shard-" in e]
    if not shards:
        raise SystemExit(f"no shards under {args.build}")
    lines = []
    for name in shards:
        lines += gzip.decompress(_read(name)).decode().splitlines()
    blob = gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0)
    with data.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(blob), f"{args.build}/cases.jsonl.gz")
    print(f"{args.build}/cases.jsonl.gz: {len(lines)} cases from {len(shards)} shards")
    print(f"sha256 {hashlib.sha256(blob).hexdigest()}")


def fetch(args) -> None:
    from trigon.evals.corpora import TEACHER_WORKFLOWS, cache_root

    target = cache_root() / TEACHER_WORKFLOWS.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ("cases.jsonl.gz", "verifiable.json"):
        blob = _read(f"{args.build}/{name}")
        (target / name).write_bytes(blob)
        print(f"{target / name}  sha256 {hashlib.sha256(blob).hexdigest()}")


def fixture(args) -> None:
    """At most twenty cases, one per domain, into the test fixtures."""
    from trigon.evals.corpora import TEACHER_WORKFLOWS, cache_root

    source = cache_root() / TEACHER_WORKFLOWS.name / "cases.jsonl.gz"
    chosen, seen = [], set()
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["domain"] in seen:
                continue
            seen.add(record["domain"])
            chosen.append(record)
            if len(chosen) == 20:
                break
    out = REPO / "tests" / "fixtures" / "teacher_workflows_sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in chosen)
    )
    print(f"wrote {len(chosen)} cases to {out}")


def cost(args) -> None:
    """Dollars per 1k labelled cases, from the seconds each shard recorded."""
    summaries = [
        json.loads(_read(e)) for e in _entries(args.build) if e.endswith(".json") and "/shard-" in e
    ]
    per_second = PRICE_PER_SECOND[args.gpu] + CPU_CORES * CPU_PRICE + MEMORY_GIB * MEMORY_PRICE
    seconds = sum(s["seconds"]["total"] for s in summaries)
    kept = sum(s["n_kept"] for s in summaries)
    planned = sum(s["n_planned"] for s in summaries)
    dollars = seconds * per_second
    out = {
        "build": args.build,
        "shards": len(summaries),
        "planned": planned,
        "kept": kept,
        "questions": sum(s["questions"] for s in summaries),
        "scored_replies": sum(s["scored_replies"] for s in summaries),
        "generated_tokens": sum(s["generated_tokens"] for s in summaries),
        "rejected": {
            k: sum(s["rejected"].get(k, 0) for s in summaries)
            for k in sorted({k for s in summaries for k in s["rejected"]})
        },
        "container_seconds": round(seconds, 1),
        "seconds_by_phase": {
            phase: round(sum(s["seconds"][phase] for s in summaries), 1)
            for phase in ("load", "generate", "label")
        },
        "dollars_per_second": per_second,
        "dollars": round(dollars, 2),
        "dollars_per_1k_kept": round(1000 * dollars / max(kept, 1), 2),
        "gpu_actual": sorted({s["gpu_actual"] for s in summaries}),
    }
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("launch")
    go.add_argument("--n", type=int, default=6000)
    go.add_argument("--seed", type=int, default=0)
    go.add_argument(
        "--shards", type=int, default=4, help="how the build is cut; one GPU per shard launched"
    )
    go.add_argument(
        "--only",
        default="",
        help="launch only these shard starts (e.g. 3000,4500), to stay under a GPU cap",
    )
    go.add_argument("--gpu", default="A10G")
    go.add_argument("--build", default="")
    go.add_argument(
        "--verifiable", type=int, default=0, help="also label this many verifiable cases"
    )
    go.add_argument("--allow-dirty", action="store_true")
    for name in ("status", "cancel", "merge", "fetch", "fixture", "cost"):
        command = sub.add_parser(name)
        command.add_argument("build")
        if name == "cost":
            command.add_argument("--gpu", default="A10G")
    args = parser.parse_args(argv)
    handlers = {
        "launch": launch,
        "status": status,
        "cancel": cancel,
        "merge": merge,
        "fetch": fetch,
        "fixture": fixture,
        "cost": cost,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
