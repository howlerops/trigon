"""Run a corpus training sweep on a Modal GPU, and bring the reports back.

    python scripts/modal_train.py launch --corpus helpsteer2 --n 12000 --epochs 6
    python scripts/modal_train.py status <run id>
    python scripts/modal_train.py collect <run id>

`docs/gpu-access.md` picked this route, and the reason was durability before
speed: three HelpSteer2 attempts died in this sandbox, one to an
out-of-memory kill and two to the session VM being reclaimed mid-run.

**The launcher exits in seconds, and that is the durability.** Two earlier
designs each claimed to survive this VM and neither did. The first returned
reports to the local entrypoint, so they lived only on the machine that gets
reclaimed. The second added `modal run --detach` and a Volume -- and when this
container restarted twelve minutes into a four-seed sweep, Modal cancelled
all four inputs, because a `starmap` belongs to the process feeding it and
that process was gone. `--detach` keeps an *app* alive, not the calls a dead
client was driving.

So `launch` deploys the app, `spawn`s one call per seed, prints the run id
and returns. A spawned call on a deployed app belongs to Modal, not to the
launcher; each seed writes its reports to the `trigon-runs` Volume before it
returns, and `collect` fetches them whenever this session next exists.

**Each seed is its own container.** Four seeds cost the wall clock of one.

## What it is careful about

**The commit is recorded, and it has to mean something.** The image is built
from the working tree, not from the commit, so a SHA recorded over a dirty
tree names code that did not run. `launch` refuses a dirty tree unless
`--allow-dirty` is passed, and then records that it was.

**The GPU is used, not just recorded.** The job passes `--device cuda` to
`train_corpus.py`, which refuses to fall back to the CPU; the GPU the
container actually reports is written beside every result.

**The corpus and the backbone are downloaded inside the job**, not shipped
from here -- someone else's data and weights under licences that govern
redistribution. Backbone weights are cached on the `trigon-weights` Volume so
four seeds do not fetch 3 GB four times.

**Nothing is certified by this script.** It runs seeds and returns reports;
whether they certify is what the gates say, and `CLAUDE.md`'s rule about the
median and the spread applies exactly as it does locally.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = "/runs"
WEIGHTS = "/weights"
APP_NAME = "trigon-train"
#: The workspace's plan caps it at ten GPUs at once. Calls beyond that are
#: queued, not refused -- Modal emails "you have reached your GPU limit" and
#: the extra seeds start when earlier ones finish -- so a launch that
#: overshoots is slower, not broken. `launch` says so rather than leaving a
#: queued seed to look like a hung one.
GPU_LIMIT = 10

# The training extra plus a CUDA torch. The default PyPI wheel carries CUDA on
# Linux, so nothing here pins an index -- pinning one is how a CPU wheel ends
# up on a GPU box and the job runs correctly and slowly.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "numpy>=1.26",
        "pydantic>=2",
        "fastapi>=0.110",
        "pyyaml>=6",
        "safetensors>=0.4",
        "regex>=2023",
    )
    .env({"HF_HOME": f"{WEIGHTS}/hf", "TRIGON_WEIGHTS_CACHE": f"{WEIGHTS}/backbones"})
    .add_local_dir(
        REPO,
        "/root/trigon",
        # The venv and the corpus cache are large and rebuilt inside the job;
        # reports come back through the Volume rather than riding along.
        ignore=["**/.venv/**", "**/corpora/**", "**/.git/**", "**/__pycache__/**", "**/*.pt"],
    )
)

app = modal.App(APP_NAME)
runs = modal.Volume.from_name("trigon-runs", create_if_missing=True)
weights = modal.Volume.from_name("trigon-weights", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 20,
    volumes={RUNS: runs, WEIGHTS: weights},
)
def train_one(corpus: str, seed: int, flags: list[str], run: dict) -> dict:
    """One seed, one container. Writes its reports to the Volume and returns them."""
    import torch

    root = pathlib.Path("/root/trigon")
    stem = f"{run['prefix']}-seed{seed}"
    out = root / "reports" / corpus / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    target = pathlib.Path(RUNS) / run["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    log_path = target / f"seed{seed}.log"

    # A checkpoint written inside the container dies with it, and one path
    # shared by four seeds would be one file. Any --save-model is redirected
    # to this seed's own path on the Volume, which is the only place a trained
    # model outlives the job that trained it.
    flags = list(flags)
    if "--save-model" in flags:
        flags[flags.index("--save-model") + 1] = str(target / f"seed{seed}.pt")

    started = time.time()
    # Streamed to the Volume as it runs, so `status` can show a live epoch
    # line instead of nothing until the seed returns.
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "scripts/train_corpus.py",
                corpus,
                "--seed",
                str(seed),
                "--out",
                str(out),
                "--device",
                "cuda",
                *flags,
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_commit = time.time()
        while process.poll() is None:
            time.sleep(15)
            if time.time() - last_commit > 120:
                log.flush()
                runs.commit()
                last_commit = time.time()
    elapsed = time.time() - started
    weights.commit()

    payload = {
        "seed": seed,
        "returncode": process.returncode,
        "elapsed_s": round(elapsed, 1),
        # Recorded, not assumed: a report whose hardware and commit are
        # unknown is not reproducible, and this is the only place that
        # information exists.
        **{k: run[k] for k in ("run_id", "prefix", "commit", "dirty", "flags")},
        "gpu_requested": run["gpu"],
        "gpu_actual": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "torch": torch.__version__,
        "files": {},
    }
    for path in sorted(out.parent.glob(f"{stem}*")):
        if path.suffix in {".md", ".json"}:
            payload["files"][path.name] = path.read_text()
    (target / f"seed{seed}.json").write_text(json.dumps(payload))
    runs.commit()
    return payload


# -- the local side ---------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _running_containers() -> int:
    try:
        out = subprocess.run(
            ["modal", "container", "list", "--json"], capture_output=True, text=True, check=True
        ).stdout
        return len(json.loads(out))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return 0


def launch(args) -> None:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not args.allow_dirty:
        raise SystemExit(
            "the working tree has uncommitted changes, and the image is built from the "
            "tree rather than the commit -- so the SHA recorded would name code that did "
            "not run. Commit first, or pass --allow-dirty and it will be recorded."
        )
    flags = [
        "-n",
        str(args.n),
        "--epochs",
        str(args.epochs),
        "--calibration-n",
        str(args.calibration_n),
        "--log-every",
        "200",
        # 8 x 2,500^2. HelpSteer2's longest case compiles to 7,171 tokens and
        # a chunk of eight at that width asked a 22 GiB A10 for 6.13 GiB in
        # one allocation. Splitting accumulates into the same update.
        "--max-batch-cells",
        str(args.max_batch_cells),
        *(args.extra.split() if args.extra else []),
    ]
    prefix = args.prefix or f"modal-n{args.n}-e{args.epochs}"
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    run_id = f"{args.corpus}-{prefix}-{commit[:12]}-{stamp}"
    run = {
        "run_id": run_id,
        "prefix": prefix,
        "commit": commit,
        "dirty": dirty,
        "gpu": args.gpu,
        "flags": flags,
    }
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    busy = _running_containers()
    if busy + len(seeds) > GPU_LIMIT:
        print(
            f"note: {busy} GPU containers already running and the plan allows "
            f"{GPU_LIMIT}; {busy + len(seeds) - GPU_LIMIT} of these seeds will queue "
            "until earlier ones finish"
        )
    with modal.enable_output():
        app.deploy(name=APP_NAME)
    job = modal.Function.from_name(APP_NAME, "train_one").with_options(gpu=args.gpu)
    calls = {seed: job.spawn(args.corpus, seed, flags, run).object_id for seed in seeds}

    print(f"{args.corpus}: seeds {seeds}, gpu={args.gpu}, commit {commit[:12]}")
    print(f"flags: {' '.join(flags)}")
    print(f"run id {run_id}")
    print(json.dumps({"run_id": run_id, "calls": calls}))
    print(f"collect with: python scripts/modal_train.py collect {run_id}")


def status(args) -> None:
    try:
        entries = sorted(entry.path for entry in runs.listdir(args.run_id))
    except Exception:  # noqa: BLE001 - the SDK raises a bare not-found here
        entries = []
    done = [e for e in entries if e.endswith(".json")]
    print(f"{args.run_id}: {len(done)} seed(s) finished")
    for name in (e for e in entries if e.endswith(".log")):
        text = b"".join(runs.read_file(name)).decode(errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        print(f"  {pathlib.Path(name).name}: {lines[-1][:140] if lines else '(empty)'}")


def _write(payloads: list[dict], destination: pathlib.Path, prefix: str) -> None:
    """Reports and the run record, named for the run so two runs cannot collide."""
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for payload in sorted(payloads, key=lambda p: p["seed"]):
        for name, text in payload["files"].items():
            (destination / name).write_text(text)
        summary.append({k: v for k, v in payload.items() if k != "files"})
        state = "ok" if payload["returncode"] == 0 else f"EXIT {payload['returncode']}"
        print(
            f"  seed {payload['seed']}: {state} in {payload['elapsed_s']}s on "
            f"{payload['gpu_actual']}, {len(payload['files'])} files"
        )
    record = destination / f"{prefix}-modal-run.json"
    record.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {destination}/ and {record.name}")
    print("A non-zero exit is a failed gate, which is a result. Read the reports.")


def collect(args) -> None:
    names = [entry.path for entry in runs.listdir(args.run_id)]
    payloads = [
        json.loads(b"".join(runs.read_file(name)).decode())
        for name in names
        if name.endswith(".json")
    ]
    if not payloads:
        raise SystemExit(f"nothing on the Volume under {args.run_id} yet")
    corpus = args.run_id.split("-", 1)[0]
    prefix = payloads[0]["prefix"]
    print(f"{args.run_id}: {len(payloads)} seed(s) finished")
    destination = pathlib.Path(args.out_dir) if args.out_dir else REPO / "reports" / corpus
    _write(payloads, destination, prefix)
    if args.models:
        # Checkpoints are git-ignored: they land beside the reports for
        # `trigon serve --weights`, and never in a commit.
        for name in sorted(n for n in names if n.endswith(".pt")):
            target = destination / f"{prefix}-{pathlib.Path(name).stem}.pt"
            with target.open("wb") as out:
                for chunk in runs.read_file(name):
                    out.write(chunk)
            print(f"  model {target} ({target.stat().st_size / 2**20:.0f} MiB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("launch")
    go.add_argument("--corpus", default="helpsteer2")
    go.add_argument("--seeds", default="0,1,2,3")
    go.add_argument("-n", "--n", type=int, default=12000)
    go.add_argument("--epochs", type=int, default=6)
    go.add_argument("--calibration-n", type=int, default=1000)
    go.add_argument("--gpu", default="A10G")
    go.add_argument("--extra", default="", help="more train_corpus.py flags, quoted")
    go.add_argument("--prefix", default="")
    go.add_argument("--max-batch-cells", type=int, default=50_000_000)
    go.add_argument("--allow-dirty", action="store_true")
    for name in ("status", "collect"):
        command = sub.add_parser(name)
        command.add_argument("run_id")
        if name == "collect":
            command.add_argument("--out-dir", default="")
            command.add_argument(
                "--models", action="store_true", help="also download seed checkpoints"
            )
    args = parser.parse_args(argv)
    {"launch": launch, "status": status, "collect": collect}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
