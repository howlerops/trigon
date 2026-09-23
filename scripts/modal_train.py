"""Run a corpus training sweep on a Modal GPU, and bring the reports back.

    modal run --detach scripts/modal_train.py --corpus helpsteer2 --n 12000 --epochs 6
    modal run scripts/modal_train.py::collect --run-id <printed by the launch>

`docs/gpu-access.md` picked this route, and the reason was durability before
speed: three HelpSteer2 attempts died in this sandbox, one to an
out-of-memory kill and two to the session VM being reclaimed mid-run.

**Durability is the Volume, not the remote container.** The first version of
this returned reports as function return values to the local entrypoint, so
the job did run on Modal's infrastructure -- and its results still lived only
in the process on the VM that gets reclaimed. Without `--detach`, `modal run`
also stops the app when that process dies. Each seed now writes its reports to
the `trigon-runs` Volume before returning, so a `--detach`ed run survives this
session and `collect` fetches it on a later turn. When the launching process
does survive, it collects on its own.

**Each seed is its own container.** Modal fans them out, so four seeds cost
the wall clock of one rather than four-on-four-cores. That is also why this
does not reuse `scripts/seed_sweep.py`, which parallelises with local
processes.

## What it is careful about

**The commit is recorded, and it has to mean something.** The image is built
from the working tree, not from the commit, so a SHA recorded over a dirty
tree names code that did not run. The launch refuses a dirty tree unless
`--allow-dirty` is passed, and then records that it was.

**The GPU is used, not just recorded.** The job passes `--device cuda` to
`train_corpus.py`, which refuses to fall back to the CPU. The first version
recorded the GPU the container was given while nothing in the trainer ever
moved a tensor off the CPU -- a report naming hardware it never used.

**`--gpu` is honoured.** It was recorded as "requested" while the decorator
hard-coded an A10G.

**The corpus is downloaded inside the job**, not shipped from here. It is
someone else's data under a licence that governs redistribution, and
`trigon.evals.corpora` already fetches to an ignored cache with the
attribution attached.

**Nothing is certified by this script.** It runs seeds and returns reports;
whether they certify is what the gates say, and `CLAUDE.md`'s rule about the
median and the spread applies exactly as it does locally.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = "/runs"

# The training extra plus a CUDA torch. The default PyPI wheel carries CUDA on
# Linux, so nothing here pins an index -- pinning one is how a CPU wheel ends
# up on a GPU box and the job runs correctly and slowly.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "numpy>=1.26", "pydantic>=2", "fastapi>=0.110", "pyyaml>=6")
    .add_local_dir(
        REPO,
        "/root/trigon",
        # The venv and the corpus cache are large and rebuilt inside the job;
        # reports come back through the Volume rather than riding along.
        ignore=["**/.venv/**", "**/corpora/**", "**/.git/**", "**/__pycache__/**", "**/*.pt"],
    )
)

app = modal.App("trigon-train")
volume = modal.Volume.from_name("trigon-runs", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 12, volumes={RUNS: volume})
def train_one(corpus: str, seed: int, flags: list[str], run: dict) -> dict:
    """One seed, one container. Writes its reports to the Volume and returns them."""
    import torch

    sys.path.insert(0, "/root/trigon/src")
    root = pathlib.Path("/root/trigon")
    stem = f"{run['prefix']}-seed{seed}"
    out = root / "reports" / corpus / f"{stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    result = subprocess.run(
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
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started

    payload = {
        "seed": seed,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        # Recorded, not assumed: a report whose hardware and commit are
        # unknown is not reproducible, and this is the only place that
        # information exists.
        **{k: run[k] for k in ("run_id", "commit", "dirty", "flags")},
        "gpu_requested": run["gpu"],
        "gpu_actual": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "torch": torch.__version__,
        "files": {},
    }
    for path in sorted(out.parent.glob(f"{stem}*")):
        if path.suffix in {".md", ".json"}:
            payload["files"][path.name] = path.read_text()

    # Written before returning, so a detached run whose launcher died still
    # has its results somewhere `collect` can reach.
    target = pathlib.Path(RUNS) / run["run_id"]
    target.mkdir(parents=True, exist_ok=True)
    (target / f"seed{seed}.json").write_text(json.dumps(payload))
    (target / f"seed{seed}.log").write_text(result.stdout[-200_000:] + result.stderr[-200_000:])
    volume.commit()
    return payload


def _write(payloads: list[dict], destination: pathlib.Path) -> None:
    """Reports and `modal-run.json`, from whichever route brought them back."""
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for payload in sorted(payloads, key=lambda p: p["seed"]):
        for name, text in payload["files"].items():
            (destination / name).write_text(text)
        summary.append({k: v for k, v in payload.items() if k != "files"})
        status = "ok" if payload["returncode"] == 0 else f"EXIT {payload['returncode']}"
        print(
            f"  seed {payload['seed']}: {status} in {payload['elapsed_s']}s on "
            f"{payload['gpu_actual']}, {len(payload['files'])} files"
        )
    (destination / "modal-run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {destination}/ and modal-run.json")
    print("A non-zero exit is a failed gate, which is a result. Read the reports.")


@app.local_entrypoint()
def main(
    corpus: str = "helpsteer2",
    seeds: str = "0,1,2,3",
    n: int = 12000,
    epochs: int = 6,
    calibration_n: int = 1000,
    gpu: str = "A10G",
    extra: str = "",
    prefix: str = "",
    out_dir: str = "",
    allow_dirty: bool = False,
) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise SystemExit(
            "the working tree has uncommitted changes, and the image is built from the "
            "tree rather than the commit -- so the SHA recorded would name code that did "
            "not run. Commit first, or pass --allow-dirty and it will be recorded."
        )

    flags = [
        "-n",
        str(n),
        "--epochs",
        str(epochs),
        "--calibration-n",
        str(calibration_n),
        "--log-every",
        "200",
        *(extra.split() if extra else []),
    ]
    prefix = prefix or f"modal-n{n}-e{epochs}"
    run_id = f"{corpus}-{prefix}-{commit[:12]}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    run = {
        "run_id": run_id,
        "prefix": prefix,
        "commit": commit,
        "dirty": dirty,
        "gpu": gpu,
        "flags": flags,
    }
    wanted = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"{corpus}: seeds {wanted}, n={n}, epochs={epochs}, gpu={gpu}, commit {commit[:12]}")
    print(f"run id {run_id}")
    print(f"if this process dies: modal run scripts/modal_train.py::collect --run-id {run_id}")

    job = train_one.with_options(gpu=gpu)
    payloads = list(job.starmap([(corpus, seed, flags, run) for seed in wanted]))
    _write(payloads, pathlib.Path(out_dir) if out_dir else REPO / "reports" / corpus)


@app.local_entrypoint()
def collect(run_id: str, out_dir: str = "") -> None:
    """Fetch a run's reports from the Volume -- the route that survives this VM."""
    names = [entry.path for entry in volume.listdir(run_id)]
    payloads = [
        json.loads(b"".join(volume.read_file(name)).decode())
        for name in names
        if name.endswith(".json")
    ]
    if not payloads:
        raise SystemExit(f"nothing on the Volume under {run_id} yet; seeds write when they finish")
    corpus = run_id.split("-", 1)[0]
    print(f"{run_id}: {len(payloads)} seed(s) finished")
    _write(payloads, pathlib.Path(out_dir) if out_dir else REPO / "reports" / corpus)
