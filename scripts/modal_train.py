"""Run a corpus training sweep on a Modal GPU, and bring the reports back.

    modal run scripts/modal_train.py --corpus helpsteer2 --n 12000 --epochs 6
    modal run scripts/modal_train.py --corpus banking77 --gpu L4

`docs/gpu-access.md` picked this route, and the reason was durability before
speed: three HelpSteer2 attempts died in this sandbox, one to an
out-of-memory kill and two to the session VM being reclaimed mid-run. A Modal
job runs on Modal's infrastructure, so the VM here can be reclaimed without
costing anything — the results are collected on a later turn.

**Each seed is its own container.** Modal fans them out, so four seeds cost
the wall clock of one rather than four-on-four-cores. That is also why this
does not reuse `scripts/seed_sweep.py`, which parallelises with local
processes.

## What it is careful about

**The commit is recorded in every report.** A number from a GPU nobody can
see is worth less than one from a machine you can name, so each run writes
the git SHA it was built from into `<out>-modal.json` alongside the GPU type
and the wall clock. A report that cannot be tied to a commit is a claim.

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

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

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
        # reports come back as return values rather than riding along.
        ignore=["**/.venv/**", "**/corpora/**", "**/.git/**", "**/__pycache__/**", "**/*.pt"],
    )
)

app = modal.App("trigon-train")


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 6)
def train_one(corpus: str, seed: int, flags: list[str], commit: str, gpu: str) -> dict:
    """One seed, one container. Returns the reports as text."""
    import time

    sys.path.insert(0, "/root/trigon/src")
    root = pathlib.Path("/root/trigon")
    out = root / "reports" / corpus / f"modal-seed{seed}.md"
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
            *flags,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started

    import torch

    payload = {
        "seed": seed,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "elapsed_s": round(elapsed, 1),
        # Recorded, not assumed: a report whose hardware and commit are
        # unknown is not reproducible, and this is the only place that
        # information exists.
        "commit": commit,
        "gpu_requested": gpu,
        "gpu_actual": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "cuda_available": torch.cuda.is_available(),
        "files": {},
    }
    for path in sorted(out.parent.glob(f"modal-seed{seed}*")):
        if path.suffix in {".md", ".json"}:
            payload["files"][path.name] = path.read_text()
    return payload


@app.local_entrypoint()
def main(
    corpus: str = "helpsteer2",
    seeds: str = "0,1,2,3",
    n: int = 12000,
    epochs: int = 6,
    calibration_n: int = 1000,
    gpu: str = "A10G",
    extra: str = "",
) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()

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
    wanted = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"{corpus}: seeds {wanted}, n={n}, epochs={epochs}, gpu={gpu}, commit {commit[:12]}")

    results = list(train_one.starmap([(corpus, seed, flags, commit, gpu) for seed in wanted]))

    destination = REPO / "reports" / corpus
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for payload in results:
        seed = payload["seed"]
        for name, text in payload["files"].items():
            (destination / name).write_text(text)
        summary.append(
            {k: v for k, v in payload.items() if k not in {"files", "stdout_tail", "stderr_tail"}}
        )
        status = "ok" if payload["returncode"] == 0 else f"EXIT {payload['returncode']}"
        print(
            f"  seed {seed}: {status} in {payload['elapsed_s']}s on "
            f"{payload['gpu_actual']}, {len(payload['files'])} files"
        )
        if payload["returncode"] != 0:
            print(payload["stderr_tail"][-1500:])

    (destination / "modal-run.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {destination}/modal-seed*.md and modal-run.json")
    print("A non-zero exit is a failed gate, which is a result. Read the reports.")
