"""B.1 on Modal's L4 -- a preliminary number, labelled as one.

    python scripts/modal_burn_in.py            # runs, then writes reports/burn-in/modal-l4/

`docs/gpu-access.md` says B.1 should run on a dedicated, rented L4, and that
is still the measurement the cost argument needs: a serverless container on a
shared host is not the node the cost model assumes, and Modal's per-second
price is not a rental price. The owner chose to take a preliminary reading
here first (2026-09-25). Everything below keeps it from passing for the real
one: the report is written under `modal-l4/`, its header says preliminary,
and it records the commit and whether the tree was clean the way the training
runs do.

It times the shapes `scripts/burn_in.py` always timed and, beside them, the
certified Banking77 adapter itself, read from the Volume where training wrote
it -- the model the deployment serves.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
RUN = "banking77-qwen15b-e4-lr1e-4-872ed726edcd-20260923T145427"
# Modal's published L4 rate, per second, as an hourly figure. An input, not a
# measurement, and a serverless price rather than a rental one.
MODAL_L4_USD_PER_HOUR = 0.80


def _git(*args: str) -> str:
    # Modal imports this file again inside the container, which has no git and
    # no .git; there the image's environment already carries the answer.
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


COMMIT = _git("rev-parse", "HEAD") or os.environ.get("TRIGON_COMMIT", "unknown")
# Tracked changes only: an untracked file is not part of the commit named.
DIRTY = bool(_git("status", "--porcelain", "--untracked-files=no"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2", "numpy>=1.26", "pydantic>=2", "pyyaml>=6", "safetensors>=0.4", "regex>=2023"
    )
    .env(
        {
            "TRIGON_WEIGHTS_CACHE": "/weights/backbones",
            "PYTHONPATH": "/root/trigon/src",
            "TRIGON_COMMIT": COMMIT,
            "TRIGON_DIRTY": "1" if DIRTY else "0",
        }
    )
    .add_local_dir(
        REPO,
        "/root/trigon",
        ignore=["**/.venv/**", "**/corpora/**", "**/.git/**", "**/__pycache__/**", "**/*.pt"],
    )
)
app = modal.App("trigon-burn-in")
runs = modal.Volume.from_name("trigon-runs")
weights = modal.Volume.from_name("trigon-weights")


@app.function(image=image, gpu="L4", volumes={"/runs": runs, "/weights": weights}, timeout=3600)
def burn_in() -> dict[str, str]:
    import pathlib as p

    out = p.Path("/tmp/burn-in")
    subprocess.run(
        [
            sys.executable,
            "/root/trigon/scripts/burn_in.py",
            "--weights",
            f"/runs/{RUN}/seed2.pt",
            "--usd-per-hour",
            str(MODAL_L4_USD_PER_HOUR),
            "--out",
            str(out),
        ],
        check=True,
        env=dict(os.environ),
    )
    return {f.name: f.read_text() for f in out.iterdir()}


@app.local_entrypoint()
def main() -> None:
    if DIRTY:
        raise SystemExit("refusing a dirty tree: the report must name a commit")
    files = burn_in.remote()
    out = REPO / "reports" / "burn-in" / "modal-l4"
    out.mkdir(parents=True, exist_ok=True)
    banner = (
        "> **Preliminary: Modal's serverless L4, not a dedicated rental.** "
        "`docs/next.md` B.1 stays open until the same command runs on a rented L4.\n\n"
    )
    for name, text in files.items():
        (out / name).write_text(banner + text if name.endswith(".md") else text)
    print(f"wrote {out}")
