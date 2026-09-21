#!/usr/bin/env python
"""Train one configuration under several seeds, and report the spread.

    python scripts/seed_sweep.py --seeds 0 1 2 3 -n 2500 --epochs 6
    python scripts/seed_sweep.py --seeds 0 1 2 3 --jobs 4 -n 8000 --epochs 8

With ``--jobs`` each seed writes its own ``<name>-seed<N>.log`` beside its
report, so a sweep in flight can be followed with ``tail -f``. Each training
run is single-threaded by design, so ``--jobs`` is cores rather than
oversubscription.

**Why this exists.** The eval harness already refuses to quote an ECE without
simulating what a perfectly calibrated model would score on the same run --
``metrics.noise_floor``, and the rule that a gate which a perfect model fails
half the time is not a test. That discipline was applied to the *measurement*
and never to the *training run*.

It should have been. The committed reference configuration turns out to have
enormous seed variance: at 2,500 cases its first-epoch loss ranges from 0.99 to
1.15 across four seeds, and whether it escapes the marginal at all depends on
which draw you get. The original reference run reported one draw as a result.
A perturbation far smaller than a seed change -- the summation order of a
batched matmul, on the order of 1e-6 -- is enough to move a given seed from one
outcome to the other.

So: **a single-seed run is not evidence that a configuration works.** It is one
sample from a distribution whose width nobody measured. This measures the
width, and what it reports is the median and the range, not the best.

Use it before certifying a configuration, and prefer a configuration whose
spread is narrow over one whose best seed is good.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


#: What ``--require`` accepts, and what each one means for the exit code.
#:
#: ``none`` reports and always exits 0 -- the exploratory mode, and the
#: default, because a sweep used to choose a configuration should not stop at
#: the first one that disappoints.
#:
#: ``all`` is the one to gate on. A configuration certifies only if every seed
#: certifies, which is the rule ``docs/decisions.md`` argues for: a
#: configuration that passes on some draws and fails on others has not been
#: shown to work, and a single run of it reports a coin flip as a result.
#:
#: ``median`` is deliberately absent. The median seed certifying means half the
#: draws do not, and a release gate that half of runs fail is the same broken
#: instrument as an ECE gate below its noise floor.
REQUIREMENTS = ("none", "all")


def verdict(rows: list[dict], require: str) -> tuple[int, str]:
    """The sweep's exit code and the line that explains it.

    Separate from the run loop so the rule can be tested without training
    anything -- the rule is the part that decides whether CI goes red, and it
    would otherwise be reachable only by a twenty-minute job.
    """
    certified = [r for r in rows if r["certified"]]
    if require == "none":
        return 0, ""
    if not rows:
        return 1, "**No seeds ran.** An empty sweep certifies nothing."
    if len(certified) == len(rows):
        return 0, f"**Certified on all {len(rows)} seeds.**"
    blocked = sorted({name for r in rows for name in r["blocked"]})
    return 1, (
        f"**Not certified: {len(certified)} of {len(rows)} seeds passed.** "
        f"Blocked on {', '.join(f'`{name}`' for name in blocked)}. "
        "A configuration that certifies on some draws and not others has not "
        "been shown to work; widen the data or the training budget until the "
        "outcome stops depending on the seed."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out-dir", default="reports/sweeps")
    parser.add_argument("--name", default="sweep")
    parser.add_argument("--keep-reports", action="store_true")
    parser.add_argument(
        "--require",
        choices=REQUIREMENTS,
        default="none",
        help="'all' exits non-zero unless every seed certifies; see REQUIREMENTS",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="seeds to train concurrently; each run is single-threaded, so this is cores",
    )
    known, passthrough = parser.parse_known_args()

    out_dir = pathlib.Path(known.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(seed: int) -> dict:
        report = out_dir / f"{known.name}-seed{seed}.md"
        command = [
            sys.executable,
            "-m",
            "trigon.cli",
            "train",
            "--seed",
            str(seed),
            "--out",
            str(report),
            *passthrough,
        ]
        # Each seed's output goes to its own log rather than the terminal,
        # because several runs interleaving on stderr is unreadable. It has to
        # go *somewhere*: a sweep is tens of minutes, and one with no visible
        # progress is indistinguishable from a hung one -- the same reason the
        # trainer prints per-epoch even when log_every is 0.
        log = out_dir / f"{known.name}-seed{seed}.log"
        with log.open("w") as sink:
            # A failed gate exits non-zero and is a result, not an error: a
            # sweep that stopped at the first seed that did not certify would
            # report only the seeds that worked, which is the bias this exists
            # to remove.
            result = subprocess.run(command, cwd=ROOT, stdout=sink, stderr=sink, check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"seed {seed} failed to run ({result.returncode}); see {log}")

        sidecar = json.loads(pathlib.Path(f"{report.with_suffix('')}-training.json").read_text())
        gates = json.loads(report.with_suffix(".json").read_text())["gates"]
        by_name = {g["name"]: g for g in gates}
        blocked = [g["name"] for g in gates if not g["passed"] and not g.get("advisory")]
        if not known.keep_reports:
            for extra in out_dir.glob(f"{known.name}-seed{seed}*"):
                if extra.suffix in {".pt", ".log"}:
                    extra.unlink()
        return {
            "seed": seed,
            "final_loss": sidecar["final_loss"],
            "kept_epoch": sidecar.get("kept_epoch"),
            "lift": by_name.get("accuracy_over_baseline", {}).get("value"),
            "ece": by_name.get("workhorse_ece", {}).get("value"),
            "certified": not blocked,
            "blocked": blocked,
        }

    rows = []
    if known.jobs > 1:
        # Each training run is single-threaded on purpose (TrainingConfig
        # explains why), so concurrency here is cores, not oversubscription.
        with concurrent.futures.ThreadPoolExecutor(max_workers=known.jobs) as pool:
            futures = {pool.submit(run_one, seed): seed for seed in known.seeds}
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                print(f"— seed {futures[future]} done", file=sys.stderr, flush=True)
        rows.sort(key=lambda r: r["seed"])
    else:
        for seed in known.seeds:
            print(f"— seed {seed}", file=sys.stderr, flush=True)
            rows.append(run_one(seed))

    print()
    print("| Seed | Final loss | Kept epoch | Lift over baseline | ECE | Certified | Blocked on |")
    print("| ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        lift = "—" if row["lift"] is None else f"{row['lift']:+.4f}"
        ece = "—" if row["ece"] is None else f"{row['ece']:.4f}"
        blocked = ", ".join(f"`{name}`" for name in row["blocked"]) or "—"
        print(
            f"| {row['seed']} | {row['final_loss']:.4f} | {row['kept_epoch']} | "
            f"{lift} | {ece} | {'**yes**' if row['certified'] else 'no'} | {blocked} |"
        )

    losses = [r["final_loss"] for r in rows]
    certified = sum(r["certified"] for r in rows)
    print()
    print(
        f"final loss: median {statistics.median(losses):.4f}, "
        f"range {min(losses):.4f}–{max(losses):.4f}, "
        f"spread {max(losses) - min(losses):.4f}"
    )
    print(f"certified: {certified} of {len(rows)} seeds")
    if certified not in (0, len(rows)):
        print(
            "\n**This configuration does not certify reliably.** Some seeds pass the "
            "gates and some do not, so a single-seed run of it reports a coin flip "
            "as a result. Widen the data or the training budget until the outcome "
            "stops depending on the draw."
        )
    code, line = verdict(rows, known.require)
    if line:
        print(f"\n{line}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
