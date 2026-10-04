#!/usr/bin/env python
"""Turn a training run into a published bundle: files, model card, checksums.

    python scripts/publish_bundle.py runs/broad-q3-06b-s0 --name broad-q3-06b --version v1 \\
        --generality reports/generality/broad-q3-06b --webact reports/webact/broad-q3-06b \\
        --out dist/broad-q3-06b/v1 [--r2 trigon-models] [--modal-volume trigon-runs]

A bundle is what `docker/entrypoint.py`, `trigon serve` and the hosted gateway
load: ``adapter.pt``, the calibrators that were accepted, ``mix.json`` (what it
was trained on and what it was not), a ``README.md`` model card, and
``SHA256SUMS`` over every other file. The card is generated, not written: the
corpora, their licences and attribution come from the corpus specs the run
recorded, and the numbers from the evaluation results passed in, so it can say
nothing the files do not.

**A published name is immutable.** ``bundles/<name>/<version>/`` is never
overwritten -- the Worker serves it with an immutable cache header, and a
caller pinned to it is entitled to the same bytes forever. A new model is a
new version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from trigon.backends.hub import BACKBONES  # noqa: E402
from trigon.evals.corpora import corpus  # noqa: E402

FILES = ("adapter.pt", "temperatures.json", "isotonic.json", "mix.json")


def _sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _table(results: dict, columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| Task | " + " | ".join(h for h, _ in columns) + " |"]
    lines.append("| --- |" + " ---: |" * len(columns))
    for name, row in results.items():
        if not isinstance(row, dict) or not row.get("n"):
            continue
        cells = []
        for _, key in columns:
            value = row.get(key)
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value or ""))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def card(name: str, version: str, mix: dict, generality: dict | None, webact: dict | None) -> str:
    backbone = BACKBONES[mix["backbone"]]
    lines = [
        f"# trigon {name} {version}",
        "",
        "A typed-decision model: send state and a map of typed questions (choice, yes/no,",
        "score), get one calibrated distribution per question in a single prefill pass.",
        "Served by the trigon gateway (`docs/self-host.md`).",
        "",
        "## What it is",
        "",
        f"- **Backbone**: `{backbone.repo}` at `{backbone.revision}` ({backbone.licence}),",
        "  frozen; fetched from its own repository, never re-published here.",
        f"- **This bundle**: LoRA rank {mix.get('lora_rank', 16)} and readout heads"
        " (`adapter.pt`),",
        f"  build `{mix['model_version']}`, Choice head crossover"
        f" {mix.get('option_crossover', 64)}.",
        f"- **Trained**: {mix['epochs']} epochs, lr {mix['lr']}, seed {mix['seed']},"
        f" on {mix['device']} in {mix['train_seconds'] / 3600:.1f} h.",
        "- **Licence**: Apache-2.0 for this bundle. Every training corpus is green-tier",
        "  (`docs/data.md`); share-alike and research-only corpora evaluate only and never train.",
        "",
        "## Trained on",
        "",
        "| Corpus | Cases | Calibration cases | Licence | Attribution |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for corpus_name, counts in mix["corpora"].items():
        spec = corpus(corpus_name)
        lines.append(
            f"| {corpus_name} | {counts['train']:,} | {counts['calibration']:,} | "
            f"{spec.licence} | {spec.attribution} |"
        )
    held = mix.get("held_out", {})
    lines += [
        "",
        "**Held out**: corpora "
        + ", ".join(f"`{c}`" for c in held.get("corpora", []))
        + "; teacher domains "
        + ", ".join(f"`{d}`" for d in held.get("teacher_domains", []))
        + "."
        + (
            " Every website in the web-action evaluation is excluded from `mind2web-train`."
            if "mind2web-train" in mix["corpora"]
            else ""
        ),
        "",
        "**Calibration** was fitted on "
        f"{mix['calibrated_on']['cases']:,} held-out cases from "
        + ", ".join(f"`{c}`" for c in mix["calibrated_on"]["corpora"])
        + ", with option sets reshaped as training reshapes them. Teacher labels are never "
        "calibration evidence.",
        "",
        "## Evaluation",
        "",
    ]
    if generality:
        lines += [
            "Generality (`scripts/generality.py`, 1,000 cases per task, fixed seed):",
            "",
            *_table(
                generality["tasks"],
                [
                    ("Accuracy", "accuracy"),
                    ("Chance", "chance"),
                    ("ECE", "ece"),
                    ("ECE floor p95", "ece_floor_p95"),
                    ("Order agreement", "agreement"),
                ],
            ),
            "",
        ]
    if webact:
        w = webact["webact"]
        lines += [
            "Web actions (`scripts/webact.py`, Mind2Web, websites unseen in training):",
            "",
            "| Steps | Operation | Element | Step success | ECE (floor p95) |",
            "| ---: | ---: | ---: | ---: | --- |",
            f"| {w['n']} | {w['operation_accuracy']:.3f} | {w['target_accuracy']:.3f} | "
            f"{w['step_success']:.3f} | {w['executed_confidence_ece']:.3f} "
            f"({w['executed_confidence_ece_floor_p95']:.3f}) |",
            "",
        ]
    lines += [
        "Single seed: a measurement, not a certification (`CLAUDE.md`).",
        "",
        "## Use it",
        "",
        "```bash",
        f"curl -LO <host>/models/bundles/{name}/{version}/adapter.pt  # + SHA256SUMS files",
        "sha256sum -c SHA256SUMS",
        "trigon serve --backend torch --weights adapter.pt  # TRIGON_*_PATH for the calibrators",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--generality", type=pathlib.Path, default=None)
    parser.add_argument("--webact", type=pathlib.Path, default=None)
    parser.add_argument("--r2", default=None, help="R2 bucket to upload to, via wrangler")
    parser.add_argument("--modal-volume", default=None, help="Modal volume to upload to")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; a published version is never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    present = [f for f in FILES if (args.run / f).exists()]
    if "adapter.pt" not in present or "mix.json" not in present:
        raise SystemExit(f"{args.run} has no adapter.pt and mix.json; not a train_mix.py run")
    for f in present:
        shutil.copy2(args.run / f, args.out / f)
    mix = json.loads((args.out / "mix.json").read_text())
    mix["argv"] = [a if not a.startswith("/") else "<run>" for a in mix.get("argv", [])]
    (args.out / "mix.json").write_text(json.dumps(mix, indent=2))
    generality = (
        json.loads((args.generality / "results.json").read_text()) if args.generality else None
    )
    webact = json.loads((args.webact / "results.json").read_text()) if args.webact else None
    (args.out / "README.md").write_text(card(args.name, args.version, mix, generality, webact))
    sums = [f"{_sha(args.out / f)}  {f}" for f in sorted(p.name for p in args.out.iterdir())]
    (args.out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print((args.out / "SHA256SUMS").read_text(), end="")

    prefix = f"bundles/{args.name}/{args.version}"
    if args.r2:
        for path in sorted(args.out.iterdir()):
            kind = "text/markdown" if path.suffix == ".md" else None
            command = [
                "npx",
                "wrangler",
                "r2",
                "object",
                "put",
                f"{args.r2}/{prefix}/{path.name}",
                "--file",
                str(path),
                "--remote",
            ]
            if kind:
                command += ["--content-type", kind]
            subprocess.run(command, check=True, capture_output=True)
            print(f"r2: {args.r2}/{prefix}/{path.name}")
    if args.modal_volume:
        for path in sorted(args.out.iterdir()):
            subprocess.run(
                [
                    "modal",
                    "volume",
                    "put",
                    args.modal_volume,
                    str(path),
                    f"bundles/{args.name}-{args.version}/{path.name}",
                ],
                check=True,
                capture_output=True,
            )
        print(f"modal: {args.modal_volume}/bundles/{args.name}-{args.version}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
