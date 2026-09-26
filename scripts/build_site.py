#!/usr/bin/env python
"""Render the GitHub Pages site into ``_site/``.

    pip install -e ".[site]"
    python scripts/build_site.py            # writes _site/
    python scripts/build_site.py --out /tmp/site
    python -m http.server -d _site 8000     # and look at it

The hand-written half lives in ``site/``: the stylesheets, the fonts and the
page layout. Everything else is generated here, from the files that already
hold each fact, so the site is one more reader of the repository rather than
a second copy of it (``CLAUDE.md``: one source of truth per fact):

- **Docs** are ``docs/*.md`` and ``README.md``, rendered with a static sidebar.
  A link to another doc goes to its page; a link or a code span naming any
  other file in the repository goes to that file on GitHub, at ``BRANCH``.
- **Examples** import ``trigon.usecases`` and answer each request through the
  lexical floor, so the response shown is what the engine really returns --
  and is labelled as the floor's, because its numbers are not a model's.
- **The API reference** is read from ``spec/openapi.json``.
- **Proofs** reads every certified run's per-seed JSON under ``reports/``,
  draws reliability diagrams from the bins those files carry, and lifts the
  ledger's *Believed, then disproved* and *Corrected in our own favour*
  sections as they stand.
- **The landing page's numbers** are computed from those same report files.
  None is typed here, and ``tests/test_site.py`` recomputes them.

Pure Python plus ``markdown``, which sits behind the ``site`` extra. Nothing
under ``src/`` imports this, so ``import trigon`` never needs it.
"""

from __future__ import annotations

import argparse
import ast
import html
import io
import json
import keyword
import posixpath
import re
import shutil
import statistics
import string
import sys
import tokenize
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SITE_SRC = ROOT / "site"
sys.path.insert(0, str(ROOT / "src"))

from trigon import limits  # noqa: E402

# ---------------------------------------------------------------------------
# Where links to the repository point. One constant: change the branch here.

REPO = "howlerops/trigon"
BRANCH = "main"
GITHUB = f"https://github.com/{REPO}"


def blob_url(path: str) -> str:
    """A repository file on GitHub, or a directory's tree view."""
    kind = "tree" if (ROOT / path).is_dir() else "blob"
    return f"{GITHUB}/{kind}/{BRANCH}/{path.rstrip('/')}"


# ---------------------------------------------------------------------------
# The docs, in sidebar order.


@dataclass(frozen=True)
class Doc:
    slug: str
    source: str
    label: str
    #: The page's <h1>; defaults to the source's own first heading.
    title: str | None = None

    @property
    def output(self) -> str:
        return f"docs/{self.slug}.html"


DOC_GROUPS: tuple[tuple[str, tuple[Doc, ...]], ...] = (
    (
        "Getting started",
        (
            Doc("index", "README.md", "Introduction", title="Introduction"),
            Doc("model-card", "docs/model-card.md", "Model card"),
        ),
    ),
    (
        "The model and contract",
        (
            Doc("architecture", "docs/architecture.md", "Architecture"),
            Doc("compat", "docs/compat.md", "Drop-in compatibility"),
        ),
    ),
    ("Calibration and evals", (Doc("evals", "docs/evals.md", "Evals and release gates"),)),
    (
        "Data and training",
        (
            Doc("data", "docs/data.md", "Data and licence audit"),
            Doc("training", "docs/training.md", "Training"),
        ),
    ),
    (
        "Running it",
        (
            Doc("gpu-access", "docs/gpu-access.md", "Getting onto a GPU"),
            Doc("security", "docs/security.md", "Security and going public"),
            Doc("pricing", "docs/pricing.md", "Pricing a use case"),
        ),
    ),
    (
        "The honest record",
        (
            Doc("ledger", "docs/ledger.md", "Ledger"),
            Doc("decisions", "docs/decisions.md", "Decisions"),
            Doc("next", "docs/next.md", "The next plan"),
            Doc("roadmap", "docs/roadmap.md", "Roadmap"),
            Doc("plan", "docs/plan.md", "The plan that got here"),
        ),
    ),
)

#: docs/*.md files deliberately given no page, each with the reason.
#: ``tests/test_site.py`` fails when a doc is neither here nor in DOC_GROUPS.
DOCS_EXCLUDED: dict[str, str] = {}

DOCS: tuple[Doc, ...] = tuple(d for _, group in DOC_GROUPS for d in group)
DOC_BY_SOURCE = {d.source: d for d in DOCS}

#: The non-doc pages that share the docs shell, in sidebar order.
EVIDENCE_PAGES = (
    ("examples/index.html", "Examples"),
    ("examples/api.html", "API reference"),
    ("proofs/index.html", "Proofs"),
)


# ---------------------------------------------------------------------------
# The runs the site quotes. Which files are the certified ones is decided in
# each directory's README and in docs/ledger.md; this names them and the
# role those documents give them. Every number is read from the files.


@dataclass(frozen=True)
class RunSpec:
    key: str
    title: str
    directory: str
    prefix: str
    readme: str
    backbone: str
    #: What the README says this run is. A word, not a number.
    role: str
    summary: str
    suffix: str = ""
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    #: Scored against one annotator drawn per case (decision Q20): the
    #: accuracy gates are advisory there and ``brier_over_marginal`` blocks.
    drawn_annotator: bool = False

    def path(self, seed: int) -> str:
        return f"{self.directory}/{self.prefix}-seed{seed}{self.suffix}.json"

    def report_md(self, seed: int) -> str:
        return f"{self.directory}/{self.prefix}-seed{seed}{self.suffix}.md"

    @property
    def modal_run(self) -> str:
        return f"{self.directory}/{self.prefix}-modal-run.json"


QWEN = "Qwen2.5-1.5B, LoRA rank 16"
SPIKE = "reference spike, d_model 128, 2 layers"

RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        key="banking77-qwen",
        title="Banking77 on Qwen2.5-1.5B",
        directory="reports/banking77",
        prefix="qwen15b-e4-lr1e-4-regate080",
        readme="reports/banking77/README.md",
        backbone=QWEN,
        role="certified",
        summary=(
            "The certified configuration (lr 1e-4, four epochs), re-gated with no "
            "retraining under the current calibrator rule, `ACCEPT_CONFIDENCE` 0.80. "
            "Seed 2 is the deployed model."
        ),
    ),
    RunSpec(
        key="banking77-qwen-as-trained",
        title="Banking77 on Qwen2.5-1.5B, as first gated",
        directory="reports/banking77",
        prefix="qwen15b-e4-lr1e-4",
        readme="reports/banking77/README.md",
        backbone=QWEN,
        role="certified",
        summary=(
            "The same four checkpoints under the calibrator rule they were trained "
            "with (0.95). Kept because the re-gate is only evidence beside it."
        ),
    ),
    RunSpec(
        key="banking77-qwen-lr3e-4",
        title="Banking77 on Qwen2.5-1.5B at lr 3e-4",
        directory="reports/banking77",
        prefix="qwen15b-e4",
        readme="reports/banking77/README.md",
        backbone=QWEN,
        role="did not certify",
        summary=(
            "The spike's configuration with only the model changed. One seed of four "
            "collapsed to chance at the peak learning rate, so the configuration does "
            "not certify, whatever the other three scored."
        ),
    ),
    RunSpec(
        key="synthetic-qwen",
        title="The synthetic suite on Qwen2.5-1.5B",
        directory="reports/synthetic",
        prefix="qwen15b-synthetic-e8",
        readme="reports/synthetic/README.md",
        backbone=QWEN,
        role="certified on the median",
        summary=(
            "The generator's three questions, `size` included. The per-primitive gate "
            "became blocking for backbone runs after these ran; the README records "
            "which seeds clear it."
        ),
    ),
    RunSpec(
        key="helpsteer2-annotators",
        title="HelpSteer2 annotator distributions on Qwen2.5-1.5B",
        directory="reports/helpsteer2-annotators",
        prefix="qwen15b-annotators-e3-lr1e-4",
        readme="reports/helpsteer2-annotators/README.md",
        backbone=QWEN,
        role="certified",
        drawn_annotator=True,
        summary=(
            "Trained on each response's empirical distribution of annotator ratings "
            "and scored against one annotator drawn per pair, so a calibrated model "
            "is one that reproduces how much people disagree."
        ),
    ),
    RunSpec(
        key="helpsteer2-annotators-hard",
        title="HelpSteer2, the majority-vote ablation",
        directory="reports/helpsteer2-annotators",
        prefix="qwen15b-annotators-e3-lr1e-4-hard",
        readme="reports/helpsteer2-annotators/README.md",
        backbone=QWEN,
        role="ablation",
        drawn_annotator=True,
        summary=(
            "Everything identical but the targets: each pair's majority vote instead "
            "of its distribution. Compare the raw ECE column with the run above."
        ),
    ),
    RunSpec(
        key="banking77-spike",
        title="Banking77 on the reference spike",
        directory="reports/banking77",
        prefix="b77",
        readme="reports/banking77/README.md",
        backbone=SPIKE,
        role="certified",
        summary="The CPU-sized model the tests run, on the same splits and gates.",
    ),
    RunSpec(
        key="synthetic-spike",
        title="The synthetic suite on the reference spike",
        directory="reports/iso",
        prefix="iso",
        suffix="-regated",
        readme="reports/iso/README.md",
        backbone=SPIKE,
        role="certified",
        summary="The spike's certified synthetic configuration, re-gated under the current rule.",
    ),
)
RUN_BY_KEY = {r.key: r for r in RUNS}


# ---------------------------------------------------------------------------
# Reading a report.


@dataclass(frozen=True)
class Gate:
    name: str
    value: float
    limit: float
    passed: bool
    advisory: bool
    #: Computed here from the report's own numbers rather than read from it.
    derived: bool = False


@dataclass(frozen=True)
class SeedResult:
    seed: int
    path: str
    model: str
    n: int
    accuracy: float
    baseline: float
    lift: float
    ece: float
    adaptive_ece: float
    brier: float
    floor_mean: float | None
    floor_p95: float | None
    distinguishable: bool | None
    raw_ece: float | None
    raw_accuracy: float | None
    bins: tuple[dict, ...]
    raw_bins: tuple[dict, ...]
    per_question: dict[str, float]
    worst_primitive: tuple[str, float] | None
    gates: tuple[Gate, ...]
    marginal_brier: float | None = None

    @property
    def brier_skill(self) -> float | None:
        if self.marginal_brier is None:
            return None
        return 1.0 - self.brier / self.marginal_brier

    @property
    def blocking_passed(self) -> bool:
        return all(g.passed for g in self.gates if not g.advisory)

    @property
    def every_gate_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if not g.passed]


def _gated_result(data: dict) -> dict:
    """The suite the gates read. Older reports do not name it, so match it."""
    results = data["results"]
    name = data.get("gated")
    if name:
        return next(r for r in results if r["suite"] == name)
    ece_gate = next((g for g in data["gates"] if g["name"] == "workhorse_ece"), None)
    if ece_gate is not None:
        for result in results:
            cal = result.get("calibration") or {}
            if cal.get("ece") is not None and abs(cal["ece"] - ece_gate["value"]) < 1e-12:
                return result
    return results[-1]


_MARGINAL_BRIER = re.compile(r"^\|\s*ignores its input\s*\|\s*([0-9.]+)\s*\|", re.MULTILINE)


def marginal_brier(md_path: str) -> float:
    """The training marginal's Brier, which the report prints and the JSON lacks."""
    text = (ROOT / md_path).read_text()
    found = _MARGINAL_BRIER.search(text)
    if not found:
        raise ValueError(f"{md_path} prints no marginal-predictor Brier")
    return float(found.group(1))


def load_seed(spec: RunSpec, seed: int) -> SeedResult:
    path = spec.path(seed)
    data = json.loads((ROOT / path).read_text())
    gated = _gated_result(data)
    raw = next(
        (r for r in data["results"] if r is not gated and r["suite"].endswith("uncalibrated")),
        None,
    )
    cal = gated["calibration"]
    floor = cal.get("floor") or {}
    prims = gated.get("per_primitive") or {}
    if not prims:
        prims = {
            k.removeprefix("primitive:"): v
            for k, v in (data.get("slices") or {}).items()
            if k.startswith("primitive:")
        }
    worst = max(((k, v["ece"]) for k, v in prims.items()), key=lambda kv: kv[1], default=None)

    gates = [
        Gate(g["name"], g["value"], g["limit"], g["passed"], bool(g.get("advisory")))
        for g in data["gates"]
    ]
    mb = None
    if spec.drawn_annotator:
        # Decision Q20, as `check_gates` applies it: on a drawn-annotator
        # corpus the accuracy gates are advisory and the Brier skill over the
        # training marginal blocks. These reports predate that gate, so it is
        # computed from their own numbers, as the README's table was.
        mb = marginal_brier(spec.report_md(seed))
        skill = 1.0 - cal["brier"] / mb
        gates = [
            Gate(g.name, g.value, g.limit, g.passed, True, g.derived)
            if g.name in ("accuracy_over_baseline", "worst_question_over_baseline")
            else g
            for g in gates
        ]
        limit = limits.MIN_BRIER_SKILL_OVER_MARGINAL
        gates.insert(
            2, Gate("brier_over_marginal", skill, limit, skill >= limit, False, derived=True)
        )

    return SeedResult(
        seed=seed,
        path=path,
        model=gated.get("model", ""),
        n=cal["n"],
        accuracy=gated["accuracy"],
        baseline=gated["baseline_accuracy"],
        lift=gated["lift_over_baseline"],
        ece=cal["ece"],
        adaptive_ece=cal["adaptive_ece"],
        brier=cal["brier"],
        floor_mean=floor.get("mean"),
        floor_p95=floor.get("p95"),
        distinguishable=cal.get("distinguishable"),
        raw_ece=(raw or {}).get("calibration", {}).get("ece") if raw else None,
        raw_accuracy=raw["accuracy"] if raw else None,
        bins=tuple(cal.get("bins") or ()),
        raw_bins=tuple(((raw or {}).get("calibration") or {}).get("bins") or ()),
        per_question={k: v["lift"] for k, v in (gated.get("per_question") or {}).items()},
        worst_primitive=worst,
        gates=tuple(gates),
        marginal_brier=mb,
    )


def load_run(spec: RunSpec) -> list[SeedResult]:
    return [load_seed(spec, s) for s in spec.seeds]


def spread(values: Iterable[float]) -> tuple[float, float, float]:
    """Median, minimum and maximum: how a configuration is certified."""
    vals = [v for v in values if v is not None]
    return statistics.median(vals), min(vals), max(vals)


# ---------------------------------------------------------------------------
# The landing page's numbers.


@dataclass
class Metric:
    key: str
    value: float
    text: str

    def html(self) -> str:
        return (
            f'<span data-metric="{self.key}" data-value="{self.value!r}">'
            f"{html.escape(self.text)}</span>"
        )


def f4(v: float) -> str:
    return f"{v:.4f}"


def s4(v: float) -> str:
    return f"{v:+.4f}"


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def compute_headlines() -> dict[str, Metric]:
    """Every number the landing page shows, from the committed reports."""
    out: dict[str, Metric] = {}

    def put(key: str, value: float, fmt: Callable[[float], str]) -> None:
        out[key] = Metric(key, value, fmt(value))

    b77 = load_run(RUN_BY_KEY["banking77-qwen"])
    med, lo, hi = spread(r.accuracy for r in b77)
    put("b77_accuracy_median", med, pct)
    put("b77_accuracy_min", lo, pct)
    put("b77_accuracy_max", hi, pct)
    put("b77_baseline_median", statistics.median(r.baseline for r in b77), pct)
    med, lo, hi = spread(r.ece for r in b77)
    put("b77_ece_median", med, f4)
    put("b77_ece_min", lo, f4)
    put("b77_ece_max", hi, f4)
    _, lo, hi = spread(r.floor_p95 for r in b77)
    put("b77_floor_p95_min", lo, f4)
    put("b77_floor_p95_max", hi, f4)
    put("b77_seeds_blocking", sum(r.blocking_passed for r in b77), lambda v: f"{v:.0f}")
    put("b77_seeds", len(b77), lambda v: f"{v:.0f}")

    syn = load_run(RUN_BY_KEY["synthetic-qwen"])
    _, lo, hi = spread(r.per_question["size"] for r in syn)
    put("syn_size_lift_min", lo, s4)
    put("syn_size_lift_max", hi, s4)
    put("syn_accuracy_median", statistics.median(r.accuracy for r in syn), pct)
    worst = [r.worst_primitive[1] for r in syn if r.worst_primitive]
    med, lo, hi = spread(worst)
    put("syn_worst_primitive_ece_median", med, f4)
    put("syn_worst_primitive_ece_min", lo, f4)
    put("syn_worst_primitive_ece_max", hi, f4)
    _, lo, hi = spread(r.floor_p95 for r in syn)
    put("syn_floor_p95_min", lo, f4)
    put("syn_floor_p95_max", hi, f4)
    put("syn_seeds_every_gate", sum(r.every_gate_passed for r in syn), lambda v: f"{v:.0f}")
    put("syn_seeds", len(syn), lambda v: f"{v:.0f}")

    hs2 = load_run(RUN_BY_KEY["helpsteer2-annotators"])
    hard = load_run(RUN_BY_KEY["helpsteer2-annotators-hard"])
    med, lo, hi = spread(r.brier_skill for r in hs2)
    put("hs2_brier_skill_median", med, s4)
    put("hs2_brier_skill_min", lo, s4)
    put("hs2_brier_skill_max", hi, s4)
    put("hs2_brier_skill_limit", limits.MIN_BRIER_SKILL_OVER_MARGINAL, s4)
    _, lo, hi = spread(r.ece for r in hs2)
    put("hs2_ece_min", lo, f4)
    put("hs2_ece_max", hi, f4)
    _, lo, hi = spread(r.floor_p95 for r in hs2)
    put("hs2_floor_p95_min", lo, f4)
    put("hs2_floor_p95_max", hi, f4)
    put(
        "hs2_seeds_within_floor",
        sum(r.distinguishable is False for r in hs2),
        lambda v: f"{v:.0f}",
    )
    put("hs2_raw_ece_soft_median", statistics.median(r.raw_ece for r in hs2), f4)
    put("hs2_raw_ece_hard_median", statistics.median(r.raw_ece for r in hard), f4)
    put("hs2_seeds_blocking", sum(r.blocking_passed for r in hs2), lambda v: f"{v:.0f}")
    put("hs2_seeds", len(hs2), lambda v: f"{v:.0f}")
    return out


# ---------------------------------------------------------------------------
# Syntax colouring, CSS-only. Small on purpose: three languages, no dependency.


def _span(cls: str, text: str) -> str:
    return f'<span class="tk-{cls}">{html.escape(text, quote=False)}</span>'


_JSON_TOKEN = re.compile(
    r'(?P<str>"(?:\\.|[^"\\])*")(?P<colon>\s*:)?'
    r"|(?P<num>-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)"
    r"|(?P<lit>\btrue\b|\bfalse\b|\bnull\b)"
)


def highlight_json(code: str) -> str:
    out, pos = [], 0
    for m in _JSON_TOKEN.finditer(code):
        out.append(html.escape(code[pos : m.start()], quote=False))
        if m.group("str") is not None:
            out.append(_span("key" if m.group("colon") else "str", m.group("str")))
            if m.group("colon"):
                out.append(html.escape(m.group("colon"), quote=False))
        elif m.group("num") is not None:
            out.append(_span("num", m.group("num")))
        else:
            out.append(_span("lit", m.group("lit")))
        pos = m.end()
    out.append(html.escape(code[pos:], quote=False))
    return "".join(out)


def highlight_python(code: str) -> str:
    lines = code.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def at(row: int, col: int) -> int:
        return offsets[row - 1] + col if row - 1 < len(offsets) else len(code)

    out, pos, previous = [], 0, ""
    string_types = {tokenize.STRING}
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        if hasattr(tokenize, name):
            string_types.add(getattr(tokenize, name))
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return html.escape(code, quote=False)
    for tok in tokens:
        start, end = at(*tok.start), at(*tok.end)
        if start < pos:
            continue
        cls = None
        if tok.type == tokenize.COMMENT:
            cls = "com"
        elif tok.type in string_types:
            cls = "str"
        elif tok.type == tokenize.NUMBER:
            cls = "num"
        elif tok.type == tokenize.NAME:
            if tok.string in ("True", "False", "None"):
                cls = "lit"
            elif keyword.iskeyword(tok.string):
                cls = "kw"
            elif previous in ("def", "class"):
                cls = "def"
        if cls:
            out.append(html.escape(code[pos:start], quote=False))
            out.append(_span(cls, code[start:end]))
            pos = end
        if tok.type == tokenize.NAME:
            previous = tok.string
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            previous = ""
    out.append(html.escape(code[pos:], quote=False))
    return "".join(out)


def highlight_shell(code: str) -> str:
    out = []
    for line in code.splitlines(keepends=True):
        m = re.search(r"(^|\s)(#.*)$", line.rstrip("\n"))
        if m:
            cut = m.start(2)
            out.append(html.escape(line[:cut], quote=False))
            out.append(_span("com", line[cut:].rstrip("\n")))
            if line.endswith("\n"):
                out.append("\n")
        else:
            out.append(html.escape(line, quote=False))
    return "".join(out)


HIGHLIGHTERS: dict[str, Callable[[str], str]] = {
    "python": highlight_python,
    "py": highlight_python,
    "json": highlight_json,
    "bash": highlight_shell,
    "sh": highlight_shell,
    "shell": highlight_shell,
}


def code_block(code: str, lang: str = "", cls: str = "") -> str:
    painter = HIGHLIGHTERS.get(lang)
    body = painter(code) if painter else html.escape(code, quote=False)
    klass = f' class="{cls}"' if cls else ""
    lang_attr = f' class="language-{lang}"' if lang else ""
    return f"<pre{klass}><code{lang_attr}>{body}</code></pre>"


# ---------------------------------------------------------------------------
# Markdown, with the repository's links made to work on the site.


def rel(from_page: str, to_path: str) -> str:
    """A relative href from one built page to another built file."""
    base = posixpath.dirname(from_page) or "."
    return posixpath.relpath(to_path, base)


_LIST_START = re.compile(r"^(?:[-*+]|\d+\.)\s")
_HTML_COMMENT_LINE = re.compile(r"^\s*<!--.*-->\s*$")


def preprocess(text: str) -> str:
    """Bridge GitHub's markdown dialect to Python-Markdown's.

    The docs are written for GitHub and are not edited here, so the dialect is
    translated instead:

    - GitHub starts a list or a table on the line after a paragraph;
      Python-Markdown wants a blank line first.
    - GitHub nests a block (a table, a second paragraph) in a list item at two
      spaces of indent; Python-Markdown needs four.
    - ``~~struck~~``, which may span lines, becomes ``<del>``.
    - Standalone HTML comments (the generated-table markers) are dropped, so a
      table that follows one is not swallowed as raw HTML.
    """
    out: list[str] = []
    fenced = False
    in_item = item_had_blank = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        prev = out[-1] if out else ""
        if in_item and line.startswith("  "):
            line = "  " + line
        elif line.strip() and not _LIST_START.match(line):
            in_item = False
        if stripped.startswith("```"):
            if not fenced and prev.strip() and not in_item:
                out.append("")
            fenced = not fenced
            out.append(line)
            continue
        if fenced:
            out.append(line)
            continue
        if _HTML_COMMENT_LINE.match(line):
            out.append("")
            continue
        is_item = _LIST_START.match(line) is not None
        starts_block = is_item or line.startswith("|")
        prev_is_para = (
            prev.strip()
            and not prev.startswith((" ", "\t", "|", "#", ">"))
            and not _LIST_START.match(prev)
        )
        if starts_block and prev_is_para:
            out.append("")
        elif is_item and item_had_blank and prev.strip():
            # An item after one that held a nested block: already a loose
            # list on GitHub, and Python-Markdown needs the blank to see it.
            out.append("")
        if is_item:
            in_item, item_had_blank = True, False
        elif in_item and not line.strip():
            item_had_blank = True
        out.append(line)
    joined = "\n".join(out)
    # Strikethrough outside fenced code, allowed to span lines but not blocks.
    pieces = re.split(r"(^\s*```.*?^\s*```[^\n]*$)", joined, flags=re.DOTALL | re.MULTILINE)
    for i in range(0, len(pieces), 2):
        pieces[i] = re.sub(
            r"~~((?:(?!\n\s*\n).)+?)~~", r"<del>\1</del>", pieces[i], flags=re.DOTALL
        )
    return "".join(pieces)


def _module_path(dotted: str) -> str | None:
    parts = dotted.split(".")
    for cut in (len(parts), len(parts) - 1):
        if cut < 2:
            continue
        base = "src/" + "/".join(parts[:cut])
        for candidate in (base + ".py", base + "/__init__.py"):
            if (ROOT / candidate).is_file():
                return candidate
    return None


def repo_target(text: str) -> str | None:
    """The repository path a code span names, if it names one that exists."""
    text = text.strip()
    if not text or " " in text or len(text) > 200:
        return None
    if re.fullmatch(r"trigon(\.\w+)+", text):
        return _module_path(text)
    candidate = text.split("#")[0].split("::")[0].rstrip(":,")
    if "/" not in candidate and "." not in candidate:
        return None
    if candidate.startswith(("/", "http", "..")) or "*" in candidate:
        return None
    try:
        exists = (ROOT / candidate).exists()
    except OSError:
        return None
    return candidate if exists and candidate not in (".", "./") else None


def href_for_repo_path(path: str, page: str, anchor: str = "") -> str:
    path = posixpath.normpath(path)
    doc = DOC_BY_SOURCE.get(path)
    if doc is not None:
        return rel(page, doc.output) + anchor
    return blob_url(path) + anchor


class LinkReport:
    """Links in the docs that point at nothing in the repository."""

    def __init__(self) -> None:
        self.dangling: list[tuple[str, str]] = []


LINKS = LinkReport()


def rewrite_links(body: str, source: str, page: str) -> str:
    source_dir = posixpath.dirname(source)

    def fix(match: re.Match) -> str:
        attr, target = match.group(1), html.unescape(match.group(2))
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            return match.group(0)
        path, _, frag = target.partition("#")
        anchor = f"#{frag}" if frag else ""
        resolved = posixpath.normpath(posixpath.join(source_dir, path))
        if resolved in DOC_BY_SOURCE or (ROOT / resolved).exists():
            new = href_for_repo_path(resolved, page, anchor)
        else:
            LINKS.dangling.append((source, target))
            new = blob_url(resolved) + anchor
        return f'{attr}="{html.escape(new)}"'

    return re.sub(r'\b(href|src)="([^"]*)"', fix, body)


_PROTECTED = re.compile(r"(<pre\b.*?</pre>|<a\b.*?</a>|<h[1-6]\b.*?</h[1-6]>)", re.DOTALL)


def autolink_code(body: str, page: str) -> str:
    """Make a code span that names a repository file a link to it."""

    def link(match: re.Match) -> str:
        target = repo_target(html.unescape(match.group(1)))
        if target is None:
            return match.group(0)
        return f'<a href="{html.escape(href_for_repo_path(target, page))}">{match.group(0)}</a>'

    parts = _PROTECTED.split(body)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"<code>([^<]+)</code>", link, parts[i])
    return "".join(parts)


def paint_code_blocks(body: str) -> str:
    def paint(match: re.Match) -> str:
        lang, code = match.group(1), html.unescape(match.group(2))
        return code_block(code, lang)

    return re.sub(
        r'<pre><code class="language-([\w+-]+)">(.*?)</code></pre>', paint, body, flags=re.DOTALL
    )


def wrap_tables(body: str) -> str:
    body = re.sub(r"<table>", '<div class="doc-table"><table>', body)
    return body.replace("</table>", "</table></div>")


@dataclass
class Rendered:
    title: str
    body: str
    toc: list[tuple[int, str, str]] = field(default_factory=list)


def render_markdown(text: str, source: str, page: str, *, drop_h1: bool = True) -> Rendered:
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc"],
        extension_configs={"toc": {"toc_depth": "1-3"}},
    )
    body = md.convert(preprocess(text))
    title = ""
    first = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.DOTALL)
    if first:
        title = re.sub(r"<[^>]+>", "", first.group(1)).strip()
        if drop_h1:
            body = body[: first.start()] + body[first.end() :]
    toc: list[tuple[int, str, str]] = []

    def walk(tokens: list[dict]) -> None:
        for t in tokens:
            if t["level"] in (2, 3):
                toc.append((t["level"], t["id"], html.unescape(re.sub(r"<[^>]+>", "", t["name"]))))
            walk(t.get("children", []))

    walk(md.toc_tokens)
    body = paint_code_blocks(body)
    body = rewrite_links(body, source, page)
    body = autolink_code(body, page)
    body = wrap_tables(body)
    return Rendered(title=html.unescape(title), body=body, toc=toc)


def inline_markdown(text: str, source: str, page: str) -> str:
    """One paragraph of markdown, without the <p>."""
    body = render_markdown(text, source, page, drop_h1=False).body.strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", body, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# The ledger, read by section.


def ledger_sections() -> dict[str, str]:
    text = (ROOT / "docs/ledger.md").read_text()
    sections: dict[str, str] = {}
    current, buf = "", []
    for line in text.split("\n"):
        if line.startswith("## "):
            sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return {k: re.sub(r"\n-{3,}\s*$", "", v).strip() for k, v in sections.items()}


def _cells(row: str) -> list[str]:
    row = row.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", row)]


def markdown_table(section: str) -> tuple[list[str], list[list[str]]]:
    rows = [ln for ln in section.split("\n") if ln.strip().startswith("|")]
    if len(rows) < 2:
        return [], []
    return _cells(rows[0]), [_cells(r) for r in rows[2:]]


def ledger_prose(section: str) -> str:
    """A section's text up to its first table or list."""
    lines = []
    for line in section.split("\n"):
        if line.strip().startswith(("|", "- ")):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def ledger_bullets(section: str) -> list[str]:
    items: list[str] = []
    for line in section.split("\n"):
        if line.startswith("- "):
            items.append(line[2:])
        elif items and line.startswith("  "):
            items[-1] += " " + line.strip()
    return items


def open_items() -> list[tuple[str, str]]:
    """The ledger's open items that are still open: (title, first sentence)."""
    out = []
    for item in ledger_bullets(ledger_sections().get("Open", "")):
        if item.startswith("~~"):
            continue
        m = re.match(r"\*\*(.+?)\*\*\s*(.*)", item)
        if not m:
            continue
        title, rest = m.group(1).strip(), m.group(2).strip()
        sentence = re.match(r"(.+?[.!?])(\s|$)", rest)
        out.append((title, sentence.group(1) if sentence else rest))
    return out


# ---------------------------------------------------------------------------
# Page chrome.

LAYOUT = string.Template((SITE_SRC / "layout.html").read_text())

TOP_NAV = (
    ("docs/index.html", "Docs"),
    ("examples/index.html", "Examples"),
    ("examples/api.html", "API"),
    ("proofs/index.html", "Proofs"),
)


def page(
    *,
    path: str,
    title: str,
    description: str,
    body: str,
    section: str,
    current: str,
    stylesheets: tuple[str, ...],
) -> str:
    root = rel(path, "index.html")[: -len("index.html")]
    nav = []
    for target, label in TOP_NAV:
        mark = ' aria-current="page"' if target == current else ""
        nav.append(f'    <a href="{rel(path, target)}"{mark}>{label}</a>')
    nav.append(f'    <a href="{GITHUB}">GitHub</a>')
    footer = [f'      <a href="{rel(path, "index.html")}">Home</a>'] + [
        f'      <a href="{rel(path, t)}">{label}</a>' for t, label in TOP_NAV
    ]
    footer.append(f'      <a href="{GITHUB}">GitHub</a>')
    sheets = "\n".join(
        f'<link rel="stylesheet" href="{rel(path, s)}" />' for s in ("theme.css", *stylesheets)
    )
    return LAYOUT.substitute(
        title=html.escape(title),
        description=html.escape(description),
        stylesheets=sheets,
        root=root,
        section=html.escape(section),
        nav="\n".join(nav),
        footer_nav="\n".join(footer),
        body=body,
        github=GITHUB,
        branch=html.escape(BRANCH),
    )


def sidebar(path: str, subs: list[tuple[str, str]] | None = None) -> str:
    """The static sidebar: every doc, then the evidence pages."""
    out = ['<nav class="doc-side" aria-label="Documentation">']

    def item(target: str, label: str) -> None:
        mark = ' aria-current="page"' if target == path else ""
        out.append(f'<li><a href="{rel(path, target)}"{mark}>{html.escape(label)}</a></li>')
        if target == path and subs:
            for anchor, text in subs:
                out.append(f'<li class="sub"><a href="#{anchor}">{html.escape(text)}</a></li>')

    for group, docs in DOC_GROUPS:
        out.append(f'<h2 class="nav-section">{html.escape(group)}</h2>')
        out.append('<ul class="nav-list">')
        for doc in docs:
            item(doc.output, doc.label)
        out.append("</ul>")
    out.append('<h2 class="nav-section">Evidence and examples</h2>')
    out.append('<ul class="nav-list">')
    for target, label in EVIDENCE_PAGES:
        item(target, label)
    out.append("</ul></nav>")
    return "\n".join(out)


def shell(
    path: str,
    article: str,
    *,
    toc: list[tuple[int, str, str]] | None = None,
    subs: list[tuple[str, str]] | None = None,
    wide: bool = False,
) -> str:
    right = ""
    if toc and not wide:
        links = "".join(
            f'<li><a href="#{anchor}">{html.escape(text)}</a></li>'
            for level, anchor, text in toc
            if level == 2
        )
        right = (
            '<aside class="doc-toc" aria-label="On this page">'
            f'<h2 class="nav-section">On this page</h2><ul class="nav-list">{links}</ul></aside>'
        )
    klass = "doc-shell wide" if wide or not right else "doc-shell"
    return (
        f'<div class="{klass}">\n{sidebar(path, subs)}\n'
        f'<main id="main" class="doc-article">\n{article}\n</main>\n{right}\n</div>'
    )


# ---------------------------------------------------------------------------
# Docs.


def build_docs(out: Path) -> list[str]:
    written = []
    for i, doc in enumerate(DOCS):
        text = (ROOT / doc.source).read_text()
        rendered = render_markdown(text, doc.source, doc.output)
        title = doc.title or rendered.title or doc.label
        group = next(g for g, docs in DOC_GROUPS if doc in docs)
        prev_doc = DOCS[i - 1] if i > 0 else None
        next_doc = DOCS[i + 1] if i + 1 < len(DOCS) else None
        pager = ['<nav class="doc-pager" aria-label="Previous and next">']
        if prev_doc:
            pager.append(
                f'<a class="pager" href="{rel(doc.output, prev_doc.output)}">'
                f'<span class="pager-kind">Previous</span>'
                f'<span class="pager-label">{html.escape(prev_doc.label)}</span></a>'
            )
        if next_doc:
            pager.append(
                f'<a class="pager pager-next" href="{rel(doc.output, next_doc.output)}">'
                f'<span class="pager-kind">Next</span>'
                f'<span class="pager-label">{html.escape(next_doc.label)}</span></a>'
            )
        pager.append("</nav>")
        article = (
            f'<p class="doc-kicker">{html.escape(group)}</p>\n'
            f"<h1>{html.escape(title)}</h1>\n"
            f'<p class="source-line">Rendered from <a href="{blob_url(doc.source)}">'
            f"<code>{doc.source}</code></a>.</p>\n"
            f"{rendered.body}\n{''.join(pager)}"
        )
        description = _first_sentence(text)
        (out / doc.output).parent.mkdir(parents=True, exist_ok=True)
        (out / doc.output).write_text(
            page(
                path=doc.output,
                title=f"{title} — trigon docs",
                description=description,
                body=shell(doc.output, article, toc=rendered.toc),
                section="docs",
                current="docs/index.html",
                stylesheets=("docs.css",),
            )
        )
        written.append(doc.output)
    return written


def _first_sentence(md_text: str) -> str:
    for block in re.split(r"\n\s*\n", md_text):
        block = block.strip()
        if not block or block.startswith(("#", "|", "```", "-", ">", "<")):
            continue
        plain = re.sub(r"[`*_]|\[([^\]]*)\]\([^)]*\)", r"\1", " ".join(block.split()))
        m = re.match(r"(.+?[.!?])(\s|$)", plain)
        return (m.group(1) if m else plain)[:300]
    return "trigon documentation"


# ---------------------------------------------------------------------------
# Examples.


def lexical_response(request: dict) -> dict:
    """What the engine returns for a request, on the lexical floor.

    The id is random per request and the timings are this machine's, so both
    are pinned: the page shows the response's shape, not a measurement.
    """
    from trigon.backends.lexical import LexicalBackend
    from trigon.engine import Engine
    from trigon.types import DecisionRequest

    response = Engine(LexicalBackend()).answer(DecisionRequest.model_validate(request))
    data = json.loads(response.model_dump_json())
    data["id"] = "so_…"
    if "timing" in data:
        data["timing"] = {k: "…" for k in data["timing"]}
    return data


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def request_response(request: dict, *, name: str) -> str:
    response = _round(lexical_response(request))
    req = json.dumps(request, indent=2, ensure_ascii=False)
    res = json.dumps(response, indent=2, ensure_ascii=False)
    return (
        '<div class="pair">'
        f"<figure><figcaption>Request · {html.escape(name)}</figcaption>"
        f"{code_block(req, 'json')}</figure>"
        "<figure><figcaption>Response · lexical floor</figcaption>"
        f"{code_block(res, 'json')}</figure>"
        "</div>"
    )


def build_examples(out: Path) -> str:
    from trigon.usecases import all_use_cases

    path = "examples/index.html"
    parts = [
        '<p class="doc-kicker">Evidence and examples</p>',
        "<h1>Examples</h1>",
        "<p>Every request below is a file in the repository or a use case in "
        f'<a href="{blob_url("src/trigon/usecases.py")}"><code>trigon.usecases</code></a>, '
        "and every response was produced by the engine when this page was built.</p>",
        '<div class="aside"><p><strong>The responses come from the lexical floor.</strong> '
        "It needs no weights and no GPU, so it runs anywhere, and its numbers are not a "
        "model's: a floor exercises the whole path (compile, answer, calibrate, "
        "serialise) and says nothing about quality. The shapes are the contract's; for "
        "what a trained model answers, see "
        f'<a href="{rel(path, "proofs/index.html")}">Proofs</a>. The id and the timings are '
        "elided because they differ on every call.</p></div>",
    ]
    subs = [("request-file", "The request file")]

    ticket = json.loads((ROOT / "examples/support-ticket.json").read_text())
    parts += [
        '<h2 id="request-file">The request file</h2>',
        f'<p><a href="{blob_url("examples/support-ticket.json")}">'
        "<code>examples/support-ticket.json</code></a> asks four questions of one ticket, "
        "one of each primitive and a second Noul. A Noul answer has no confidence field by "
        "design: for a yes/no question the probability already is the answer.</p>",
        code_block("trigon ask examples/support-ticket.json", "bash"),
        request_response(ticket, name="support-ticket.json"),
    ]

    parts.append('<h2 id="use-cases">Use cases</h2>')
    subs.append(("use-cases", "Use cases"))
    parts.append(
        "<p>The committed use cases, each a schema plus a representative state. Their "
        "states are illustrative and hand-written to the right shape and length; they "
        f'are what <a href="{blob_url("scripts/price.py")}"><code>scripts/price.py</code></a> '
        f'prices in <a href="{rel(path, "docs/pricing.html")}">Pricing a use case</a>.</p>'
    )
    for uc in all_use_cases():
        request = {
            "state": uc.state,
            "questions": {
                qid: {"type": q.type, **q.model_dump(mode="json", exclude_defaults=True)}
                for qid, q in uc.questions.items()
            },
        }
        anchor = f"use-case-{uc.name.replace('_', '-')}"
        parts += [
            f'<h3 id="{anchor}"><code>{html.escape(uc.name)}</code></h3>',
            f"<p>{html.escape(uc.purpose)}</p>",
            '<dl class="facts">',
            f"<dt>Would train on</dt><dd>{html.escape(uc.corpus)}</dd>",
            f"<dt>Priced at</dt><dd>{uc.monthly_volume:,} requests a month</dd>",
            f"<dt>Why this shape</dt><dd>{html.escape(uc.notes)}</dd>" if uc.notes else "",
            "</dl>",
            code_block(f"python examples/usecase_cookbook.py {uc.name}", "bash"),
            request_response(request, name=uc.name),
        ]

    parts.append('<h2 id="cookbooks">Cookbooks</h2>')
    subs.append(("cookbooks", "Cookbooks"))
    for py in sorted((ROOT / "examples").glob("*.py")):
        source = py.read_text()
        rel_path = f"examples/{py.name}"
        doc = ast.get_docstring(ast.parse(source)) or ""
        intro = next((b for b in doc.split("\n\n")[1:] if b.strip()), doc.split("\n\n")[0])
        anchor = py.stem.replace("_", "-")
        parts += [
            f'<h3 id="{anchor}"><code>{rel_path}</code></h3>',
            f"<p>{inline_markdown(intro, rel_path, path)}</p>",
            f'<p class="code-caption"><a href="{blob_url(rel_path)}">{rel_path}</a>, '
            f"{source.count(chr(10))} lines</p>",
            code_block(source, "python"),
        ]
    body = shell(path, "\n".join(parts), subs=subs, wide=True)
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    (out / path).write_text(
        page(
            path=path,
            title="Examples — trigon",
            description="Requests and responses for every committed use case and example.",
            body=body,
            section="examples",
            current=path,
            stylesheets=("docs.css",),
        )
    )
    return path


# ---------------------------------------------------------------------------
# The API reference.


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def schema_type(s: dict) -> str:
    if not s:
        return "any"
    if "$ref" in s:
        name = _ref_name(s["$ref"])
        return f'<a href="#schema-{name}">{name}</a>'
    for key in ("anyOf", "oneOf"):
        if key in s:
            return " | ".join(schema_type(x) for x in s[key])
    if "const" in s:
        return html.escape(json.dumps(s["const"]))
    if "enum" in s:
        return " | ".join(html.escape(json.dumps(v)) for v in s["enum"])
    t = s.get("type")
    if t == "array":
        return f"array&lt;{schema_type(s.get('items') or {})}&gt;"
    if t == "object":
        extra = s.get("additionalProperties")
        if isinstance(extra, dict):
            return f"map&lt;string, {schema_type(extra)}&gt;"
        return "object"
    return html.escape(str(t or "any"))


_CONSTRAINTS = (
    ("default", "default"),
    ("minimum", "≥"),
    ("maximum", "≤"),
    ("exclusiveMinimum", ">"),
    ("minLength", "min length"),
    ("maxLength", "max length"),
    ("minItems", "min items"),
    ("maxItems", "max items"),
    ("minProperties", "min entries"),
    ("maxProperties", "max entries"),
)


def _constraints(s: dict) -> str:
    bits = []
    for key, label in _CONSTRAINTS:
        if key in s:
            bits.append(f"{label} <code>{html.escape(json.dumps(s[key]))}</code>")
    return " · ".join(bits)


def build_api(out: Path) -> str:
    path = "examples/api.html"
    spec = json.loads((ROOT / "spec/openapi.json").read_text())
    schemas = spec["components"]["schemas"]
    parts = [
        '<p class="doc-kicker">Evidence and examples</p>',
        f"<h1>{html.escape(spec['info']['title'])}</h1>",
        f'<p class="source-line">Generated from <a href="{blob_url("spec/openapi.json")}">'
        f"<code>spec/openapi.json</code></a>, OpenAPI {html.escape(spec['openapi'])}, "
        f"contract version {html.escape(spec['info']['version'])}. That file is itself "
        "generated from the gateway and drift-tested against it.</p>",
    ]
    parts += [
        f"<p>{html.escape(p)}</p>"
        for p in re.split(r"\n\s*\n", spec["info"].get("description", ""))
        if p.strip()
    ]
    parts.append('<h2 id="endpoints">Endpoints</h2>')
    subs = [("endpoints", "Endpoints")]
    from trigon.server.compat import COMPAT_PATH

    for route, methods in spec["paths"].items():
        # The compatibility route's path is the incumbent's, and this project
        # does not print their name (docs/decisions.md, Q23): the site shows
        # what the path is, not what it says.
        shown = route.replace(COMPAT_PATH, "/<the incumbent's path>")
        for method, op in methods.items():
            anchor = "op-" + re.sub(r"[^a-z0-9]+", "-", f"{method}{shown}".lower()).strip("-")
            parts.append(
                f'<h3 id="{anchor}" class="endpoint"><span class="method">{method.upper()}</span>'
                f"<span>{html.escape(shown)}</span></h3>"
            )
            if op.get("summary"):
                parts.append(f"<p><strong>{html.escape(op['summary'])}</strong></p>")
            if op.get("description"):
                for para in re.split(r"\n\s*\n", op["description"]):
                    parts.append(f"<p>{html.escape(' '.join(para.split()))}</p>")
            rows = []
            body = op.get("requestBody", {}).get("content", {}).get("application/json")
            if body:
                rows.append(("Request body", schema_type(body.get("schema", {}))))
            for code, resp in op.get("responses", {}).items():
                content = resp.get("content", {}).get("application/json")
                kind = schema_type(content.get("schema", {})) if content else "—"
                rows.append((f"{code} {html.escape(resp.get('description', ''))}", kind))
            parts.append('<div class="doc-table"><table><tbody>')
            for label, kind in rows:
                parts.append(
                    f'<tr><th scope="row">{label}</th><td class="schema-type">{kind}</td></tr>'
                )
            parts.append("</tbody></table></div>")

    parts.append('<h2 id="schemas">Schemas</h2>')
    subs.append(("schemas", "Schemas"))
    for name, s in schemas.items():
        parts.append(f'<h3 id="schema-{name}"><code>{html.escape(name)}</code></h3>')
        if s.get("description"):
            text = html.escape(" ".join(s["description"].split()))
            text = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", text)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            parts.append(f"<p>{text}</p>")
        props = s.get("properties") or {}
        if not props:
            parts.append(f'<p class="schema-type">{schema_type(s)}</p>')
            continue
        required = set(s.get("required", []))
        parts.append(
            '<div class="doc-table"><table><thead><tr><th scope="col">Field</th>'
            '<th scope="col">Type</th><th scope="col">Description</th></tr></thead><tbody>'
        )
        for prop, ps in props.items():
            req = ' <span class="req">required</span>' if prop in required else ""
            desc = html.escape(" ".join((ps.get("description") or "").split()))
            desc = re.sub(r"`([^`]+)`", r"<code>\1</code>", desc)
            desc = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", desc)
            extra = _constraints(ps)
            if extra:
                desc = f"{desc}<br /><small>{extra}</small>" if desc else f"<small>{extra}</small>"
            parts.append(
                f'<tr><td class="field"><code>{html.escape(prop)}</code>{req}</td>'
                f'<td class="schema-type">{schema_type(ps)}</td><td>{desc}</td></tr>'
            )
        parts.append("</tbody></table></div>")
    body = shell(path, "\n".join(parts), subs=subs, wide=True)
    (out / path).write_text(
        page(
            path=path,
            title="API reference — trigon",
            description=spec["info"]["title"] + ": endpoints and schemas.",
            body=body,
            section="API",
            current=path,
            stylesheets=("docs.css",),
        )
    )
    return path


# ---------------------------------------------------------------------------
# Proofs.


@dataclass(frozen=True)
class Claim:
    title: str
    statement: str
    #: Report files or directories that show it, if it is a measurement.
    evidence: tuple[str, ...]
    #: (test file, test function names). Every name is checked to exist.
    tests: tuple[tuple[str, tuple[str, ...]], ...]


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "Adding, removing or reordering a question moves no other answer",
        "The block mask gives no path from one question's schema or readout to "
        "another's. Exact to floating-point equality where the shapes match; across "
        "sequence lengths a different GEMM kernel may round differently, so there the "
        "tests hold to a float32 bound.",
        (),
        (
            (
                "tests/test_independence.py",
                (
                    "test_adding_a_question_moves_nothing_else",
                    "test_removing_every_other_question_moves_nothing",
                    "test_question_map_order_does_not_matter",
                    "test_twenty_extra_questions_still_move_nothing",
                ),
            ),
            (
                "tests/test_end_to_end.py",
                ("test_added_questions_do_not_move_the_others_on_the_served_path",),
            ),
        ),
    ),
    Claim(
        "The schema half of a request is a cacheable prefix",
        "The schema encodes identically whatever the state, so its KV prefix is keyed "
        "on the schema alone and reused across requests. The cache is never used in "
        "training, where the weights that produced a prefix move every step.",
        ("reports/cache/README.md",),
        (
            ("tests/test_independence.py", ("test_schema_states_do_not_depend_on_state",)),
            (
                "tests/test_prefix_cache.py",
                (
                    "test_the_cached_path_agrees_with_the_uncached_one",
                    "test_a_different_state_reuses_the_prefix_and_still_answers_differently",
                    "test_training_never_uses_a_cached_prefix",
                    "test_neither_path_lets_a_question_reach_another_and_neither_is_exact",
                ),
            ),
        ),
    ),
    Claim(
        "Schema violations are unrepresentable, not validated",
        "Each head is a softmax over exactly the labels the request declared; a Noul "
        "has no confidence field in the published contract, and the served path "
        "refuses what the contract forbids.",
        ("spec/openapi.json",),
        (
            (
                "tests/test_independence.py",
                ("test_output_shapes_always_match_the_declared_label_sets",),
            ),
            (
                "tests/test_openapi_drift.py",
                ("test_noul_answer_has_no_confidence_field_in_the_published_contract",),
            ),
            (
                "tests/test_end_to_end.py",
                ("test_the_deployment_refuses_what_the_contract_forbids",),
            ),
        ),
    ),
    Claim(
        "Calibration never certifies alone",
        "A model that reports each question's marginal is calibrated by construction "
        "and useless. The gates keep a term that fails it: accuracy over the marginal "
        "predictor, or on a drawn-annotator corpus the Brier skill over the marginal.",
        ("reports/README.md",),
        (
            (
                "tests/test_evals.py",
                (
                    "test_a_marginal_predictor_passes_every_calibration_gate",
                    "test_on_a_drawn_annotator_corpus_the_proper_score_rejects_the_marginal",
                ),
            ),
        ),
    ),
    Claim(
        "A number is not evidence until the floor is under it",
        "Every ECE is published beside the simulated ECE of a perfectly calibrated "
        "model at the same sample size. A run too small to separate the two cannot "
        "certify anything.",
        (),
        (
            (
                "tests/test_evals.py",
                (
                    "test_a_run_too_small_to_test_cannot_certify_anything",
                    "test_the_floor_is_reported_alongside_the_number",
                ),
            ),
            (
                "tests/test_noise_floor.py",
                (
                    "test_a_calibrated_model_is_reported_as_indistinguishable",
                    "test_real_miscalibration_is_reported_as_distinguishable",
                ),
            ),
        ),
    ),
    Claim(
        "Pooled ECE is not a model's calibration",
        "Pooling cancels heads that err in opposite directions, so the gates also "
        "read the worst primitive.",
        ("reports/synthetic/README.md",),
        (
            (
                "tests/test_evals.py",
                (
                    "test_pooled_ece_cancels_heads_that_err_in_opposite_directions",
                    "test_the_worst_primitive_gate_reads_the_worst_primitive",
                ),
            ),
        ),
    ),
    Claim(
        "A calibrator is a proposal, not a result",
        "A temperature and an isotonic map are both fitted, scored on a slice neither "
        "saw, and applied only if one demonstrably helps. A tilted head has no correct "
        "temperature; isotonic fixes it.",
        ("reports/calibration/decline-power.md",),
        (
            (
                "tests/test_isotonic.py",
                (
                    "test_it_corrects_a_tilt_that_no_temperature_can",
                    "test_a_noul_map_is_fitted_on_the_quantity_it_is_applied_to",
                ),
            ),
        ),
    ),
    Claim(
        "Every front door answers the same",
        "The CLI and the gateway share one engine and return the same probabilities; "
        "the incumbent-compatible path and the native one agree per primitive.",
        ("docs/compat.md",),
        (
            ("tests/test_server.py", ("test_gateway_and_cli_answer_identically",)),
            (
                "tests/test_compat.py",
                ("test_the_compat_path_and_the_native_path_answer_identically",),
            ),
        ),
    ),
    Claim(
        "The numbers in the docs are the numbers the code enforces",
        "Budgets, gate limits, the cost tables, the contract and the ledger's counts "
        "are each pinned against their source, so relaxing one in code and not in "
        "prose fails the build.",
        (),
        (
            (
                "tests/test_docs_drift.py",
                (
                    "test_the_budget_table_matches_the_budgets",
                    "test_the_published_gate_limits_are_the_enforced_ones",
                    "test_the_testability_rule_is_stated_as_it_is_enforced",
                ),
            ),
            (
                "tests/test_attention_table.py",
                ("test_checked_in_tables_match_what_the_code_produces",),
            ),
            ("tests/test_openapi_drift.py", ("test_checked_in_spec_matches_the_app",)),
            ("tests/test_ledger_drift.py", ("test_the_gate_count_matches_the_gates_that_exist",)),
        ),
    ),
    Claim(
        "Licence tiers are enforced in code",
        "An amber corpus evaluates and never trains; a red one loads for nothing; the "
        "committed tiers match the licence audit.",
        ("docs/data.md",),
        (
            (
                "tests/test_corpora.py",
                (
                    "test_an_amber_corpus_evals_but_never_trains",
                    "test_a_red_corpus_may_not_be_loaded_for_anything",
                    "test_the_committed_tiers_match_the_licence_audit",
                ),
            ),
        ),
    ),
    Claim(
        "The conformal wrapper checks its own promise",
        "Coverage is measured on a third split and the fit fails when it falls more "
        "than three sigma of sampling noise below target.",
        ("reports/conformal",),
        (("tests/test_conformal_gate.py", ("test_a_predictor_that_under_covers_is_caught",)),),
    ),
    Claim(
        "This site's headline numbers are the reports'",
        "The landing page's figures are recomputed from the committed JSON on every test run.",
        (),
        (("tests/test_site.py", ("test_the_landing_page_numbers_are_the_reports",)),),
    ),
)


def test_line(test_file: str, name: str) -> int:
    """Where a named test is defined; raises if it is not, so a claim cannot
    keep citing a test that was renamed or deleted."""
    for i, line in enumerate((ROOT / test_file).read_text().splitlines(), 1):
        if re.match(rf"\s*(async\s+)?def {re.escape(name)}\(", line):
            return i
    raise LookupError(f"{test_file} defines no {name}: update CLAIMS in scripts/build_site.py")


def render_claims(path: str) -> str:
    out = []
    for claim in CLAIMS:
        items = []
        for test_file, names in claim.tests:
            for name in names:
                line = test_line(test_file, name)
                items.append(
                    f'<li><a href="{blob_url(test_file)}#L{line}"><code>{test_file}</code></a> '
                    f"<code>{html.escape(name)}</code></li>"
                )
        evidence = ""
        if claim.evidence:
            links = ", ".join(
                f'<a href="{html.escape(href_for_repo_path(e, path))}"><code>{e}</code></a>'
                for e in claim.evidence
            )
            evidence = f"<p>Measured in {links}.</p>"
        out.append(
            f'<section class="claim"><h3>{html.escape(claim.title)}</h3>'
            f"<p>{html.escape(claim.statement)}</p>{evidence}"
            f'<p class="code-caption">Pinned by</p><ul>{"".join(items)}</ul></section>'
        )
    return "\n".join(out)


def reliability_svg(bins: tuple[dict, ...], raw_bins: tuple[dict, ...], label: str) -> str:
    """A reliability diagram from a report's own bins: no smoothing, no invention.

    Filled gold is the gated (calibrated) run, hollow is the raw one; a point's
    area follows its bin's count. Empty bins are not drawn.
    """
    size, pad = 200, 26
    span = size - pad - 8

    def xy(conf: float, acc: float) -> tuple[float, float]:
        return pad + conf * span, 8 + (1 - acc) * span

    parts = [
        f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="{html.escape(label)}">',
        f"<title>{html.escape(label)}</title>",
    ]
    for t in (0.25, 0.5, 0.75):
        x, _ = xy(t, 0)
        _, y = xy(0, t)
        parts.append(f'<line class="rd-grid" x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{8 + span}"/>')
        parts.append(
            f'<line class="rd-grid" x1="{pad}" y1="{y:.1f}" x2="{pad + span}" y2="{y:.1f}"/>'
        )
    parts.append(
        f'<line class="rd-axis" x1="{pad}" y1="{8 + span}" x2="{pad + span}" y2="{8 + span}"/>'
        f'<line class="rd-axis" x1="{pad}" y1="8" x2="{pad}" y2="{8 + span}"/>'
        f'<line class="rd-diag" x1="{pad}" y1="{8 + span}" x2="{pad + span}" y2="8"/>'
    )
    parts.append(
        f'<text class="rd-label" x="{pad + span / 2}" y="{size - 4}" text-anchor="middle">'
        "confidence</text>"
        f'<text class="rd-label" x="10" y="{8 + span / 2}" text-anchor="middle" '
        f'transform="rotate(-90 10 {8 + span / 2})">accuracy</text>'
        f'<text class="rd-label" x="{pad}" y="{8 + span + 12}" text-anchor="middle">0</text>'
        f'<text class="rd-label" x="{pad + span}" y="{8 + span + 12}" text-anchor="middle">1</text>'
    )

    def points(rows: tuple[dict, ...], cls: str, what: str) -> None:
        total = sum(b["count"] for b in rows) or 1
        for b in rows:
            if not b["count"]:
                continue
            x, y = xy(b["mean_confidence"], b["accuracy"])
            r = max(2.2, min(8.0, 2.2 + 14 * (b["count"] / total) ** 0.5))
            tip = (
                f"{what}: confidence {b['lower']:.2f}–{b['upper']:.2f}, n={b['count']:,}, "
                f"mean confidence {b['mean_confidence']:.3f}, accuracy {b['accuracy']:.3f}"
            )
            parts.append(
                f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}">'
                f"<title>{html.escape(tip)}</title></circle>"
            )

    points(raw_bins, "rd-raw", "raw")
    points(bins, "rd-cal", "gated")
    parts.append("</svg>")
    return "".join(parts)


def _verdict(ok: bool, yes: str = "pass", no: str = "fail") -> str:
    return f'<span class="verdict {"pass" if ok else "fail"}">{yes if ok else no}</span>'


def _separable(r: SeedResult) -> str:
    if r.distinguishable is None:
        return "—"
    return "above" if r.distinguishable else "within"


def render_run(spec: RunSpec, path: str) -> str:
    seeds = load_run(spec)
    head = [
        ("Seed", ""),
        ("Accuracy", "num"),
        ("Marginal", "num"),
        ("Lift", "num"),
        ("Raw ECE", "num"),
        ("ECE", "num"),
        ("Floor p95", "num"),
        ("vs floor", ""),
        ("Adaptive ECE", "num"),
        ("Worst head ECE", "num"),
    ]
    if spec.drawn_annotator:
        head.append(("Brier skill", "num"))
    head += [("Blocking gates", ""), ("Every gate", "")]

    def row(r: SeedResult) -> str:
        cells = [
            str(r.seed),
            f4(r.accuracy),
            f4(r.baseline),
            s4(r.lift),
            f4(r.raw_ece) if r.raw_ece is not None else "—",
            f4(r.ece),
            f4(r.floor_p95) if r.floor_p95 is not None else "—",
            _separable(r),
            f4(r.adaptive_ece),
            f"{f4(r.worst_primitive[1])}<br /><small>{r.worst_primitive[0]}</small>"
            if r.worst_primitive
            else "—",
        ]
        if spec.drawn_annotator:
            cells.append(s4(r.brier_skill))
        cells.append(_verdict(r.blocking_passed))
        failed = [g.name for g in r.failed]
        cells.append(
            _verdict(r.every_gate_passed)
            + (
                f' <small title="{html.escape(", ".join(failed))}">{len(failed)} failed</small>'
                if failed
                else ""
            )
        )
        return (
            "<tr>"
            + "".join(
                f'<td class="{cls}">{c}</td>' for c, (_, cls) in zip(cells, head, strict=True)
            )
            + "</tr>"
        )

    def summary(label: str, pick: Callable[[tuple[float, float, float]], str]) -> str:
        cols: list[str] = [label]
        getters: list[Callable[[SeedResult], float | None]] = [
            lambda r: r.accuracy,
            lambda r: r.baseline,
            lambda r: r.lift,
            lambda r: r.raw_ece,
            lambda r: r.ece,
            lambda r: r.floor_p95,
            None,
            lambda r: r.adaptive_ece,
            lambda r: r.worst_primitive[1] if r.worst_primitive else None,
        ]
        if spec.drawn_annotator:
            getters.append(lambda r: r.brier_skill)
        for i, get in enumerate(getters):
            if get is None:
                cols.append("")
                continue
            vals = [get(r) for r in seeds]
            vals = [v for v in vals if v is not None]
            if not vals:
                cols.append("—")
                continue
            signed = head[i + 1][0] in ("Lift", "Brier skill")
            cols.append(pick(spread(vals), signed))
        if label == "Median":
            cols.append(f"{sum(r.blocking_passed for r in seeds)} of {len(seeds)}")
            cols.append(f"{sum(r.every_gate_passed for r in seeds)} of {len(seeds)}")
        else:
            cols += ["", ""]
        return (
            '<tr class="summary">'
            + "".join(f'<td class="{cls}">{c}</td>' for c, (_, cls) in zip(cols, head, strict=True))
            + "</tr>"
        )

    def med(s: tuple[float, float, float], signed: bool) -> str:
        return (s4 if signed else f4)(s[0])

    def rng(s: tuple[float, float, float], signed: bool) -> str:
        fmt = s4 if signed else f4
        return f"{fmt(s[1])}–<wbr />{fmt(s[2])}"

    table = (
        '<div class="doc-table"><table class="seeds"><thead><tr>'
        + "".join(f'<th scope="col" class="{cls}">{h}</th>' for h, cls in head)
        + "</tr></thead><tbody>"
        + "".join(row(r) for r in seeds)
        + summary("Median", med)
        + summary("Range", rng).replace('<tr class="summary">', '<tr class="summary range">')
        + "</tbody></table></div>"
    )

    certified_word = spec.role
    ok = spec.role.startswith("certified")
    pill = f'<span class="verdict {"pass" if ok else "info"}">{html.escape(certified_word)}</span>'
    meta = [f"{html.escape(spec.backbone)}", f"n = {seeds[0].n:,} scored per seed"]
    modal = ROOT / spec.modal_run
    if modal.exists():
        runs = json.loads(modal.read_text())
        commits = sorted({r["commit"][:7] for r in runs})
        gpus = sorted({r.get("gpu_actual", "") for r in runs} - {""})
        meta.append(f"commit {', '.join(commits)}")
        if gpus:
            meta.append(", ".join(gpus))
        meta.append(f'<a href="{blob_url(spec.modal_run)}">launch record</a>')
    meta.append(f'<a href="{blob_url(spec.readme)}">README</a>')
    meta.append(f'<a href="{blob_url(spec.directory)}">reports</a>')

    diagrams = "".join(
        "<figure>"
        + reliability_svg(
            r.bins,
            r.raw_bins,
            f"{spec.title}, seed {r.seed}: reliability diagram, ECE {f4(r.ece)}",
        )
        + f"<figcaption>seed {r.seed} · ECE {f4(r.ece)}"
        + (f" · floor {f4(r.floor_p95)}" if r.floor_p95 is not None else "")
        + "</figcaption></figure>"
        for r in seeds
    )

    gate_rows = []
    for r in seeds:
        for g in r.gates:
            note = " (computed here)" if g.derived else ""
            gate_rows.append(
                f"<tr><td>{r.seed}</td><td><code>{g.name}</code>{note}</td>"
                f'<td class="num">{g.value:.4f}</td><td class="num">{g.limit:.4f}</td>'
                f"<td>{'advisory' if g.advisory else 'blocking'}</td>"
                f"<td>{_verdict(g.passed)}</td></tr>"
            )
    gates = (
        "<details><summary>Every gate, every seed</summary>"
        '<div class="doc-table"><table><thead><tr>'
        '<th scope="col">Seed</th><th scope="col">Gate</th>'
        '<th scope="col" class="num">Value</th><th scope="col" class="num">Limit</th>'
        '<th scope="col">Kind</th><th scope="col">Verdict</th></tr></thead><tbody>'
        + "".join(gate_rows)
        + "</tbody></table></div>"
        + "<p>Files: "
        + ", ".join(f'<a href="{blob_url(r.path)}"><code>{r.path}</code></a>' for r in seeds)
        + "</p></details>"
    )
    extra = ""
    if spec.drawn_annotator:
        extra = (
            "<p><small>On this corpus the accuracy gates are advisory and "
            "<code>brier_over_marginal</code> blocks (decision Q20). These reports predate "
            "that gate, so it is computed here from each report's own Brier and the marginal "
            "Brier it prints, against "
            "<code>limits.MIN_BRIER_SKILL_OVER_MARGINAL</code>.</small></p>"
        )
    return (
        f'<section class="run" id="run-{spec.key}">'
        f"<h3>{html.escape(spec.title)} {pill}</h3>"
        f'<p class="run-meta">{" · ".join(meta)}</p>'
        f"<p>{inline_markdown(spec.summary, 'docs/ledger.md', path)}</p>"
        f"{table}{extra}"
        '<div class="legend" aria-hidden="true"><span><i class="swatch cal"></i>gated</span>'
        '<span><i class="swatch raw"></i>raw</span>'
        '<span><i class="swatch diag"></i>perfect calibration</span></div>'
        f'<div class="reliability">{diagrams}</div>'
        f"{gates}</section>"
    )


def render_disproved(section: str, path: str) -> str:
    header, rows = markdown_table(section)
    if not rows:
        return render_markdown(section, "docs/ledger.md", path, drop_h1=False).body
    intro = ledger_prose(section)
    cards = []
    for cells in rows:
        belief, said = (cells + ["", ""])[:2]
        cards.append(
            '<div class="contrast">'
            f'<p class="was"><span class="tag">{html.escape(header[0])}</span>'
            f'<span class="struck">{inline_markdown(belief, "docs/ledger.md", path)}</span></p>'
            f'<p class="is"><span class="tag">{html.escape(header[1])}</span>'
            f"{inline_markdown(said, 'docs/ledger.md', path)}</p></div>"
        )
    lead = f"<p>{inline_markdown(intro, 'docs/ledger.md', path)}</p>" if intro else ""
    return f'{lead}<div class="contrast-grid">{"".join(cards)}</div>'


def render_favour(section: str, path: str) -> str:
    items = ledger_bullets(section)
    intro = ledger_prose(section)
    lead = f"<p>{inline_markdown(intro, 'docs/ledger.md', path)}</p>" if intro else ""
    lis = "".join(f"<li>{inline_markdown(i, 'docs/ledger.md', path)}</li>" for i in items)
    return f'{lead}<ol class="favour">{lis}</ol>'


def build_proofs(out: Path) -> str:
    path = "proofs/index.html"
    sections = ledger_sections()
    gate = limits.CALIBRATION_GATES["workhorse_max_ece"]
    ledger_href = rel(path, "docs/ledger.html")
    subs = [
        ("floors", "How to read a number"),
        ("claims", "What the tests pin"),
        ("disproved", "Believed, then disproved"),
        ("favour", "Corrected in our own favour"),
        ("runs", "Seed spreads"),
        ("measured", "Measured"),
        ("confirmed", "Confirmed the hard way"),
        ("open", "Still open"),
    ]
    toc = "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in subs)
    parts = [
        '<p class="doc-kicker">Evidence and examples</p>',
        "<h1>Proofs</h1>",
        "<p>The argument that the claims hold: each one beside the test that pins it or "
        "the report that measured it, every certified run as the spread of its seeds, "
        "every ECE beside its noise floor, and the record of what was believed and "
        "turned out to be wrong. Tables and diagrams are generated from the committed "
        "files at build time; the ledger's sections appear as they stand in "
        f'<a href="{ledger_href}"><code>docs/ledger.md</code></a>.</p>',
        f'<ul class="page-toc">{toc}</ul>',
        '<h2 id="floors">How to read a number here</h2>',
        "<p>An ECE on its own is not evidence: both estimators are biased upward at small "
        "samples by about as much as a gate. So every report simulates the ECE a "
        "<strong>perfectly calibrated</strong> model would score on the same answers, and "
        "the gates check the measurement before the model.</p>",
        '<div class="doc-table"><table><thead><tr><th scope="col">Rule</th>'
        '<th scope="col" class="num">Value</th><th scope="col">Source</th></tr></thead><tbody>'
        f'<tr><td>Workhorse ECE and adaptive ECE gate</td><td class="num">≤ {gate}</td>'
        '<td><code>CALIBRATION_GATES["workhorse_max_ece"]</code></td></tr>'
        f'<tr><td>Fewest scored answers an ECE may be quoted from</td><td class="num">'
        f"{limits.MIN_CALIBRATION_SAMPLES:,}</td><td><code>MIN_CALIBRATION_SAMPLES</code></td></tr>"
        f'<tr><td>Floor p95 may be at most this fraction of the gate</td><td class="num">'
        f"{limits.MAX_FLOOR_FRACTION_OF_GATE}</td><td><code>MAX_FLOOR_FRACTION_OF_GATE</code></td></tr>"
        f'<tr><td>Accuracy over the marginal predictor</td><td class="num">≥ '
        f"{limits.MIN_ACCURACY_OVER_BASELINE}</td><td><code>MIN_ACCURACY_OVER_BASELINE</code></td></tr>"
        f'<tr><td>Brier skill over the marginal, drawn-annotator corpora</td><td class="num">≥ '
        f"{limits.MIN_BRIER_SKILL_OVER_MARGINAL}</td>"
        "<td><code>MIN_BRIER_SKILL_OVER_MARGINAL</code></td></tr>"
        "</tbody></table></div>",
        f'<p>All from <a href="{blob_url("src/trigon/limits.py")}">'
        "<code>src/trigon/limits.py</code></a>, "
        "imported by this page's build. In the tables below, <em>floor p95</em> is the 95th "
        "percentile of that simulated ECE, and <em>within floor</em> means the measured ECE "
        "cannot be told apart from a perfectly calibrated model's.</p>",
        '<h2 id="claims">What the tests pin</h2>',
        "<p>The architectural claims are properties of the layout, asserted against the "
        "reference model; the process claims are asserted against the gates and the "
        "docs. Each test link goes to the line that defines it, and the build fails if "
        "a cited test no longer exists.</p>",
        render_claims(path),
        '<h2 id="disproved">Believed, then disproved</h2>',
        render_disproved(sections.get("Believed, then disproved", ""), path),
        '<h2 id="favour">Corrected in our own favour</h2>',
        render_favour(sections.get("Corrected in our own favour", ""), path),
        '<h2 id="runs">Seed spreads</h2>',
        "<p>A single-seed run is one sample from a distribution nobody measured, so a "
        "configuration is certified on its median and range across four seeds, never "
        "on its best draw. <em>Blocking gates</em> is the verdict as the run was gated; "
        "<em>every gate</em> counts the advisory ones too, which is what the "
        "per-question and per-primitive gates are for backbone runs now that they block. "
        "Each diagram plots a report's own bins: filled points are the gated answers, "
        "hollow ones the raw model, and a point's size follows its bin's count. Hover a "
        "point for its numbers.</p>",
        "".join(render_run(spec, path) for spec in RUNS),
        '<h2 id="measured">Measured</h2>',
        render_markdown(sections.get("Measured", ""), "docs/ledger.md", path, drop_h1=False).body,
        '<h2 id="confirmed">Confirmed the hard way</h2>',
        render_markdown(
            sections.get("Confirmed the hard way", ""), "docs/ledger.md", path, drop_h1=False
        ).body,
        '<h2 id="open">Still open</h2>',
        render_markdown(sections.get("Open", ""), "docs/ledger.md", path, drop_h1=False).body,
    ]
    body = shell(path, "\n".join(parts), subs=subs, wide=True)
    (out / path).parent.mkdir(parents=True, exist_ok=True)
    (out / path).write_text(
        page(
            path=path,
            title="Proofs — trigon",
            description=(
                "Each claim beside the test that pins it or the report that measured it, "
                "seed spreads, reliability diagrams and noise floors."
            ),
            body=body,
            section="proofs",
            current=path,
            stylesheets=("docs.css",),
        )
    )
    return path


# ---------------------------------------------------------------------------
# The landing page.


def _readme_block(heading: str, lang: str) -> str:
    """The first ``lang`` code block after a README heading (or anywhere)."""
    text = (ROOT / "README.md").read_text()
    if heading:
        text = text.split(f"## {heading}", 1)[-1]
    m = re.search(rf"```{lang}\n(.*?)```", text, re.DOTALL)
    if not m:
        raise LookupError(f"README.md has no {lang} block under {heading!r}")
    return m.group(1).rstrip("\n")


def build_index(out: Path, metrics: dict[str, Metric]) -> str:
    from trigon.evals.jaggedness.benchmarks import all_benchmarks

    path = "index.html"
    m = {k: v.html() for k, v in metrics.items()}
    proofs = "proofs/index.html"
    non_goals = [b for b in all_benchmarks() if b.non_goal]
    non_goal_list = ", ".join(
        f"<code>{html.escape(b.name)}</code> ({html.escape(b.failure_mode)})" for b in non_goals
    )
    opens = "".join(
        f'<div class="card"><h3>{inline_markdown(t, "docs/ledger.md", path)}</h3>'
        f"<p>{inline_markdown(s, 'docs/ledger.md', path)}</p></div>"
        for t, s in open_items()
    )

    stats = f"""
    <div class="stats">
      <article class="card stat" id="banking77">
        <p class="stat-corpus">Banking77 · 77 intents</p>
        <p class="stat-figure">{m["b77_accuracy_median"]}</p>
        <p class="stat-caption">median accuracy over four seeds, range
          {m["b77_accuracy_min"]}–{m["b77_accuracy_max"]}; ignoring the input scores
          {m["b77_baseline_median"]}</p>
        <dl>
          <dt>ECE, median</dt><dd>{m["b77_ece_median"]}
            (range {m["b77_ece_min"]}–{m["b77_ece_max"]})</dd>
          <dt>Noise floor</dt><dd>p95 {m["b77_floor_p95_min"]}–{m["b77_floor_p95_max"]}</dd>
          <dt>Gates</dt><dd>{m["b77_seeds_blocking"]} of {m["b77_seeds"]} seeds clear every
            blocking gate</dd>
        </dl>
        <p class="evidence">
          <a href="{blob_url("reports/banking77/README.md")}">reports/banking77</a>
          <a href="{proofs}#run-banking77-qwen">seed spread →</a>
        </p>
      </article>
      <article class="card stat" id="synthetic">
        <p class="stat-corpus">Synthetic suite · 3 questions</p>
        <p class="stat-figure">{m["syn_size_lift_min"]}</p>
        <p class="stat-caption">lift over its own marginal on <code>size</code>, the worst of
          four seeds (best {m["syn_size_lift_max"]}): the question seven interventions on the
          spike never moved</p>
        <dl>
          <dt>Accuracy, median</dt><dd>{m["syn_accuracy_median"]}</dd>
          <dt>Worst-head ECE</dt><dd>median {m["syn_worst_primitive_ece_median"]}
            (range {m["syn_worst_primitive_ece_min"]}–{m["syn_worst_primitive_ece_max"]})</dd>
          <dt>Noise floor</dt><dd>p95 {m["syn_floor_p95_min"]}–{m["syn_floor_p95_max"]}</dd>
          <dt>Gates</dt><dd>{m["syn_seeds_every_gate"]} of {m["syn_seeds"]} seeds clear every
            gate, the per-primitive one included</dd>
        </dl>
        <p class="evidence">
          <a href="{blob_url("reports/synthetic/README.md")}">reports/synthetic</a>
          <a href="{proofs}#run-synthetic-qwen">seed spread →</a>
        </p>
      </article>
      <article class="card stat" id="helpsteer2">
        <p class="stat-corpus">HelpSteer2 · annotator ratings</p>
        <p class="stat-figure">{m["hs2_brier_skill_median"]}</p>
        <p class="stat-caption">Brier skill over the marginal, median of four seeds (range
          {m["hs2_brier_skill_min"]} to {m["hs2_brier_skill_max"]}) against a blocking
          {m["hs2_brier_skill_limit"]}</p>
        <dl>
          <dt>ECE</dt><dd>{m["hs2_ece_min"]}–{m["hs2_ece_max"]} against a random annotator</dd>
          <dt>Noise floor</dt><dd>p95 {m["hs2_floor_p95_min"]}–{m["hs2_floor_p95_max"]};
            {m["hs2_seeds_within_floor"]} of {m["hs2_seeds"]} seeds within it</dd>
          <dt>Raw ECE</dt><dd>{m["hs2_raw_ece_soft_median"]} trained on distributions,
            {m["hs2_raw_ece_hard_median"]} on majority votes (medians)</dd>
          <dt>Gates</dt><dd>{m["hs2_seeds_blocking"]} of {m["hs2_seeds"]} seeds clear every
            blocking gate</dd>
        </dl>
        <p class="evidence">
          <a href="{blob_url("reports/helpsteer2-annotators/README.md")}"
            >reports/helpsteer2-annotators</a>
          <a href="{proofs}#run-helpsteer2-annotators">seed spread →</a>
        </p>
      </article>
    </div>"""

    body = f"""<main id="main">

  <section class="hero">
    <div class="wrap">
      <span class="hero-badge"><span class="hero-dot"></span>
        Phase 0 · the loop closes · certified on real data</span>
      <h1 class="display hero-h1">
        Typed decisions in one forward pass,<br />
        <em>with probabilities you can believe.</em>
      </h1>
      <p class="lede hero-lede">
        <strong>trigon</strong> is a prefill-only, typed, calibrated decision model.
        Send state plus a map of typed questions and get one answer per question: a full
        distribution over exactly the labels you declared, in a single pass, with no
        decode loop, no sampler and nothing to parse. <strong>Calibration is the
        product</strong>: releases are gated on ECE, and every ECE is published beside
        the noise floor that says whether it means anything.
      </p>
      <div class="hero-cta">
        <a class="btn primary" href="{proofs}">Read the proofs →</a>
        <a class="btn ghost" href="docs/index.html">Read the docs</a>
        <a class="btn ghost" href="examples/index.html">See examples</a>
      </div>
      <ul class="chips">
        <li>Choice · Score · Noul</li>
        <li>Core depends on <code>pydantic</code> only</li>
        <li>Qwen2.5-1.5B + LoRA</li>
        <li>OpenAPI contract</li>
        <li>Apache-2.0</li>
      </ul>
    </div>
  </section>

  <section class="band" id="results">
    <div class="wrap">
      <span class="eyebrow">Measured, not claimed</span>
      <h2 class="display band-h2">Four seeds, <em>and the floor under each.</em></h2>
      <p class="lede">
        Qwen2.5-1.5B with LoRA adapters, under the same layout, mask and heads as the
        CPU-sized reference model. Each figure is a four-seed median or range, and each
        ECE sits beside the ECE a perfectly calibrated model would score on the same
        number of answers.
      </p>
      {stats}
      <p class="fineprint">
        <strong>Computed, not typed.</strong> Every number in this band is read from the
        committed per-seed JSON under <code>reports/</code> when the site is built, and
        <code>tests/test_site.py</code> recomputes it. Banking77 is the certified sweep
        re-gated under the current calibrator rule (same weights, no retraining). An ECE
        is quoted only from runs of at least {limits.MIN_CALIBRATION_SAMPLES:,} answers.
        The reference spike's own runs, and a configuration that did not certify, are on
        <a href="{proofs}#runs">the proofs page</a>.
      </p>
    </div>
  </section>

  <section class="band">
    <div class="wrap">
      <span class="eyebrow">What it does</span>
      <h2 class="display band-h2">Nothing to validate,
        <em>because nothing can be wrong-shaped.</em></h2>
      <div class="card-grid" style="margin-top: 44px">
        <div class="card">
          <h3>Three primitives</h3>
          <p><strong>Choice</strong> picks one option from a declared set, <strong>Score</strong>
          places an answer on ordered levels, <strong>Noul</strong> gives the probability of
          yes. Each head is a softmax over exactly what the request declared, so schema
          violations are unrepresentable rather than caught.</p>
        </div>
        <div class="card">
          <h3>Questions that cannot see each other</h3>
          <p>A block mask gives no path from one question to another. Adding, removing or
          reordering questions moves no other answer, and
          <code>tests/test_independence.py</code> asserts it against a real model. Up to
          {limits.MAX_QUESTIONS_PER_REQUEST:,} questions and
          {limits.MAX_OPTIONS_PER_QUESTION:,} options per Choice, in one pass.</p>
        </div>
        <div class="card">
          <h3>The schema is a cached prefix</h3>
          <p>The schema encodes identically whatever the state, so its KV prefix is keyed on
          the schema and reused across requests. On by default at the gateway, never in
          training, and <code>/healthz</code> says which way it is set.</p>
        </div>
        <div class="card">
          <h3>A calibrator is a proposal</h3>
          <p>A temperature and an isotonic map are fitted per primitive, scored on a slice
          neither saw, and applied only if one demonstrably helps. Otherwise the head ships
          raw, and the report says so.</p>
        </div>
        <div class="card">
          <h3>Gates that fail a useless model</h3>
          <p>A model that reports each question's marginal is calibrated by construction.
          Every gate set keeps a term that rejects it: accuracy over the marginal predictor,
          or the Brier skill over the marginal where labels are drawn annotators.</p>
        </div>
        <div class="card">
          <h3>Honest by default</h3>
          <p>An uncalibrated or untrained deployment is allowed; a silent one is not.
          Responses name the build that answered them, and benchmarks for things this does
          not do are published anyway.</p>
        </div>
      </div>
    </div>
  </section>

  <section class="band" id="quickstart">
    <div class="wrap band-inner">
      <div>
        <span class="eyebrow">Quickstart</span>
        <h2 class="display band-h2">Runs on a fresh clone, <em>no weights, no GPU.</em></h2>
        <p class="lede">
          The lexical floor backend is a real baseline, so the whole path runs before any
          model exists: compile, answer, calibrate, serve. Its answers are a floor, not a
          model's; point <code>--weights</code> at a trained checkpoint for those.
        </p>
        <a class="btn primary" href="docs/index.html">Read the introduction →</a>
      </div>
      <div>
        {code_block(_readme_block("Install and run", "bash"), "bash", "band-code")}
        {code_block(_readme_block("", "python"), "python", "band-code")}
      </div>
    </div>
  </section>

  <section class="band" id="honest">
    <div class="wrap">
      <span class="eyebrow">Where it stops</span>
      <h2 class="display band-h2">The things it is <em>not</em>, and what is still open.</h2>
      <p class="lede" style="margin-top: 18px">
        Stated here rather than discovered later. The open items are read from the
        ledger's <a href="docs/ledger.html#open">Open</a> section when the site is built.
      </p>
      <div class="card-grid" style="margin-top: 40px">
        <div class="card">
          <h3>Not a generator</h3>
          <p>No text generation, no image or audio input, no multi-turn state. It answers
          typed questions about one state in one pass.</p>
        </div>
        <div class="card">
          <h3>Not arithmetic, counting or dates</h3>
          <p>Keep math in code. The non-goals are measured and published anyway:
          {non_goal_list}.</p>
        </div>
        <div class="card">
          <h3>Not state of the art on accuracy</h3>
          <p>Banking77 is a solved benchmark. The claim is that the pipeline certifies a
          well-calibrated model when given real data, not that 1.5B parameters set a
          record.</p>
        </div>
        <div class="card">
          <h3>Not published anywhere</h3>
          <p>Not on a package registry and not a GitHub Release yet. The certified adapter
          is served behind an API key; it works end to end and is tested, which is a
          different claim from production-ready.</p>
        </div>
      </div>
      <h3 class="eyebrow" style="margin-top: 56px">Open in the ledger</h3>
      <div class="card-grid">
        {opens}
      </div>
    </div>
  </section>

</main>"""
    (out / path).write_text(
        page(
            path=path,
            title="trigon — typed, calibrated decisions in one forward pass",
            description=(
                "A prefill-only, typed, calibrated decision model where calibration is the "
                "product: every ECE published beside its noise floor, every claim beside "
                "the test that pins it."
            ),
            body=body,
            section="Home",
            current="",
            stylesheets=("home.css",),
        )
    )
    return path


# ---------------------------------------------------------------------------


MARKER = ".trigon-site"


def build(out: Path) -> list[str]:
    """Render the whole site into ``out`` and return the pages written."""
    out = out.resolve()
    if out.exists() and any(out.iterdir()) and not (out / MARKER).exists():
        raise SystemExit(f"refusing to overwrite {out}: it is not a previous site build")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("generated by scripts/build_site.py\n")
    (out / ".nojekyll").write_text("")
    for name in ("theme.css", "home.css", "docs.css"):
        shutil.copy2(SITE_SRC / name, out / name)
    shutil.copytree(SITE_SRC / "fonts", out / "fonts")

    LINKS.dangling.clear()
    metrics = compute_headlines()
    pages = [build_index(out, metrics)]
    pages += build_docs(out)
    pages.append(build_examples(out))
    pages.append(build_api(out))
    pages.append(build_proofs(out))
    (out / "headlines.json").write_text(
        json.dumps({k: v.value for k, v in metrics.items()}, indent=2, sort_keys=True) + "\n"
    )
    return pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "_site"), help="output directory")
    args = parser.parse_args(argv)
    pages = build(Path(args.out))
    print(f"wrote {len(pages)} pages to {args.out}")
    for source, target in LINKS.dangling:
        print(f"  note: {source} links to {target}, which is not in the repository")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
