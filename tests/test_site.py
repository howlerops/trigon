"""The GitHub Pages site: built, linked, self-contained, and honest about its numbers.

`scripts/build_site.py` renders the site from the files that hold each fact.
These tests build it into a temporary directory and check the four things that
would otherwise rot silently:

- every internal link and asset resolves to a file the build wrote;
- every `docs/*.md` has a page, or is excluded by name with a reason;
- no page reaches for a script, stylesheet or font on another origin;
- the headline numbers on the landing page are the ones the committed reports
  hold -- recomputed here from the JSON, not read back from the builder.

Skipped when `markdown` (the `site` extra) is not installed.
"""

from __future__ import annotations

import html
import importlib.util
import json
import pathlib
import re
import statistics
import sys
from urllib.parse import unquote, urlsplit

import pytest

pytest.importorskip("markdown")

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
build_site = importlib.util.module_from_spec(_spec)
# Registered first: its dataclasses resolve their module through sys.modules.
sys.modules.setdefault("build_site", build_site)
_spec.loader.exec_module(build_site)


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("site") / "_site"
    build_site.build(out)
    return out


def _pages(site: pathlib.Path) -> list[pathlib.Path]:
    return sorted(site.rglob("*.html"))


_REF = re.compile(r'\b(href|src)="([^"]*)"')


def _refs(page: pathlib.Path) -> list[tuple[str, str]]:
    return [(a, html.unescape(v)) for a, v in _REF.findall(page.read_text())]


def test_every_internal_href_and_src_resolves_to_a_built_file(site):
    broken = []
    for page in _pages(site):
        for _, ref in _refs(page):
            parts = urlsplit(ref)
            if parts.scheme or ref.startswith("//"):
                continue
            if not parts.path:  # a same-page anchor
                continue
            target = (page.parent / unquote(parts.path)).resolve()
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file() or site.resolve() not in target.parents:
                broken.append(f"{page.relative_to(site)} -> {ref}")
    assert not broken, "\n".join(broken)


def test_same_page_anchors_exist(site):
    missing = []
    for page in _pages(site):
        text = page.read_text()
        ids = set(re.findall(r'\bid="([^"]+)"', text))
        for _, ref in _refs(page):
            if ref.startswith("#") and len(ref) > 1 and unquote(ref[1:]) not in ids:
                missing.append(f"{page.relative_to(site)} -> {ref}")
    assert not missing, "\n".join(missing)


def test_css_urls_resolve(site):
    for sheet in site.glob("*.css"):
        for url in re.findall(r"url\(['\"]?([^'\")]+)", sheet.read_text()):
            if url.startswith("data:"):
                continue
            assert (sheet.parent / url).is_file(), f"{sheet.name}: {url}"


def test_every_doc_has_a_page_or_a_stated_reason(site):
    pages = {d.source for d in build_site.DOCS}
    excluded = build_site.DOCS_EXCLUDED
    for reason in excluded.values():
        assert reason.strip(), "an exclusion needs a reason"
    missing = []
    for md in sorted((ROOT / "docs").glob("*.md")):
        source = f"docs/{md.name}"
        if source in excluded:
            continue
        if source not in pages:
            missing.append(source)
            continue
        doc = build_site.DOC_BY_SOURCE[source]
        assert (site / doc.output).is_file(), source
    assert not missing, f"docs with neither a page nor an exclusion: {missing}"
    assert (site / "docs/index.html").is_file(), "README.md is the introduction"


def test_a_link_to_another_doc_goes_to_its_page_and_the_rest_to_github(site):
    intro = (site / "docs/index.html").read_text()
    # README links docs/decisions.md; it must land on the built page.
    assert 'href="decisions.html"' in intro
    assert "docs/decisions.md" not in re.findall(r'href="([^"]*)"', intro)
    repo_links = [r for _, r in _refs(site / "docs/evals.html") if "github.com" in r]
    assert repo_links, "evals.md names repository files; they should link to GitHub"
    prefix = f"{build_site.GITHUB}/"
    for ref in repo_links:
        if ref.startswith(prefix + "blob/") or ref.startswith(prefix + "tree/"):
            assert f"/{build_site.BRANCH}/" in ref, ref


def test_no_page_references_an_external_script_stylesheet_or_font(site):
    offenders = []
    for page in _pages(site):
        text = page.read_text()
        for tag in re.findall(r"<script\b[^>]*>", text):
            offenders.append(f"{page.name}: {tag}")  # no scripts at all
        for tag in re.findall(r"<link\b[^>]*>", text):
            href = re.search(r'href="([^"]*)"', tag)
            if href and urlsplit(href.group(1)).scheme in ("http", "https", ""):
                if href.group(1).startswith(("http:", "https:", "//")):
                    offenders.append(f"{page.name}: {tag}")
        for url in re.findall(r"url\(['\"]?(https?:|//)", text):
            offenders.append(f"{page.name}: url({url}")
        for tag in re.findall(r"<(?:img|iframe|video|audio|source)\b[^>]*>", text):
            if re.search(r'src="(https?:|//)', tag):
                offenders.append(f"{page.name}: {tag}")
    for sheet in site.rglob("*.css"):
        text = sheet.read_text()
        if re.search(r"@import|url\(['\"]?(https?:|//)", text):
            offenders.append(sheet.name)
    assert not offenders, "\n".join(offenders)


# ---------------------------------------------------------------------------
# The landing page's numbers, recomputed independently of the builder. Only
# *which* files are certified is taken from it (RUNS); the arithmetic is here.


def _gated(path: str) -> dict:
    data = json.loads((ROOT / path).read_text())
    name = data["gated"]
    return next(r for r in data["results"] if r["suite"] == name)


def _paths(key: str) -> list[str]:
    spec = build_site.RUN_BY_KEY[key]
    return [spec.path(s) for s in spec.seeds]


def _shown(site: pathlib.Path) -> dict[str, tuple[float, str]]:
    text = (site / "index.html").read_text()
    found = re.findall(r'data-metric="([^"]+)" data-value="([^"]+)">([^<]*)<', text)
    assert found, "the landing page carries no metrics"
    return {k: (float(v), html.unescape(t)) for k, v, t in found}


def test_the_landing_page_numbers_are_the_reports(site):
    shown = _shown(site)

    def check(key: str, expected: float, text: str) -> None:
        assert key in shown, f"{key} is not on the landing page"
        value, rendered = shown[key]
        assert value == pytest.approx(expected, abs=1e-12), key
        assert rendered == text, f"{key}: shows {rendered!r}, reports say {text!r}"

    b77 = [_gated(p) for p in _paths("banking77-qwen")]
    assert len(b77) == 4
    acc = [r["accuracy"] for r in b77]
    ece = [r["calibration"]["ece"] for r in b77]
    p95 = [r["calibration"]["floor"]["p95"] for r in b77]
    check("b77_accuracy_median", statistics.median(acc), f"{statistics.median(acc) * 100:.1f}%")
    check("b77_accuracy_min", min(acc), f"{min(acc) * 100:.1f}%")
    check("b77_accuracy_max", max(acc), f"{max(acc) * 100:.1f}%")
    check("b77_ece_median", statistics.median(ece), f"{statistics.median(ece):.4f}")
    check("b77_ece_min", min(ece), f"{min(ece):.4f}")
    check("b77_ece_max", max(ece), f"{max(ece):.4f}")
    check("b77_floor_p95_min", min(p95), f"{min(p95):.4f}")
    check("b77_floor_p95_max", max(p95), f"{max(p95):.4f}")
    passed = sum(json.loads((ROOT / p).read_text())["passed"] for p in _paths("banking77-qwen"))
    check("b77_seeds_blocking", passed, str(passed))

    syn = [_gated(p) for p in _paths("synthetic-qwen")]
    size = [r["per_question"]["size"]["lift"] for r in syn]
    check("syn_size_lift_min", min(size), f"{min(size):+.4f}")
    check("syn_size_lift_max", max(size), f"{max(size):+.4f}")
    worst = [max(v["ece"] for v in r["per_primitive"].values()) for r in syn]
    check(
        "syn_worst_primitive_ece_median",
        statistics.median(worst),
        f"{statistics.median(worst):.4f}",
    )

    # HelpSteer2's Brier skill: 1 - Brier / the marginal Brier the report prints.
    skills = []
    for p in _paths("helpsteer2-annotators"):
        md = (ROOT / p).with_suffix(".md").read_text()
        marginal = float(re.search(r"\|\s*ignores its input\s*\|\s*([0-9.]+)", md).group(1))
        skills.append(1.0 - _gated(p)["calibration"]["brier"] / marginal)
    med = statistics.median(skills)
    check("hs2_brier_skill_median", med, f"{med:+.4f}")
    check("hs2_brier_skill_min", min(skills), f"{min(skills):+.4f}")
    check("hs2_brier_skill_max", max(skills), f"{max(skills):+.4f}")
    hs2_ece = [_gated(p)["calibration"]["ece"] for p in _paths("helpsteer2-annotators")]
    check("hs2_ece_min", min(hs2_ece), f"{min(hs2_ece):.4f}")
    check("hs2_ece_max", max(hs2_ece), f"{max(hs2_ece):.4f}")


def test_every_number_on_the_landing_page_is_a_computed_metric(site):
    """Each data-metric is one the builder computed, and the builder computed
    nothing the page leaves out -- a stale key would be a number nobody shows."""
    shown = _shown(site)
    computed = json.loads((site / "headlines.json").read_text())
    assert set(shown) == set(computed)
    for key, (value, _) in shown.items():
        assert value == pytest.approx(computed[key], abs=1e-12), key


def test_every_ece_on_the_landing_page_sits_beside_its_floor(site):
    shown = _shown(site)
    for prefix in ("b77", "syn", "hs2"):
        assert f"{prefix}_floor_p95_min" in shown and f"{prefix}_floor_p95_max" in shown


def test_every_certified_run_is_on_the_proofs_page_with_its_diagrams(site):
    proofs = (site / "proofs/index.html").read_text()
    for spec in build_site.RUNS:
        assert f'id="run-{spec.key}"' in proofs, spec.key
    # Four diagrams per run, drawn from bins the reports carry.
    assert proofs.count('<svg viewBox="0 0 200 200"') == 4 * len(build_site.RUNS)
    for heading in ("Believed, then disproved", "Corrected in our own favour"):
        assert heading in proofs


def test_every_cited_test_exists():
    for claim in build_site.CLAIMS:
        for test_file, names in claim.tests:
            for name in names:
                assert build_site.test_line(test_file, name) > 0
