"""A published release's checksums name the committed files they came from.

`releases/*/SHA256SUMS` is what a downloader checks the bundle against, and
the bundle's calibrators and report are copies of files under `reports/`. If
one of those is regenerated and the release is not, the checksum in the
repository and the file beside it describe different things, and nothing
else would say so.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

SOURCES = {
    "banking77-qwen15b-v1": {
        "temperatures.json": "reports/banking77/qwen15b-e4-lr1e-4-seed2-temperatures.json",
        "isotonic.json": "reports/banking77/qwen15b-e4-lr1e-4-seed2-isotonic.json",
        "report.md": "reports/banking77/qwen15b-e4-lr1e-4-seed2.md",
        "report.json": "reports/banking77/qwen15b-e4-lr1e-4-seed2.json",
        "modal-run.json": "reports/banking77/qwen15b-e4-lr1e-4-modal-run.json",
        # Weights are not committed (`*.pt` is ignored); the adapter's
        # checksum is checked only where a local copy exists.
        "adapter.pt": "reports/banking77/qwen15b-e4-lr1e-4-seed2.pt",
    }
}


def _entries(release: str) -> list[tuple[str, str]]:
    path = ROOT / "releases" / release / "SHA256SUMS"
    return [tuple(line.split()[::-1]) for line in path.read_text().splitlines() if line]


@pytest.mark.parametrize("release", sorted(SOURCES))
def test_every_released_file_has_a_committed_source(release):
    assert {name for name, _ in _entries(release)} == set(SOURCES[release])


@pytest.mark.parametrize(
    ("release", "name", "digest"),
    [(r, n, d) for r in sorted(SOURCES) for n, d in _entries(r)],
)
def test_the_checksum_matches_the_committed_file(release, name, digest):
    source = ROOT / SOURCES[release][name]
    if not source.exists():
        pytest.skip(f"{source.name} is not committed and no local copy is present")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("release", sorted(SOURCES))
def test_the_bundle_checksum_is_what_the_release_workflow_checks(release):
    """`.github/workflows/release.yml` runs `sha256sum -c` on this file against
    the bundle it fetched from the Modal Volume, so it must name exactly that
    bundle, `<release>.tar.gz`, with a full SHA-256."""
    line = (ROOT / "releases" / release / "BUNDLE.sha256").read_text().strip()
    digest, name = line.split()
    assert name == f"{release}.tar.gz"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
