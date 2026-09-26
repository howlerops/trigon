"""The incumbent's names appear nowhere in this project but where a drop-in needs them.

The owner's decision of 2026-09-26 (`docs/decisions.md`, Q23): this is a
functional drop-in for an existing typed-decision API, and it is not named
after it. Nothing here should read as that product, its company or its model
-- a naming conflict is not worth a sentence of copy.

The one exception is functional. A migration changes a base URL and nothing
else only if the compatibility route's path is exactly theirs, so that path
is written once, as `trigon.server.compat.COMPAT_PATH`, and appears in the
generated spec that documents it. Every other tracked file is scanned.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Assembled, so this file does not match what it scans for.
_PRODUCT = "system" + r"[\s_-]?" + "one"
_COMPANY = "type" + "safe"
_MODEL = r"\bj" + r"ev\b"
NAMES = re.compile(f"{_PRODUCT}|{_COMPANY}|{_MODEL}", re.IGNORECASE)

# Where the compatibility route's path may appear, and only as that path.
PATH_ALLOWED = {"src/trigon/server/compat.py", "spec/openapi.json"}
PATH_FORMS = re.compile(r"/v1/" + "system" + "one")


def _tracked() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.decode().split("\0") if p]


def test_no_tracked_file_names_the_incumbent():
    found = []
    for rel in _tracked():
        path = ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if rel in PATH_ALLOWED:
            text = PATH_FORMS.sub("", text)
        for match in NAMES.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            found.append(f"{rel}:{line}: {match.group(0)!r}")
    assert not found, "\n".join(found)


def test_the_compat_path_is_still_theirs():
    """The guard must not be satisfied by breaking the drop-in."""
    from trigon.server.compat import COMPAT_PATH

    assert PATH_FORMS.fullmatch(COMPAT_PATH)
