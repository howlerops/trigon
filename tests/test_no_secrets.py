"""No credential is committed, and no file names one by value.

This repository deploys to Modal, trains on Modal and serves behind an API
key, so it is handled by people and sessions that hold real credentials. They
live in environment variables and in Modal Secrets, never in the tree. This
test is the tripwire for the day one of them lands in a file anyway: it scans
every tracked file for the shapes those credentials take.

It checks shapes, not values, so it cannot leak a value by containing it. A
pattern that matches something legitimate goes in ``ALLOWED`` with the reason,
rather than the pattern being loosened.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

PATTERNS = {
    "Modal token id": re.compile(r"\bak-[A-Za-z0-9]{16,}"),
    "Modal token secret": re.compile(r"\bas-[A-Za-z0-9]{16,}"),
    "Modal proxy key": re.compile(r"\bwk-[A-Za-z0-9]{16,}"),
    "Modal proxy secret": re.compile(r"\bws-[A-Za-z0-9]{16,}"),
    "GitHub token": re.compile(
        r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    ),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer literal": re.compile(r"Bearer [A-Za-z0-9_\-]{24,}"),
}

# (path, pattern name): why the match is not a credential.
ALLOWED: dict[tuple[str, str], str] = {}


def _tracked() -> list[pathlib.Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / p for p in out.decode().split("\0") if p]


def test_no_tracked_file_contains_a_credential():
    found = []
    for path in _tracked():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary: weights, fonts
        rel = path.relative_to(ROOT).as_posix()
        for name, pattern in PATTERNS.items():
            if (rel, name) in ALLOWED:
                continue
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                # The location only: printing the match would print the secret.
                found.append(f"{rel}:{line}: looks like a {name}")
    assert not found, "\n".join(found)


def test_the_patterns_fire():
    """A tripwire that never fires is not one."""
    samples = {
        "Modal token id": "ak-" + "A1b2C3d4E5f6G7h8",
        "GitHub token": "ghp_" + "a" * 36,
        # Assembled, so this file does not trip the scan it is testing.
        "private key": "-----BEGIN RSA " + "PRIVATE KEY-----",
    }
    for name, sample in samples.items():
        assert PATTERNS[name].search(sample), name
