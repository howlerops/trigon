"""Fetch a pretrained backbone's files at a pinned revision, with the stdlib.

A checkpoint trained on "Qwen2.5-1.5B" names nothing: the repository moves,
and a tokenizer or weight file that changed underneath a fine-tune gives a
model that loads cleanly and reads every token as a different word. So every
fetch here names a **commit**, and the cache is keyed on it.

Plain HTTPS through `urllib` rather than `huggingface_hub`, for the reason the
corpus loaders are plain-file: the tokenizer half of this is imported by the
compiler's estimator path, and a download client with its own dependency tree
has no business there.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import urllib.request

__all__ = ["BACKBONES", "Backbone", "cache_root", "fetch"]


class Backbone:
    """A pretrained base, pinned to one revision."""

    def __init__(self, name: str, repo: str, revision: str, licence: str) -> None:
        self.name = name
        self.repo = repo
        self.revision = revision
        self.licence = licence

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Backbone({self.name!r}, {self.repo}@{self.revision[:8]})"


BACKBONES: dict[str, Backbone] = {
    "qwen2.5-1.5b": Backbone(
        "qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B",
        "8faed761d45a263340a0528343f099c05c9a4323",
        "Apache-2.0",
    ),
    "qwen2.5-0.5b": Backbone(
        "qwen2.5-0.5b",
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "Apache-2.0",
    ),
}


def cache_root() -> pathlib.Path:
    configured = os.environ.get("TRIGON_WEIGHTS_CACHE")
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path.home() / ".cache" / "trigon" / "backbones"


def fetch(backbone: Backbone, filename: str) -> pathlib.Path:
    """The local path of one file at the pinned revision, downloading it once."""
    target = cache_root() / backbone.repo.replace("/", "--") / backbone.revision / filename
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{backbone.repo}/resolve/{backbone.revision}/{filename}"
    partial = target.with_suffix(target.suffix + ".partial")
    # Written to a partial file and renamed, so an interrupted download of a
    # 3 GB weight file is never mistaken for a complete one on the next run.
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.rename(target)
    return target
