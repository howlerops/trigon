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

import json
import os
import pathlib
import shutil
import urllib.request

__all__ = ["BACKBONES", "Backbone", "cache_root", "fetch", "weight_files"]


class Backbone:
    """A pretrained base, pinned to one revision."""

    def __init__(
        self, name: str, repo: str, revision: str, licence: str, *, sharded: bool = False
    ) -> None:
        self.name = name
        self.repo = repo
        self.revision = revision
        self.licence = licence
        #: Weights split across ``model-0000k-of-0000n.safetensors`` files named
        #: by ``model.safetensors.index.json``, as every Qwen2.5 from 7B up is,
        #: rather than one ``model.safetensors``. Declared rather than probed,
        #: so a single-file backbone never spends a request on a 404.
        self.sharded = sharded

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
    # Qwen3 (2025), post-trained: the instruction-following model, not -Base.
    # Full attention with a per-head QK-norm and its own head_dim
    # (`QwenShape`). The newer Qwen3.5 small models are three-quarters linear
    # attention, which the block mask cannot reach, so they are not here.
    "qwen3-0.6b": Backbone(
        "qwen3-0.6b",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "Apache-2.0",
    ),
    "qwen3-1.7b": Backbone(
        "qwen3-1.7b",
        "Qwen/Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "Apache-2.0",
        sharded=True,
    ),
    # Larger Qwen3 for the scale ladder and as teachers to distil from: the same
    # architecture as 0.6B/1.7B (8B with untied embeddings), too large to train
    # on a 64 GB machine, trained on rented GPUs (`deploy/hf/`).
    "qwen3-4b": Backbone(
        "qwen3-4b",
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
        "Apache-2.0",
        sharded=True,
    ),
    "qwen3-8b": Backbone(
        "qwen3-8b",
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        "Apache-2.0",
        sharded=True,
    ),
    # The knowledge tier's dense candidate: zero-shot 0.643 on the benchmark's
    # knowledge slices against 4B's 0.507 (`reports/scaling/`). LM-score only.
    "qwen3-14b": Backbone(
        "qwen3-14b",
        "Qwen/Qwen3-14B",
        "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "Apache-2.0",
        sharded=True,
    ),
    # MiniCPM5 (OpenBMB, 2026), post-trained, Apache-2.0 on the model card. Plain
    # Llama: full attention in every layer, no q/k/v bias, no QK-norm, head_dim
    # 128, untied embeddings -- all of which `QwenShape` already expresses. Its
    # tokenizer chains a digit split before the usual regex and has no
    # normalizer, which `ByteLevelBPE` covers.
    "minicpm5-1b": Backbone(
        "minicpm5-1b",
        "openbmb/MiniCPM5-1B",
        "87179e5c1f455ef22e6223592d2d61351b525bfc",
        "Apache-2.0",
        sharded=True,
    ),
    "minicpm5-2b": Backbone(
        "minicpm5-2b",
        "openbmb/MiniCPM5-2B",
        "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
        "Apache-2.0",
        sharded=True,
    ),
    # Four bf16 shards, 15.2 GB. Untied embeddings: `lm_head.weight` is its
    # own tensor here, and unused, as the tied one is on the smaller models.
    "qwen2.5-7b": Backbone(
        "qwen2.5-7b",
        "Qwen/Qwen2.5-7B",
        "d149729398750b98c0af14eb82c78cfe92750796",
        "Apache-2.0",
        sharded=True,
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


INDEX = "model.safetensors.index.json"


def weight_files(backbone: Backbone) -> list[pathlib.Path]:
    """Every safetensors file of the pinned revision, fetched, in shard order.

    A sharded checkpoint names its shards in the index's ``weight_map``; each
    is fetched once and cached like any other file. A name that is not a
    plain file name is refused rather than joined onto the cache path.
    """
    if not backbone.sharded:
        return [fetch(backbone, "model.safetensors")]
    index = json.loads(fetch(backbone, INDEX).read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        if pathlib.PurePosixPath(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"{backbone.name}: the index names an unexpected shard {shard!r}")
    return [fetch(backbone, shard) for shard in shards]
