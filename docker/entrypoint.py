"""Fetch and verify a release bundle if one is named, then run the command.

TRIGON_BUNDLE_BASE names a published bundle's directory -- on R2,
``https://<worker>/models/bundles/<name>/<version>/``, or on Hugging Face,
``https://huggingface.co/<repo>/resolve/<revision>/`` -- and
TRIGON_BUNDLE_SUMS_SHA256 the digest of its ``SHA256SUMS``: that one pinned
digest vouches for the list, and the list for every file, each checked as it
arrives. TRIGON_BUNDLE_URL names a release tarball (`releases/<name>/` in the
repository); TRIGON_BUNDLE_SHA256 is required with it and is checked before
anything is unpacked. Without a URL, a bundle mounted at /model (``adapter.pt``
and its calibrators) is served as it is. A model the image cannot verify is refused rather than
served, for the reason `.github/workflows/release.yml` refuses one: what runs
is exactly what the repository says it is. The bundle's adapter and
calibrators then configure `trigon serve` through the usual TRIGON_* variables,
unless those are already set.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import tarfile
import urllib.request

MODEL = pathlib.Path(os.environ.get("TRIGON_MODEL_DIR", "/model"))


def _fetch(url: str, expected: str) -> pathlib.Path:
    archive = MODEL / "bundle.tar.gz"
    digest = hashlib.sha256()
    print(f"trigon: fetching {url}", file=sys.stderr, flush=True)
    with urllib.request.urlopen(url) as response, archive.open("wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != expected.lower():
        archive.unlink()
        raise SystemExit(
            f"trigon: bundle sha256 {digest.hexdigest()} does not match TRIGON_BUNDLE_SHA256 "
            f"{expected}; refusing to serve it"
        )
    with tarfile.open(archive) as tar:
        tar.extractall(MODEL, filter="data")
    archive.unlink()
    adapters = sorted(MODEL.glob("*/adapter.pt"))
    if len(adapters) != 1:
        raise SystemExit(f"trigon: expected one adapter.pt in the bundle, found {len(adapters)}")
    return adapters[0].parent


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "trigon-entrypoint"})
    with urllib.request.urlopen(request) as response:  # noqa: S310
        return response.read()


def _fetch_published(base: str, sums_digest: str) -> pathlib.Path:
    """A published bundle: SHA256SUMS against its pinned digest, then each file against it."""
    base = base.rstrip("/") + "/"
    bundle = MODEL / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    print(f"trigon: fetching {base}", file=sys.stderr, flush=True)
    sums = _download(base + "SHA256SUMS")
    if hashlib.sha256(sums).hexdigest() != sums_digest.lower():
        raise SystemExit(
            "trigon: SHA256SUMS does not match TRIGON_BUNDLE_SUMS_SHA256; refusing to serve it"
        )
    for line in sums.decode().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if "/" in name or name.startswith("."):
            raise SystemExit(f"trigon: SHA256SUMS names {name!r}, outside the bundle")
        data = _download(base + name)
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"trigon: {name} does not match SHA256SUMS; refusing to serve it")
        (bundle / name).write_bytes(data)
    if not (bundle / "adapter.pt").exists():
        raise SystemExit("trigon: the published bundle has no adapter.pt")
    return bundle


def main() -> None:
    url = os.environ.get("TRIGON_BUNDLE_URL")
    base = os.environ.get("TRIGON_BUNDLE_BASE")
    if base:
        sums_digest = os.environ.get("TRIGON_BUNDLE_SUMS_SHA256")
        if not sums_digest:
            raise SystemExit("trigon: TRIGON_BUNDLE_BASE needs TRIGON_BUNDLE_SUMS_SHA256 beside it")
        bundle = _fetch_published(base, sums_digest)
        os.environ.setdefault("TRIGON_BACKEND", "torch")
        os.environ.setdefault("TRIGON_WEIGHTS", str(bundle / "adapter.pt"))
        for name, variable in (
            ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
            ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
        ):
            if (bundle / name).exists():
                os.environ.setdefault(variable, str(bundle / name))
    elif url:
        expected = os.environ.get("TRIGON_BUNDLE_SHA256")
        if not expected:
            raise SystemExit("trigon: TRIGON_BUNDLE_URL needs TRIGON_BUNDLE_SHA256 beside it")
        bundle = _fetch(url, expected)
        os.environ.setdefault("TRIGON_BACKEND", "torch")
        os.environ.setdefault("TRIGON_WEIGHTS", str(bundle / "adapter.pt"))
        for name, variable in (
            ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
            ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
        ):
            if (bundle / name).exists():
                os.environ.setdefault(variable, str(bundle / name))
    elif (MODEL / "adapter.pt").exists():
        # A bundle mounted at /model -- a model trained here, never published.
        # Its files are the checksum's job upstream; mounting is the operator
        # saying which model this is.
        os.environ.setdefault("TRIGON_BACKEND", "torch")
        os.environ.setdefault("TRIGON_WEIGHTS", str(MODEL / "adapter.pt"))
        for name, variable in (
            ("temperatures.json", "TRIGON_TEMPERATURE_PATH"),
            ("isotonic.json", "TRIGON_ISOTONIC_PATH"),
        ):
            if (MODEL / name).exists():
                os.environ.setdefault(variable, str(MODEL / name))
    command = sys.argv[1:] or ["trigon", "serve", "--host", "0.0.0.0", "--port", "8000"]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
