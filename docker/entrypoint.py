"""Fetch and verify a release bundle if one is named, then run the command.

TRIGON_BUNDLE_URL names a release bundle (`releases/<name>/` in the
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


def main() -> None:
    url = os.environ.get("TRIGON_BUNDLE_URL")
    if url:
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
