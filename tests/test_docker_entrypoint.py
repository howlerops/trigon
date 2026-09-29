"""The image's entrypoint serves a release bundle only if its checksum matches.

`docker/entrypoint.py` is what stands between `TRIGON_BUNDLE_URL` and a
served model, so it is tested without Docker: a bundle it cannot verify is
refused before anything is unpacked, as `.github/workflows/release.yml`
refuses one.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import pathlib
import tarfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _entrypoint(monkeypatch, model_dir):
    monkeypatch.setenv("TRIGON_MODEL_DIR", str(model_dir))
    spec = importlib.util.spec_from_file_location("entrypoint", ROOT / "docker" / "entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path) -> tuple[pathlib.Path, str]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in (("adapter.pt", b"weights"), ("temperatures.json", b"{}")):
            info = tarfile.TarInfo(f"release/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path = tmp_path / "release.tar.gz"
    path.write_bytes(buffer.getvalue())
    return path, hashlib.sha256(buffer.getvalue()).hexdigest()


def test_a_bundle_with_the_right_checksum_is_unpacked(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    module = _entrypoint(monkeypatch, tmp_path / "model")
    path, digest = _bundle(tmp_path)
    unpacked = module._fetch(path.as_uri(), digest)
    assert (unpacked / "adapter.pt").read_bytes() == b"weights"
    assert not (tmp_path / "model" / "bundle.tar.gz").exists()


def test_a_bundle_with_the_wrong_checksum_is_refused_and_not_unpacked(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    module = _entrypoint(monkeypatch, tmp_path / "model")
    path, _ = _bundle(tmp_path)
    with pytest.raises(SystemExit, match="refusing to serve it"):
        module._fetch(path.as_uri(), "0" * 64)
    assert list((tmp_path / "model").iterdir()) == []


def test_a_bundle_url_without_a_checksum_is_refused(tmp_path, monkeypatch):
    module = _entrypoint(monkeypatch, tmp_path)
    monkeypatch.setenv("TRIGON_BUNDLE_URL", "file:///nowhere")
    monkeypatch.delenv("TRIGON_BUNDLE_SHA256", raising=False)
    with pytest.raises(SystemExit, match="needs TRIGON_BUNDLE_SHA256"):
        module.main()
