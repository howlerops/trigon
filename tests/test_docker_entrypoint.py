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


def test_a_mounted_bundle_is_served_without_a_url(tmp_path, monkeypatch):
    """docker-compose mounts a locally trained bundle at /model."""
    (tmp_path / "adapter.pt").write_bytes(b"weights")
    (tmp_path / "temperatures.json").write_text("{}")
    module = _entrypoint(monkeypatch, tmp_path)
    # main() sets variables with os.environ.setdefault, which monkeypatch does
    # not track: on the real environment they outlived this test and pointed
    # every later app at a deleted adapter. A copy is thrown away afterwards.
    environ = {
        k: v
        for k, v in module.os.environ.items()
        if k
        not in (
            "TRIGON_BUNDLE_URL",
            "TRIGON_BACKEND",
            "TRIGON_WEIGHTS",
            "TRIGON_TEMPERATURE_PATH",
            "TRIGON_ISOTONIC_PATH",
        )
    }
    monkeypatch.setattr(module.os, "environ", environ)
    ran = []
    monkeypatch.setattr(module.sys, "argv", ["entrypoint"])
    monkeypatch.setattr(module.os, "execvp", lambda cmd, args: ran.append(args))
    module.main()
    assert module.os.environ["TRIGON_WEIGHTS"] == str(tmp_path / "adapter.pt")
    assert module.os.environ["TRIGON_TEMPERATURE_PATH"] == str(tmp_path / "temperatures.json")
    assert "TRIGON_ISOTONIC_PATH" not in module.os.environ
    assert ran and ran[0][:2] == ["trigon", "serve"]


def _published(tmp_path, tamper: str | None = None) -> tuple[str, str]:
    """A bundle laid out as publish_bundle.py publishes it, served from file:// URLs."""
    folder = tmp_path / "published"
    folder.mkdir()
    files = {"adapter.pt": b"weights", "temperatures.json": b"{}", "README.md": b"# card"}
    for name, data in files.items():
        (folder / name).write_bytes(data)
    sums = "".join(f"{hashlib.sha256(d).hexdigest()}  {n}\n" for n, d in sorted(files.items()))
    (folder / "SHA256SUMS").write_text(sums)
    if tamper:
        (folder / tamper).write_bytes(b"something else")
    return folder.as_uri() + "/", hashlib.sha256(sums.encode()).hexdigest()


def test_a_published_bundle_is_fetched_file_by_file_and_checked(tmp_path, monkeypatch):
    base, digest = _published(tmp_path)
    entry = _entrypoint(monkeypatch, tmp_path / "model")
    bundle = entry._fetch_published(base, digest)
    assert (bundle / "adapter.pt").read_bytes() == b"weights"
    assert (bundle / "README.md").exists()


def test_a_tampered_file_or_list_is_refused(tmp_path, monkeypatch):
    base, digest = _published(tmp_path, tamper="adapter.pt")
    entry = _entrypoint(monkeypatch, tmp_path / "model")
    with pytest.raises(SystemExit, match="adapter.pt does not match"):
        entry._fetch_published(base, digest)
    with pytest.raises(SystemExit, match="SHA256SUMS does not match"):
        entry._fetch_published(base, "0" * 64)
