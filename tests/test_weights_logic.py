"""Weight-free tests of facecloak.weights hardening (no network)."""

from __future__ import annotations

import hashlib
import zipfile
from typing import ClassVar

import pytest

from facecloak import weights as W


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)


def test_extract_skips_dot_members_and_flattens(tmp_path):
    z = tmp_path / "a.zip"
    _zip(z, {"pack/a.bin": b"A", "pack/..": b"X", "./.": b"Y", "__MACOSX/b": b"Z"})
    out = tmp_path / "out"
    W._extract(z, out)
    assert sorted(p.name for p in out.iterdir()) == ["a.bin"]


def test_verify_members_detects_tamper(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    (out / "a.bin").write_bytes(b"A")
    w = W.Weight(urls=(), path="x", extract_to="o", members={"a.bin": hashlib.sha256(b"A").hexdigest()})
    W._verify_members("t", w, out)
    (out / "a.bin").write_bytes(b"B")
    with pytest.raises(RuntimeError, match="mismatch"):
        W._verify_members("t", w, out)
    assert not (out / "a.bin").exists()  # tampered file is removed
    with pytest.raises(RuntimeError, match="missing"):
        W._verify_members("t", w, out)


def test_download_verifies_part_before_replace(tmp_path, monkeypatch):
    class Resp:
        headers: ClassVar[dict] = {}

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield b"payload"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(W.requests, "get", lambda *a, **k: Resp())
    dest = tmp_path / "f.bin"
    with pytest.raises(ValueError, match="sha256 mismatch"):
        W._download("http://x", dest, "0" * 64)
    assert not dest.exists() and not (tmp_path / "f.bin.part").exists()
    good = hashlib.sha256(b"payload").hexdigest()
    assert W._download("http://x", dest, good) == good and dest.read_bytes() == b"payload"
