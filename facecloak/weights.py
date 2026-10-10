"""Model-weight manifest, downloader and checker.

Usage::

    python -m facecloak.weights --check
    python -m facecloak.weights --fetch                 # every non-kaggle_only entry
    python -m facecloak.weights --fetch inswapper_128 buffalo_l
    python -m facecloak.weights --fetch simswap_checkpoints --force-kaggle-only

SECURITY NOTE: every PyTorch checkpoint (``.pt``/``.pth``/``.tar``) must be loaded with
``torch.load(path, map_location=..., weights_only=True)`` so a tampered file cannot execute
pickle code. SHA-256 checks here are the first line of defence; ``weights_only=True`` the second.

Layout under WEIGHTS_DIR: ``inswapper_128.onnx``, ``models/<pack>/*.onnx`` (so that
``FaceAnalysis(name=<pack>, root=WEIGHTS_DIR)`` works), ``facenet/*.pt``, ``simswap/*``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

from facecloak.paths import WEIGHTS_DIR


@dataclass(frozen=True)
class Weight:
    """One downloadable artefact.

    Attributes:
        urls: Candidate URLs tried in order.
        path: Download destination relative to WEIGHTS_DIR.
        sha256: Expected digest of the downloaded file (None = not yet recorded).
        extract_to: If set, the file is a zip extracted into this dir (relative to WEIGHTS_DIR).
        members: ``{filename: sha256}`` of files expected after extraction (basename match).
        kaggle_only: Skipped by default fetch (large / only needed on the GPU box).
    """

    urls: tuple[str, ...]
    path: str
    sha256: str | None = None
    extract_to: str | None = None
    members: dict[str, str | None] = field(default_factory=dict)
    kaggle_only: bool = False


_BUF = "https://github.com/deepinsight/insightface/releases/download/v0.7"

MANIFEST: dict[str, Weight] = {
    "inswapper_128": Weight(
        urls=(
            "https://huggingface.co/fofr/comfyui/resolve/main/insightface/inswapper_128.onnx",
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        ),
        path="inswapper_128.onnx",
        sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    ),
    "buffalo_l": Weight(
        urls=(f"{_BUF}/buffalo_l.zip",),
        path="downloads/buffalo_l.zip",
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        extract_to="models/buffalo_l",
        members={"w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43", "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"},
    ),
    "buffalo_s": Weight(
        urls=(f"{_BUF}/buffalo_s.zip",),
        path="downloads/buffalo_s.zip",
        sha256="d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e",
        extract_to="models/buffalo_s",
        members={"w600k_mbf.onnx": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"},
    ),
    "antelopev2": Weight(
        urls=(
            "https://huggingface.co/MonsterMMORPG/tools/resolve/main/antelopev2.zip",
            "https://huggingface.co/Charles-Elena/antelopev2/resolve/main/antelopev2.zip",
            "https://huggingface.co/combatmaestro/antelopeV2/resolve/main/antelopev2.zip",
        ),
        path="downloads/antelopev2.zip",
        sha256="8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",
        extract_to="models/antelopev2",
        members={"glintr100.onnx": "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf", "scrfd_10g_bnkps.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"},
    ),
    "facenet_vggface2": Weight(
        urls=(
            "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt",
        ),
        path="facenet/20180402-114759-vggface2.pt",
        sha256="281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b",
    ),
    "simswap_checkpoints": Weight(
        urls=("https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip",),
        path="simswap/checkpoints.zip",
        sha256="0593544d9401d6bf28247932466b54e98a79139e328e3ab3ec1c541c5efc6ee8",
        extract_to="simswap/checkpoints/people",
        members={"latest_net_G.pth": "24caf144e9aabd5acd1127b06f13ed3528240adb9e747d77a94ac3f33e672330"},
        kaggle_only=True,
    ),
    "simswap_arcface": Weight(
        urls=("https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar",),
        path="simswap/arcface_checkpoint.tar",
        sha256="52ea5ce4902017b77a2bb811dd9a82f57dee3883e06c18a42d288437263c7a20",
        kaggle_only=True,
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=(15, 60), allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, file=sys.stderr
        ) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)  # atomic


def _extract(zip_path: Path, out_dir: Path) -> None:
    """Extract flat into out_dir, collapsing any nested folder (e.g. antelopev2/antelopev2/)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not name or info.filename.startswith("__MACOSX"):
                continue
            tmp = out_dir / (name + ".part")
            with z.open(info) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, out_dir / name)


def fetch(name: str) -> None:
    w = MANIFEST[name]
    dest = WEIGHTS_DIR / w.path
    if dest.exists() and (w.sha256 is None or sha256_file(dest) == w.sha256):
        print(f"[{name}] already present")
    else:
        errors: list[str] = []
        for url in w.urls:
            try:
                print(f"[{name}] downloading {url}")
                _download(url, dest)
                digest = sha256_file(dest)
                if w.sha256 and digest != w.sha256:
                    dest.unlink()
                    raise ValueError(f"sha256 mismatch: got {digest}")
                print(f"[{name}] sha256 {digest}")
                break
            except Exception as e:  # noqa: BLE001 - try next mirror
                errors.append(f"{url}: {e}")
                print(f"[{name}] failed: {e}", file=sys.stderr)
        else:
            raise RuntimeError(f"all mirrors failed for {name}:\n" + "\n".join(errors))
    if w.extract_to:
        _extract(dest, WEIGHTS_DIR / w.extract_to)


def check(names: list[str]) -> bool:
    ok = True
    for name in names:
        w = MANIFEST[name]
        items = [(name, WEIGHTS_DIR / w.path, w.sha256)]
        if w.extract_to:
            items += [
                (f"{name}/{m}", WEIGHTS_DIR / w.extract_to / m, s) for m, s in w.members.items()
            ]
        for label, p, expected in items:
            if not p.exists():
                status = "MISSING" + (" (kaggle_only)" if w.kaggle_only else "")
                ok &= w.kaggle_only
            else:
                got = sha256_file(p)
                if expected is None:
                    status = f"UNPINNED {got}"
                    ok = False
                else:
                    status = "OK" if got == expected else f"MISMATCH {got}"
                    ok &= got == expected
            print(f"{label:40s} {p}  {status}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", nargs="*", metavar="NAME", help="fetch names (default: all local)")
    g.add_argument("--check", nargs="*", metavar="NAME", help="verify names (default: all)")
    ap.add_argument("--force-kaggle-only", action="store_true", help="allow fetching kaggle_only")
    a = ap.parse_args(argv)
    if a.fetch is not None:
        names = a.fetch or [n for n, w in MANIFEST.items() if not w.kaggle_only]
        for n in names:
            if MANIFEST[n].kaggle_only and not a.force_kaggle_only:
                print(f"[{n}] kaggle_only, skipping (use --force-kaggle-only)")
                continue
            fetch(n)
        return 0
    return 0 if check(a.check or list(MANIFEST)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
