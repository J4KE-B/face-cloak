"""LFW download, face-detection filtering, evaluation manifest and sharding helpers.

The manifest (``DATA_DIR/manifest.json``) lists *sources* (identities to cloak, with a
held-out reference image of the same person), *targets* (other identities used as swap
targets) and *pairs* ``[src_id, tgt_id]``. All paths are relative to the LFW image root so
the manifest is portable (e.g. to Kaggle).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import requests

from facecloak.paths import DATA_DIR, WEIGHTS_DIR

MANIFEST_VERSION = 1
LFW_VARIANT = "original"
DET_SCORE_MIN = 0.6
DET_SIZE = (640, 640)

# Same figshare mirror + checksums as sklearn.datasets._lfw
LFW_ARCHIVE_URL = "https://ndownloader.figshare.com/files/5976018"
LFW_ARCHIVE_SHA256 = "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"
PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"
PAIRS_SHA256 = "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"

DEFAULT_LFW_ROOT = DATA_DIR / "lfw"
DEFAULT_MANIFEST = DATA_DIR / "manifest.json"


# --------------------------------------------------------------------------- download
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    """GET ``url`` to ``dest`` (figshare rejects HEAD) and verify the checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: expected {sha256}, got {got}")
    tmp.replace(dest)


def images_dir(root: Path) -> Path:
    """Directory containing the identity folders (``root/lfw`` if present, else ``root``)."""
    root = Path(root)
    return root / "lfw" if (root / "lfw").is_dir() else root


def find_pairs_file(root: Path) -> Path | None:
    root = Path(root)
    for cand in (root / "pairs.txt", root.parent / "pairs.txt", root / "lfw" / "pairs.txt"):
        if cand.is_file():
            return cand
    found = sorted(root.glob("**/pairs.txt"))
    return found[0] if found else None


def ensure_lfw(root: Path | str | None = None) -> Path:
    """Return an LFW root, downloading + extracting the original archive when missing.

    Resolution order: explicit ``root`` arg, env ``FACECLOAK_LFW`` (existing extracted
    tree, never downloaded into), then ``DATA_DIR/lfw``.
    Layout afterwards: ``<root>/lfw/<Name>/<Name>_0001.jpg`` and ``<root>/pairs.txt``.
    """
    if root is None:
        root = os.environ.get("FACECLOAK_LFW") or DEFAULT_LFW_ROOT
    root = Path(root)
    img_dir = images_dir(root)
    has_images = img_dir.is_dir() and any(img_dir.iterdir())
    if not has_images:
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "lfw.tgz"
        if not archive.is_file() or _sha256(archive) != LFW_ARCHIVE_SHA256:
            print(f"downloading LFW -> {archive}")
            _download(LFW_ARCHIVE_URL, archive, LFW_ARCHIVE_SHA256)
        print("extracting LFW ...")
        with tarfile.open(archive) as tf:
            tf.extractall(root, filter="data")
    if find_pairs_file(root) is None:
        print("downloading pairs.txt")
        _download(PAIRS_URL, root / "pairs.txt", PAIRS_SHA256)
    return root


def list_identities(root: Path) -> dict[str, list[str]]:
    """identity -> sorted image paths relative to ``root`` (``lfw/<Name>/<file>.jpg``)."""
    root = Path(root)
    img_dir = images_dir(root)
    out: dict[str, list[str]] = {}
    for d in sorted(p for p in img_dir.iterdir() if p.is_dir()):
        files = sorted(f.name for f in d.glob("*.jpg"))
        if files:
            out[d.name] = [str((d / f).relative_to(root)) for f in files]
    return out


# --------------------------------------------------------------------------- detection
def make_detector(num_threads: int | None = None):
    """insightface buffalo_l detector (CPU). ``num_threads`` caps ORT intra-op threads."""
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    kwargs: dict[str, Any] = {}
    num_threads = num_threads or int(os.environ.get("FACECLOAK_THREADS", "8"))
    if num_threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        kwargs["sess_options"] = so
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(WEIGHTS_DIR),
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
        **kwargs,
    )
    app.prepare(ctx_id=-1, det_size=DET_SIZE)
    return app


def detect_center_face(app, img_bgr: np.ndarray, min_score: float = DET_SCORE_MIN) -> dict | None:
    """Face whose bbox centre is closest to the image centre; None if score < min_score."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    h, w = img_bgr.shape[:2]
    c = np.array([w / 2.0, h / 2.0])

    def dist(f) -> float:
        b = f.bbox
        return float(np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - c))

    best = min(faces, key=dist)
    if float(best.det_score) < min_score:
        return None
    return {
        "kps": np.asarray(best.kps, dtype=float).round(3).tolist(),
        "bbox": np.asarray(best.bbox, dtype=float)[:4].round(3).tolist(),
        "det_score": round(float(best.det_score), 4),
    }


def _detect_path(app, root: Path, rel: str) -> dict | None:
    import cv2

    img = cv2.imread(str(Path(root) / rel), cv2.IMREAD_COLOR)
    return None if img is None else detect_center_face(app, img)


# --------------------------------------------------------------------------- manifest
def build_manifest(
    n_ids: int = 150,
    n_targets: int = 30,
    targets_per_src: int = 3,
    seed: int = 0,
    lfw_root: Path | str | None = None,
    out: Path | str | None = None,
) -> dict:
    """Select sources/targets from LFW, run detection, write and return the manifest."""
    root = ensure_lfw(lfw_root)
    out = Path(out) if out else DEFAULT_MANIFEST
    if targets_per_src > n_targets:
        raise ValueError("targets_per_src must be <= n_targets")
    ids = list_identities(root)
    app = make_detector()
    rng = random.Random(seed)
    stats = {"src_candidates_checked": 0, "src_det_failures": 0,
             "tgt_candidates_checked": 0, "tgt_det_failures": 0}

    multi = sorted(k for k, v in ids.items() if len(v) >= 2)
    rng.shuffle(multi)
    sources: list[dict] = []
    for name in multi:
        if len(sources) >= n_ids:
            break
        stats["src_candidates_checked"] += 1
        a, b = ids[name][0], ids[name][1]
        da = _detect_path(app, root, a)
        db = _detect_path(app, root, b) if da else None
        if da is None or db is None:
            stats["src_det_failures"] += 1
            continue
        sources.append({
            "id": f"s{len(sources):04d}", "identity": name, "src_path": a, "ref_path": b,
            "src_kps": da["kps"], "src_bbox": da["bbox"], "ref_kps": db["kps"],
        })
    if len(sources) < n_ids:
        raise RuntimeError(f"only found {len(sources)} usable source identities (< {n_ids})")

    src_names = {s["identity"] for s in sources}
    others = sorted(k for k in ids if k not in src_names)
    rng.shuffle(others)
    targets: list[dict] = []
    for name in others:
        if len(targets) >= n_targets:
            break
        stats["tgt_candidates_checked"] += 1
        det, path = None, None
        for p in ids[name][:3]:  # first few images, in sorted order
            det = _detect_path(app, root, p)
            if det:
                path = p
                break
        if det is None:
            stats["tgt_det_failures"] += 1
            continue
        targets.append({"id": f"t{len(targets):04d}", "identity": name, "path": path,
                        "kps": det["kps"], "bbox": det["bbox"]})
    if len(targets) < n_targets:
        raise RuntimeError(f"only found {len(targets)} usable targets (< {n_targets})")

    pairs = [[s["id"], targets[(i * targets_per_src + j) % len(targets)]["id"]]
             for i, s in enumerate(sources) for j in range(targets_per_src)]
    manifest = {
        "version": MANIFEST_VERSION, "seed": seed, "lfw_variant": LFW_VARIANT,
        "n_ids": n_ids, "n_targets": n_targets, "targets_per_src": targets_per_src,
        "det_score_min": DET_SCORE_MIN, "stats": stats,
        "sources": sources, "targets": targets, "pairs": pairs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(path: Path | str | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_MANIFEST).read_text())


def manifest_identities(manifest: dict) -> set[str]:
    """All source + target identity names (to exclude from calibration pairs)."""
    return {s["identity"] for s in manifest["sources"]} | {t["identity"] for t in manifest["targets"]}


def shard(items: list, spec: str = "0/1") -> list:
    """Deterministic shard ``"i/N"``: ``sorted(items, by id)[i::N]``.

    Items may be dicts with an ``id`` key, or plain sortable values.
    """
    i_s, n_s = spec.split("/")
    i, n = int(i_s), int(n_s)
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"bad shard spec {spec!r}")
    key = (lambda x: x["id"]) if items and isinstance(items[0], dict) else (lambda x: x)
    return sorted(items, key=key)[i::n]
