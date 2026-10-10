"""Calibrate cosine-similarity verification thresholds of evaluator recognizers on LFW.

Protocol: LFW ``pairs.txt`` (10 folds x 300 genuine + 300 impostor), detection with
buffalo_l (face closest to image centre, score >= 0.6), 5-point alignment to 112x112,
cosine similarity of L2-normalised embeddings. Pairs involving manifest source/target
identities are excluded so thresholds are calibrated on people never used in the attack
evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from facecloak.data import (
    DEFAULT_MANIFEST,
    detect_center_face,
    ensure_lfw,
    find_pairs_file,
    images_dir,
    load_manifest,
    make_detector,
    manifest_identities,
)
from facecloak.metrics import calibrate_far, eer, kfold_accuracy, tar_at
from facecloak.models import ort_providers
from facecloak.paths import CACHE_DIR, RESULTS_DIR, WEIGHTS_DIR

MODELS: dict[str, Path] = {
    "glintr100": WEIGHTS_DIR / "models/antelopev2/glintr100.onnx",
    "w600k_mbf": WEIGHTS_DIR / "models/buffalo_s/w600k_mbf.onnx",
    "w600k_r50": WEIGHTS_DIR / "models/buffalo_l/w600k_r50.onnx",  # whitebox model
}

# ------------------------------------------------------------------ pairs


def parse_pairs(pairs_file: Path) -> list[dict]:
    lines = pairs_file.read_text().splitlines()
    n_folds, n_per = (int(x) for x in lines[0].split())
    out = []
    for k, line in enumerate(lines[1:]):
        p = line.split()
        fold = k // (2 * n_per)
        if len(p) == 3:
            out.append({"fold": fold, "label": 1, "a": (p[0], int(p[1])), "b": (p[0], int(p[2]))})
        elif len(p) == 4:
            out.append({"fold": fold, "label": 0, "a": (p[0], int(p[1])), "b": (p[2], int(p[3]))})
    assert n_folds == 10
    return out


def rel(img: tuple[str, int], prefix: str = "lfw") -> str:
    """Image path relative to the LFW root; ``prefix`` is ``images_dir(root)`` relative to root
    (``"lfw"`` for the standard layout, ``""`` when the identity folders sit directly in the root)."""
    p = f"{img[0]}/{img[0]}_{img[1]:04d}.jpg"
    return f"{prefix}/{p}" if prefix else p


# ------------------------------------------------------------------ workers
THREADS = int(os.environ.get("FACECLOAK_THREADS", "8"))
CHECKPOINT_EVERY = 500


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    np.savez(path, paths=np.array(list(cache)), embs=np.stack(list(cache.values())))


def load_recognizer(path: Path):
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo

    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    m = model_zoo.get_model(str(path), providers=ort_providers(), sess_options=so)
    m.prepare(ctx_id=-1)
    # prepare() calls session.set_providers(), which silently drops sess_options (=> all cores);
    # rebuild the session so the thread cap really applies.
    m.session = ort.InferenceSession(str(path), so, providers=ort_providers())
    return m


def embed(rec, root: Path, r: str, kps: list) -> np.ndarray:
    import cv2
    from insightface.utils import face_align

    img = cv2.imread(str(root / r), cv2.IMREAD_COLOR)  # BGR, as insightface expects
    crop = face_align.norm_crop(img, np.asarray(kps, dtype=np.float32), 112)
    f = np.asarray(rec.get_feat(crop)).reshape(-1).astype(np.float32)
    return f / np.linalg.norm(f)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--max-pairs", type=int, default=None, help="quick run: ~N pairs total, spread over folds")
    ap.add_argument("--lfw-root", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--no-exclude-manifest", action="store_true")
    ap.add_argument(
        "--out", type=Path, default=None,
        help="default: results/calibration.json, or results/calibration_quick.json when --max-pairs is set "
        "(a quick subset must never overwrite the full calibration)",
    )
    ap.add_argument("--threads", type=int, default=None, help="ORT intra-op threads (default: $FACECLOAK_THREADS or 8)")
    a = ap.parse_args()
    global THREADS
    if a.threads:
        THREADS = max(1, a.threads)
    if a.out is None:
        a.out = RESULTS_DIR / ("calibration_quick.json" if a.max_pairs else "calibration.json")
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    root = ensure_lfw(a.lfw_root)
    pairs = parse_pairs(find_pairs_file(root))
    n_excluded = 0
    if not a.no_exclude_manifest and a.manifest.is_file():
        excl = manifest_identities(load_manifest(a.manifest))
        keep = [p for p in pairs if p["a"][0] not in excl and p["b"][0] not in excl]
        n_excluded = len(pairs) - len(keep)
        pairs = keep
        print(f"excluded {n_excluded} pairs involving manifest identities ({len(excl)} ids)")
    if a.max_pairs:  # keep the first max_pairs per (fold,label) cell, ~balanced quick subset
        cnt: dict = {}
        sub = []
        for p in pairs:
            k = (p["fold"], p["label"])
            cnt[k] = cnt.get(k, 0) + 1
            if cnt[k] <= max(1, a.max_pairs // 20):
                sub.append(p)
        pairs = sub
    print(f"{len(pairs)} pairs, root={root}")
    assert images_dir(root).is_dir()
    pre = images_dir(root).relative_to(root).as_posix()
    pre = "" if pre == "." else pre

    imgs = sorted({rel(p[s], pre) for p in pairs for s in ("a", "b")})

    # ---- detection (shared across models, cached)
    det_cache_f = CACHE_DIR / "lfw_det_cache.json"
    det_cache: dict = json.loads(det_cache_f.read_text()) if det_cache_f.is_file() else {}
    todo = [r for r in imgs if r not in det_cache]
    if todo:
        import cv2

        det = make_detector(THREADS)
        for n, r in enumerate(tqdm(todo, desc="detect"), 1):
            img = cv2.imread(str(root / r), cv2.IMREAD_COLOR)
            det_cache[r] = None if img is None else detect_center_face(det, img)
            if n % CHECKPOINT_EVERY == 0:
                det_cache_f.write_text(json.dumps(det_cache))
        det_cache_f.write_text(json.dumps(det_cache))
    ok = {r for r in imgs if det_cache.get(r)}
    print(f"detected {len(ok)}/{len(imgs)} images ({time.time() - t0:.0f}s)")

    valid = [p for p in pairs if rel(p["a"], pre) in ok and rel(p["b"], pre) in ok]
    n_skipped = len(pairs) - len(valid)
    need = sorted({rel(p[s], pre) for p in valid for s in ("a", "b")})
    labels = np.array([p["label"] for p in valid])
    folds = np.array([p["fold"] for p in valid])

    results: dict = {}
    if a.out.is_file():
        results = json.loads(a.out.read_text())
    for name in a.models:
        tm = time.time()
        cache_f = CACHE_DIR / f"emb_cache_{name}.npz"
        cache: dict[str, np.ndarray] = {}
        if cache_f.is_file():
            z = np.load(cache_f, allow_pickle=False)
            cache = dict(zip(z["paths"].tolist(), z["embs"], strict=True))
        todo = [r for r in need if r not in cache]

        if todo:
            rec = load_recognizer(MODELS[name])
            for n, r in enumerate(tqdm(todo, desc=name), 1):
                cache[r] = embed(rec, root, r, det_cache[r]["kps"])
                if n % CHECKPOINT_EVERY == 0:
                    save_cache(cache_f, cache)
            save_cache(cache_f, cache)
        sims = np.array([float(cache[rel(p["a"], pre)] @ cache[rel(p["b"], pre)]) for p in valid])
        g, i = sims[labels == 1], sims[labels == 0]
        tau1, tau01 = calibrate_far(g, i, 0.01), calibrate_far(g, i, 0.001)
        e, tau_e = eer(g, i)
        m, s = kfold_accuracy(sims, labels, folds)
        results[name] = {
            "tau_far1": tau1, "tau_far01": tau01, "eer": e, "tau_eer": tau_e,
            "tar_far1": tar_at(g, tau1), "tar_far01": tar_at(g, tau01),
            "acc_10fold_mean": m, "acc_std": s,
            "n_genuine": len(g), "n_impostor": len(i), "n_skipped": n_skipped,
            "n_excluded_manifest": n_excluded,
            "whitebox": name == "w600k_r50",
        }
        print(name, json.dumps(results[name]), f"({time.time() - tm:.0f}s)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
