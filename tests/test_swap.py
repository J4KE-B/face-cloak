"""Face-swap attacker smoke tests (slow; need inswapper / SimSwap weights). Clean swap must transfer identity."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from facecloak.paths import DATA_DIR
from tests.conftest import need

pytestmark = pytest.mark.slow

MIN_COS = 0.3


def _load_pair() -> tuple[np.ndarray, np.ndarray]:
    """(source, target) RGB uint8 from the smoke manifest (skips if data is absent)."""

    mpath = DATA_DIR / "manifest_smoke.json"
    if not mpath.exists():
        pytest.skip("data/manifest_smoke.json missing")
    m = json.loads(mpath.read_text())
    root = DATA_DIR / "lfw"  # manifest paths are relative to the LFW root and start with lfw/
    src_id, tgt_id = m["pairs"][0]
    src_rel = next(s["src_path"] for s in m["sources"] if s["id"] == src_id)
    tgt_rel = next(t["path"] for t in m["targets"] if t["id"] == tgt_id)
    imgs = []
    for rel in (src_rel, tgt_rel):
        bgr = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)
        if bgr is None:
            pytest.skip(f"image {rel} missing under {root}")
        imgs.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    return imgs[0], imgs[1]


@pytest.fixture(scope="module")
def pair() -> tuple[np.ndarray, np.ndarray]:
    return _load_pair()


@pytest.fixture(scope="module")
def glintr():
    need("models/antelopev2/glintr100.onnx")
    from facecloak.models import ORTRecognizer

    return ORTRecognizer("glintr100", "antelopev2")


def _cos_to_source(glintr, face_app, out: np.ndarray, src: np.ndarray) -> float:
    from facecloak.models import detect_largest

    d_out, d_src = detect_largest(face_app, out), detect_largest(face_app, src)
    assert d_out is not None, "no face in swapped output"
    assert d_src is not None
    return float(glintr.embed(out, d_out["kps"]) @ glintr.embed(src, d_src["kps"]))


def _check(sw, pair, glintr, face_app) -> None:
    src, tgt = pair
    r = sw.swap(src, tgt)
    assert r.source_detected and r.image is not None
    assert r.image.dtype == np.uint8 and r.image.shape == tgt.shape
    assert r.source_embedding is not None and abs(np.linalg.norm(r.source_embedding) - 1) < 1e-3
    assert _cos_to_source(glintr, face_app, r.image, src) >= MIN_COS


def test_inswapper_clean_swap(pair, glintr, face_app):
    need("inswapper_128.onnx", "models/buffalo_l/w600k_r50.onnx")
    from facecloak.swap import InswapperSwap

    sw = InswapperSwap()
    _check(sw, pair, glintr, face_app)
    # prepared target path gives the identical result
    prepared = sw.prepare_target(pair[1])
    assert np.array_equal(sw.swap(pair[0], prepared).image, sw.swap(pair[0], pair[1]).image)


def test_inswapper_source_without_face(pair):
    need("inswapper_128.onnx", "models/buffalo_l/w600k_r50.onnx")
    from facecloak.swap import InswapperSwap

    r = InswapperSwap().swap(np.full((250, 250, 3), 127, np.uint8), pair[1])
    assert not r.source_detected and r.image is None and r.source_embedding is None


def test_simswap_clean_swap(pair, glintr, face_app):
    need(
        "simswap/arcface_checkpoint.tar",
        "simswap/checkpoints/people/latest_net_G.pth",
        "simswap_repo/models/fs_networks.py",
    )
    from facecloak.swap import SimSwapSwap

    _check(SimSwapSwap(), pair, glintr, face_app)


def test_simswap_alignment_picks_frontal_for_frontal_kps():
    from facecloak.swap import _SIMSWAP_TEMPLATES_112, simswap_estimate_norm

    kps = _SIMSWAP_TEMPLATES_112[2] * 2.0 + 10.0  # frontal template, scaled + shifted
    M = simswap_estimate_norm(kps, 224)
    mapped = kps @ M[:, :2].T + M[:, 2]
    assert np.allclose(mapped, _SIMSWAP_TEMPLATES_112[2] * 2.0, atol=1e-3)
