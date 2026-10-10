"""Real-weights cloak tests (slow; need w600k_r50 / facenet weights)."""

from __future__ import annotations

import numpy as np
import pytest

from facecloak.cloak import CloakConfig, Cloaker, decoy_embeddings, embed_uint8
from tests.conftest import need

pytestmark = pytest.mark.slow


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return float(10 * np.log10(255**2 / mse))


@pytest.fixture(scope="module")
def r50():
    need("models/buffalo_l/w600k_r50.onnx")
    from facecloak.models import load_arcface_torch

    return load_arcface_torch()


@pytest.fixture(scope="module")
def facenet():
    from facecloak.models import load_facenet_torch

    try:
        return load_facenet_torch()
    except Exception as e:  # noqa: BLE001 # weights download unavailable
        pytest.skip(f"facenet unavailable: {e}")


def test_untargeted_r50(r50, astronaut, astronaut_det):
    cfg = CloakConfig(eps=8, steps=20)
    r = Cloaker(cfg, {"w600k_r50": r50}).cloak(astronaut, astronaut_det["kps"])
    assert r.final_cos["w600k_r50"] < 0.5
    assert psnr(r.image, astronaut) > 30


def test_ensemble(r50, facenet, astronaut, astronaut_det):
    cfg = CloakConfig(eps=8, steps=3, models={"w600k_r50": 1.0, "facenet_vggface2": 1.0})
    r = Cloaker(cfg, {"w600k_r50": r50, "facenet_vggface2": facenet}).cloak(astronaut, astronaut_det["kps"])
    assert set(r.final_cos) == {"w600k_r50", "facenet_vggface2"} and r.delta_linf_255 <= 8


def test_targeted_r50(r50, astronaut, astronaut_det):
    from skimage import data

    decoy_img = np.ascontiguousarray(data.astronaut()[:, ::-1])  # placeholder decoy: mirrored image
    kps = astronaut_det["kps"]
    dk = kps.copy()
    dk[:, 0] = astronaut.shape[1] - 1 - dk[:, 0]
    dk = dk[[1, 0, 2, 4, 3]]
    d = decoy_embeddings({"w600k_r50": r50}, decoy_img, dk)
    before_decoy = float((embed_uint8(r50, astronaut, kps) * d["w600k_r50"]).sum())
    cfg = CloakConfig(eps=8, steps=20, mode="targeted")
    r = Cloaker(cfg, {"w600k_r50": r50}).cloak(astronaut, kps, decoy_embs=d)
    assert r.final_cos["w600k_r50"] < 1.0
    assert r.final_cos_decoy["w600k_r50"] > before_decoy
