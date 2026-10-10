from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from facecloak.align import align_crop_cv2, align_crop_torch, estimate_norm, warp_torch


def _to_t(img):
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)


def _to_u8(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1) * 255).numpy()


def test_estimate_norm_matches_insightface():
    from insightface.utils import face_align

    rng = np.random.default_rng(0)
    for _ in range(5):
        kps = (face_align.arcface_dst * rng.uniform(1, 4) + rng.uniform(0, 200, 2)
               + rng.normal(0, 2, (5, 2))).astype(np.float32)
        np.testing.assert_allclose(estimate_norm(kps, 112), face_align.estimate_norm(kps, 112), atol=1e-4, rtol=1e-5)
        np.testing.assert_allclose(estimate_norm(kps, 224), face_align.estimate_norm(kps, 224), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("seed", range(6))
def test_warp_torch_matches_cv2(seed):
    rng = np.random.default_rng(seed)
    H, W = int(rng.integers(120, 200)), int(rng.integers(120, 200))
    # smooth-ish random image (pure noise would amplify any sub-pixel offset)
    img = cv2.GaussianBlur(rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (0, 0), 2.0)
    ang, s = np.deg2rad(rng.uniform(-20, 20)), rng.uniform(0.5, 2.0)
    M = np.array([[s * np.cos(ang), -s * np.sin(ang), rng.uniform(-30, 30)],
                  [s * np.sin(ang), s * np.cos(ang), rng.uniform(-30, 30)]])
    out = (112, 128)  # (H, W) non-square on purpose
    ref = cv2.warpAffine(img, M, (out[1], out[0]), borderValue=0.0).astype(np.float64)
    got = _to_u8(warp_torch(_to_t(img), M, out)).astype(np.float64)
    d = np.abs(ref - got)[2:-2, 2:-2]
    print(f"seed={seed} interior MAE={d.mean():.4f} max={d.max():.1f}")
    assert d.mean() < 1.0


def test_align_crop_parity_astronaut(astronaut, astronaut_det, astronaut_warped):
    for img, kps in [(astronaut, astronaut_det["kps"]), astronaut_warped]:
        ref = align_crop_cv2(img, kps).astype(np.float64)
        got = _to_u8(align_crop_torch(_to_t(img), kps)).astype(np.float64)
        mae = np.abs(ref - got).mean()
        print(f"align crop MAE={mae:.4f}")
        assert mae < 2.0


def test_crop_gradient_localised(astronaut, astronaut_det):
    kps = astronaut_det["kps"]
    x = _to_t(astronaut).requires_grad_(True)
    crop = align_crop_torch(x, kps)
    (crop * torch.randn_like(crop)).sum().backward()
    g = x.grad[0].abs().sum(0).numpy()
    assert g.max() > 0
    ys, xs = np.nonzero(g)
    # support must equal the source footprint of the 112x112 crop (inverse-mapped corners, +1px bilinear)
    M = np.vstack([estimate_norm(kps), [0, 0, 1]])
    corners = np.linalg.inv(M) @ np.array([[0, 0, 1], [112, 0, 1], [0, 112, 1], [112, 112, 1]]).T
    x0, x1 = corners[0].min() - 2, corners[0].max() + 2
    y0, y1 = corners[1].min() - 2, corners[1].max() + 2
    assert xs.min() >= x0 and xs.max() <= x1 and ys.min() >= y0 and ys.max() <= y1
    frac = (g > 0).mean()
    print(f"grad support fraction of image: {frac:.3f}")
    assert frac < 0.35
