from __future__ import annotations

import numpy as np
import pytest
import torch

from facecloak.align import align_crop_cv2
from facecloak.models import ORTRecognizer, load_arcface_torch, load_facenet_torch
from tests.conftest import need


@pytest.fixture(scope="module")
def arc():
    need("models/buffalo_l/w600k_r50.onnx")
    return load_arcface_torch()


@pytest.fixture(scope="module")
def ort_r50():
    need("models/buffalo_l/w600k_r50.onnx")
    return ORTRecognizer("w600k_r50", "buffalo_l")


def _raw_torch(arc, crops):
    x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255
    x = (x * 255 - arc.mean) / arc.std
    with torch.no_grad():
        return arc.raw_features(x).numpy()


def _cos(a, b):
    return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))


def test_preprocessing_matches_insightface(ort_r50):
    from insightface.model_zoo.arcface_onnx import ArcFaceONNX

    ref = ArcFaceONNX(model_file=str(ort_r50.session._model_path))
    assert (ref.input_mean, ref.input_std) == (ort_r50.mean, ort_r50.std) == (127.5, 127.5)


def test_crop_parity(arc, ort_r50, astronaut, astronaut_det):
    rng = np.random.default_rng(0)
    crops = [align_crop_cv2(astronaut, astronaut_det["kps"])]
    crops += [rng.integers(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(4)]
    ft, fo = _raw_torch(arc, crops), ort_r50.raw_features(crops)
    maxdiff = float(np.abs(ft - fo).max())
    coss = [_cos(a, b) for a, b in zip(ft, fo)]
    print(f"onnx2torch parity: min cos={min(coss):.8f} max|diff|={maxdiff:.2e} (feat scale {np.abs(fo).max():.2f})")
    assert min(coss) > 0.9999
    assert maxdiff < 1e-3


def test_end_to_end(arc, ort_r50, astronaut, astronaut_det, astronaut_warped):
    for img, kps in [(astronaut, astronaut_det["kps"]), astronaut_warped]:
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None]
        with torch.no_grad():
            et = arc.embed_image(x, kps)[0].numpy()
        eo = ort_r50.embed(img, kps)
        c = _cos(et, eo)
        print(f"end-to-end cos={c:.6f}")
        assert c > 0.99
        assert abs(np.linalg.norm(et) - 1) < 1e-5


def test_params_frozen(arc):
    assert not any(p.requires_grad for p in arc.parameters())
    assert not arc.training


def test_facenet():
    need("facenet/20180402-114759-vggface2.pt")
    from skimage import data

    net = load_facenet_torch()
    assert not any(p.requires_grad for p in net.parameters()) and not net.training
    img = np.ascontiguousarray(data.astronaut())
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None]
    crop = torch.nn.functional.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    flipped = crop.flip(-1)
    with torch.no_grad():
        a, b, f = net.embed_crop(crop), net.embed_crop(crop), net.embed_crop(flipped)
    assert a.shape == (1, 512)
    assert abs(a.norm().item() - 1) < 1e-5
    assert float((a * b).sum()) > 0.99999
    c = float((a * f).sum())
    print(f"facenet astronaut vs flipped cos={c:.4f}")
    assert c > 0.5
