"""SCRFD torch vs onnxruntime parity (needs weights; run manually, not while other jobs are heavy)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from facecloak.det_torch import INPUT_SIZE, STRIDES, SCRFDTorch, load_scrfd_torch, to_blob
from facecloak.paths import WEIGHTS_DIR
from tests.conftest import need

pytestmark = pytest.mark.slow
ONNX = WEIGHTS_DIR / "models" / "buffalo_l" / "det_10g.onnx"


def _insightface_blob(img_rgb: np.ndarray) -> np.ndarray:
    """Exactly retinaface.detect() preprocessing (BGR input, letterbox, blobFromImage)."""
    img = np.ascontiguousarray(img_rgb[:, :, ::-1])
    ratio = img.shape[0] / img.shape[1]
    if ratio > 1:
        nh, nw = INPUT_SIZE, int(INPUT_SIZE / ratio)
    else:
        nw, nh = INPUT_SIZE, int(INPUT_SIZE * ratio)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    return cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (INPUT_SIZE, INPUT_SIZE), (127.5,) * 3, swapRB=True)


@pytest.fixture(scope="module")
def det() -> SCRFDTorch:
    need("models/buffalo_l/det_10g.onnx")
    return load_scrfd_torch()


@pytest.fixture(scope="module")
def sess():
    import onnxruntime as ort

    need("models/buffalo_l/det_10g.onnx")
    return ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])


def test_network_scores_match_ort(det, sess, astronaut):
    blob = _insightface_blob(astronaut)
    ref = sess.run(None, {sess.get_inputs()[0].name: blob})
    with torch.no_grad():
        out = det.raw_outputs(torch.from_numpy(blob))
    for i in range(len(STRIDES)):
        r, t = ref[i].reshape(-1), out[i].numpy().reshape(-1)
        assert r.shape == t.shape
        assert np.abs(r - t).max() < 1e-4, (i, np.abs(r - t).max())


def test_preprocess_matches_cv2(astronaut):
    ref = _insightface_blob(astronaut)
    x = torch.from_numpy(astronaut).permute(2, 0, 1)[None].float() / 255
    got = to_blob(x).numpy()
    # cv2 rounds the resized image to uint8 (<=0.5 grey level = 0.004 in blob units)
    assert np.abs(ref - got).max() < 0.01


def test_gradient_flows(det, astronaut):
    from facecloak.det_torch import det_loss

    x = (torch.from_numpy(astronaut).permute(2, 0, 1)[None].float() / 255).requires_grad_(True)
    det_loss(det, x).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
