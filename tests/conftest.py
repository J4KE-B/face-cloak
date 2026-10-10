from __future__ import annotations

import cv2
import numpy as np
import pytest

from facecloak.paths import WEIGHTS_DIR


def need(*rel: str) -> None:
    missing = [r for r in rel if not (WEIGHTS_DIR / r).exists()]
    if missing:
        pytest.skip(f"weights missing under {WEIGHTS_DIR}: {missing}; run `python -m facecloak.weights --fetch`")


@pytest.fixture(scope="session")
def astronaut() -> np.ndarray:
    from skimage import data

    return np.ascontiguousarray(data.astronaut())  # RGB uint8 512x512, public domain


@pytest.fixture(scope="session")
def face_app():
    need("models/buffalo_l/det_10g.onnx")
    from facecloak.models import load_face_app

    return load_face_app()


@pytest.fixture(scope="session")
def astronaut_det(astronaut, face_app):
    from facecloak.models import detect_largest

    det = detect_largest(face_app, astronaut)
    assert det is not None, "no face detected in astronaut"
    return det


@pytest.fixture(scope="session")
def astronaut_warped(astronaut, astronaut_det):
    """Rotated + scaled astronaut and its transformed landmarks (non-trivial warp)."""
    A = cv2.getRotationMatrix2D((256, 256), 17.0, 0.8)
    img = cv2.warpAffine(astronaut, A, (512, 512))
    kps = (astronaut_det["kps"] @ A[:, :2].T + A[:, 2]).astype(np.float32)
    return img, kps
