import numpy as np
import pytest

from facecloak.transforms import ROBUSTNESS


def test_robustness_registry():
    pytest.importorskip("cv2")
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    for transform in ROBUSTNESS.values():
        out = transform(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
