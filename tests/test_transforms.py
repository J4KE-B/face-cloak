import numpy as np
import pytest

from facecloak.transforms import ROBUSTNESS, EOTSampler, gaussian_blur_t, jpeg_ste, random_resize


def test_robustness_registry():
    pytest.importorskip("cv2")
    img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    for transform in ROBUSTNESS.values():
        out = transform(img)
        assert out.shape == img.shape
        assert out.dtype == img.dtype

def test_jpeg_ste():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    out = jpeg_ste(x, 50)
    assert out.shape == x.shape
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.sum() != 0

def test_gaussian_blur_t():
    torch = pytest.importorskip("torch")
    x = torch.rand(2, 3, 32, 32)
    out = gaussian_blur_t(x, 0.0)
    assert torch.allclose(x, out)
    
    out_blur = gaussian_blur_t(x, 1.0)
    assert out_blur.shape == x.shape

def test_random_resize():
    torch = pytest.importorskip("torch")
    x = torch.rand(2, 3, 32, 32)
    out = random_resize(x, lo=0.5, hi=1.0)
    assert out.shape == x.shape

def test_eot_sampler():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    sampler = EOTSampler(seed=42)
    x = torch.rand(2, 3, 32, 32)
    out = sampler(x)
    assert out.shape == x.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    
    # Check reproducibility
    sampler1 = EOTSampler(seed=0)
    out1 = sampler1(x)
    sampler2 = EOTSampler(seed=0)
    out2 = sampler2(x)
    assert torch.allclose(out1, out2)
