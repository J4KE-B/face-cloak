import numpy as np
import pytest

from facecloak.metrics import bootstrap_ci, calibrate_threshold, cosine, protection_success_rate, psnr, ssim


def test_psnr():
    img_a = np.zeros((32, 32, 3), dtype=np.uint8)
    img_b = np.zeros((32, 32, 3), dtype=np.uint8)
    assert psnr(img_a, img_b) == float('inf')
    
    img_c = np.ones((32, 32, 3), dtype=np.uint8) * 10
    val = psnr(img_a, img_c)
    assert np.isfinite(val)
    assert val > 0

def test_ssim():
    pytest.importorskip("skimage")
    img_a = np.zeros((32, 32, 3), dtype=np.uint8)
    assert ssim(img_a, img_a) >= 0.99

def test_cosine():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    sims = cosine(a, b)
    assert sims.shape == (2,)
    np.testing.assert_allclose(sims, [1.0, 0.0], atol=1e-6)

def test_calibrate_threshold():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.7, 0.1, 20000)
    impostor = rng.normal(0.1, 0.1, 20000)
    
    res = calibrate_threshold(genuine, impostor, far=0.01)
    
    empirical_far = np.mean(impostor >= res["tau_far"])
    assert abs(empirical_far - 0.01) <= 0.005
    assert res["eer"] < 0.1

def test_protection_success_rate():
    sims = np.array([0.1, 0.5, 0.9, np.nan])
    rate, n = protection_success_rate(sims, tau=0.6)
    assert n == 3
    assert rate == 2.0 / 3.0

def test_bootstrap_ci():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]
    mean, lo, hi = bootstrap_ci(values, n_boot=2000, seed=0)
    assert 1.0 <= lo <= mean <= hi <= 5.0
    assert mean == 3.0

@pytest.mark.slow
def test_lpips():
    pytest.importorskip("lpips")
    pytest.importorskip("torch")
    from facecloak.metrics import LPIPSMetric
    metric = LPIPSMetric(device='cpu')
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.ones((32, 32, 3), dtype=np.uint8) * 255
    dist = metric(a, b)
    assert isinstance(dist, float)
    assert dist > 0
