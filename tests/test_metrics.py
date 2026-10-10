import numpy as np
import pytest

from facecloak.metrics import (
    bootstrap_ci,
    calibrate_far,
    calibrate_threshold,
    eer,
    kfold_accuracy,
    psnr,
    ssim,
    tar_at,
)


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

def test_calibrate_threshold():
    rng = np.random.default_rng(0)
    genuine = rng.normal(0.7, 0.1, 20000)
    impostor = rng.normal(0.1, 0.1, 20000)
    
    res = calibrate_threshold(genuine, impostor, far=0.01)
    
    empirical_far = np.mean(impostor >= res["tau_far"])
    assert abs(empirical_far - 0.01) <= 0.005
    assert res["eer"] < 0.1

def test_single_far_definition():
    rng = np.random.default_rng(1)
    g, i = rng.normal(0.7, 0.1, 5000), rng.normal(0.1, 0.1, 5000)
    for far in (0.01, 0.001):
        assert calibrate_threshold(g, i, far)["tau_far"] == calibrate_far(g, i, far)
        assert np.mean(i >= calibrate_far(g, i, far)) <= far
    r = calibrate_threshold(g, i, 0.01)
    e, t = eer(g, i)
    assert (r["eer"], r["tau_eer"]) == (e, t) and r["tar_at_far"] == tar_at(g, r["tau_far"])
    sc = np.concatenate([g, i])
    lb = np.concatenate([np.ones(len(g)), np.zeros(len(i))])
    folds = np.arange(len(sc)) % 5
    assert kfold_accuracy(sc, lb, folds)[0] > 0.95


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


def test_wilson_interval_brackets_extremes():
    from facecloak.metrics import wilson_interval

    for k, n in [(0, 5), (5, 5), (0, 1), (3, 7)]:
        lo, hi = wilson_interval(k, n)
        assert lo <= k / n <= hi
