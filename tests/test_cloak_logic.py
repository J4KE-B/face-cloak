"""Weight-free logic tests for the Cloaker using a tiny frozen random conv 'recognizer'."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from facecloak.align import align_crop_torch
from facecloak.cloak import CloakConfig, Cloaker, noise_baseline, select_decoy

H = W = 96
KPS = np.array([[34, 40], [62, 40], [48, 54], [38, 68], [58, 68]], dtype=np.float32)
BBOX = np.array([24, 28, 72, 76], dtype=np.float32)


class DummyRec(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.c1 = nn.Conv2d(3, 8, 5, stride=2)
        self.c2 = nn.Conv2d(8, 16, 3, stride=2)
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3
        self.requires_grad_(False)

    def embed_image(self, img01, kps):
        x = align_crop_torch(img01, kps, 32) * 2 - 1
        x = F.relu(self.c1(x))
        x = self.c2(x).mean((2, 3))
        return F.normalize(x, dim=1)


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (H // 8, W // 8, 3)).astype(np.uint8)
    return np.kron(base, np.ones((8, 8, 1), np.uint8))


def make(**kw):
    kw.setdefault("models", {"dummy": 1.0})
    return Cloaker(CloakConfig(**kw), {"dummy": DummyRec()})


def test_linf_dtype_shape(img):
    r = make(eps=8, steps=10).cloak(img, KPS)
    assert r.image.dtype == np.uint8 and r.image.shape == img.shape
    assert r.delta_linf_255 <= 8
    assert np.abs(r.image.astype(int) - img.astype(int)).max() <= 8
    assert r.final_cos["dummy"] < 1.0


def test_loss_decreases(img):
    r = make(eps=16, steps=10, random_start=False).cloak(img, KPS)
    assert len(r.loss_history) == 10
    assert min(r.loss_history[3:]) < r.loss_history[0] - 1e-3


def test_roi_face_leaves_outside_unchanged(img):
    r = make(eps=16, steps=5, roi="face").cloak(img, KPS, bbox=BBOX)
    diff = np.abs(r.image.astype(int) - img.astype(int)).max(-1)
    assert diff[:12].sum() == 0 and diff[:, :8].sum() == 0 and diff[:, 87:].sum() == 0 and diff[91:].sum() == 0
    assert diff[30:70, 30:70].max() > 0
    with pytest.raises(ValueError):
        make(roi="face").cloak(img, KPS)


@pytest.mark.parametrize("bad", ["glintr100", "w600k_mbf"])
def test_leakage_guard(bad):
    with pytest.raises(ValueError):
        Cloaker(CloakConfig(models={bad: 1.0}), {bad: DummyRec()})


def test_seeded_determinism(img):
    a = make(steps=5, seed=3).cloak(img, KPS).image
    b = make(steps=5, seed=3).cloak(img, KPS).image
    c = make(steps=5, seed=4).cloak(img, KPS).image
    assert (a == b).all() and not (a == c).all()


def test_eot_runs(img):
    r = make(eps=8, steps=4, eot={"p_resize": 1, "p_blur": 1, "p_jpeg": 1}).cloak(img, KPS)
    assert r.delta_linf_255 <= 8


def test_targeted_needs_decoy_and_runs(img):
    rec = DummyRec()
    dec = torch.randn(16)
    cl = Cloaker(CloakConfig(models={"dummy": 1.0}, mode="both", eps=16, steps=10), {"dummy": rec})
    with pytest.raises(ValueError):
        cl.cloak(img, KPS)
    r = cl.cloak(img, KPS, decoy_embs={"dummy": dec})
    assert "dummy" in r.final_cos_decoy


def test_noise_baseline(img):
    n = noise_baseline(img, 8, seed=0)
    assert n.dtype == np.uint8 and np.abs(n.astype(int) - img.astype(int)).max() <= 8
    assert (noise_baseline(img, 8, 0) == n).all()


def test_select_decoy():
    clean = torch.tensor([1.0, 0.0])
    cands = torch.tensor([[1.0, 0.1], [-1.0, 0.0], [0.0, 1.0]])
    assert select_decoy(clean, cands) == 1


def test_config_roundtrip():
    c = CloakConfig(name="x", eps=4, models={"a": 0.5}, eot={"p_jpeg": 0.1})
    assert CloakConfig.from_dict(c.to_dict()) == c and c.alpha == 0.4


def test_eps_must_be_integer():
    CloakConfig(eps=8)
    CloakConfig(eps=8.0)
    with pytest.raises(ValueError, match="integer"):
        CloakConfig(eps=2.5)
    with pytest.raises(ValueError, match="integer"):
        CloakConfig.from_dict({"eps": 0.5})


def test_eot_best_iterate_scored_on_fixed_samples(img, monkeypatch):
    """Candidate scoring replays the SAME K seeded transform samples for every candidate."""
    import facecloak.cloak as cl

    states_score, orig = [], cl._eot_apply

    def spy(x, kps, p, rng):
        if not torch.is_grad_enabled():  # scoring passes run under no_grad; the PGD step does not
            states_score.append(repr(rng.bit_generator.state["state"]))
        return orig(x, kps, p, rng)

    monkeypatch.setattr(cl, "_eot_apply", spy)
    r = make(eps=8, steps=20, eot={"p_resize": 1, "p_blur": 1, "p_jpeg": 0}).cloak(img, KPS)
    # 1 initial + every 10 steps (2) = 3 candidates x K samples, with exactly K distinct sample streams
    assert len(states_score) == 3 * cl.EOT_SCORE_SAMPLES
    assert len(set(states_score)) == cl.EOT_SCORE_SAMPLES == 8
    assert r.delta_linf_255 <= 8
