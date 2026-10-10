"""PGD adversarial cloaking against face recognizers (identity-feature attack on face-swap).

Only differentiable *attack* recognizers may be used here. ``glintr100`` / ``w600k_mbf`` are
held-out evaluators; the Cloaker refuses them to prevent evaluator leakage.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

EVAL_ONLY_NAMES = frozenset({"glintr100", "w600k_mbf"})

DEFAULT_EOT: dict[str, Any] = {
    "p_resize": 0.5,
    "resize_range": (0.5, 1.0),
    "p_blur": 0.5,
    "sigma_range": (0.5, 1.5),
    "p_jpeg": 0.5,
    "quality_range": (50, 95),
    "kps_jitter_px": 2.0,
    "rot_jitter_deg": 3.0,
}


@dataclass
class CloakConfig:
    name: str = "cloak"
    eps: float = 8.0  # L-inf budget, /255 units
    steps: int = 100
    alpha: float | None = None  # step size, /255 units; default eps/10
    random_start: bool = True
    models: dict[str, float] = field(default_factory=lambda: {"w600k_r50": 1.0})
    mode: str = "untargeted"  # 'untargeted' | 'targeted' | 'both'
    targeted_weight: float = 1.0
    lpips_weight: float = 0.0
    eot: dict[str, Any] | None = None
    det_weight: float = 0.0
    roi: str = "full"  # 'full' | 'face'
    seed: int = 0

    def __post_init__(self) -> None:
        # The delivered image is uint8, so the achievable L-inf budget is integral: a fractional eps
        # would silently be rounded by quantisation and the "linf <= eps" claim would be loose.
        if float(self.eps) != float(round(self.eps)) or self.eps < 0:
            raise ValueError(f"eps must be a non-negative integer (uint8 /255 units), got {self.eps!r}")
        if self.alpha is None:
            self.alpha = self.eps / 10.0
        if self.mode not in ("untargeted", "targeted", "both"):
            raise ValueError(f"bad mode {self.mode!r}")
        if self.roi not in ("full", "face"):
            raise ValueError(f"bad roi {self.roi!r}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CloakConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown CloakConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CloakResult:
    image: np.ndarray  # uint8 HxWx3
    delta_linf_255: int
    loss_history: list[float]
    final_cos: dict[str, float]  # cos(E(cloaked), E(clean)) per model, quantized output, clean kps
    steps_run: int
    seconds: float
    final_cos_decoy: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers


def noise_baseline(img_uint8: np.ndarray, eps: float, seed: int = 0) -> np.ndarray:
    """Random sign noise of magnitude round(eps) (same L-inf budget), clipped to uint8."""
    rng = np.random.default_rng(seed)
    e = round(eps)
    sign = rng.choice(np.array([-1, 1], dtype=np.int16), size=img_uint8.shape)
    return np.clip(img_uint8.astype(np.int16) + sign * e, 0, 255).astype(np.uint8)


def select_decoy(clean_emb: Any, candidate_embs: Any) -> int:
    """Index of the candidate least similar (min cosine) to ``clean_emb`` (Fawkes-style)."""
    c = F.normalize(torch.as_tensor(clean_emb, dtype=torch.float32).reshape(1, -1), dim=1)
    m = F.normalize(torch.as_tensor(candidate_embs, dtype=torch.float32).reshape(-1, c.shape[1]), dim=1)
    return int(torch.argmin((m @ c.T).squeeze(1)).item())


def _to01(img_uint8: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1)[None].float().div(255).to(device)


@torch.no_grad()
def embed_uint8(rec: Any, img_uint8: np.ndarray, kps: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return rec.embed_image(_to01(img_uint8, device), kps)[0]


def decoy_embeddings(
    recognizers: dict[str, Any], decoy_img_uint8: np.ndarray, decoy_kps: np.ndarray, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Embedding of the decoy face under every attack recognizer."""
    return {k: embed_uint8(r, decoy_img_uint8, decoy_kps, device) for k, r in recognizers.items()}


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    xp = F.pad(x, (r, r, r, r), mode="reflect")
    xp = F.conv2d(xp, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(xp, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _jpeg_ste(x: torch.Tensor, quality: int) -> torch.Tensor:
    """Real JPEG round-trip in the forward pass, identity gradient (straight-through)."""
    arr = (x.detach()[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    xj = torch.from_numpy(np.ascontiguousarray(dec)).permute(2, 0, 1)[None].to(x.device, x.dtype) / 255
    return x + (xj - x).detach()


def _eot_apply(x: torch.Tensor, kps: np.ndarray, p: dict[str, Any], rng: np.random.Generator):
    """Sample a differentiable image transform and a landmark perturbation. Returns (x', kps')."""
    H, W = x.shape[-2:]
    if rng.random() < p["p_resize"]:
        s = float(rng.uniform(*p["resize_range"]))
        h, w = max(8, round(H * s)), max(8, round(W * s))
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=s < 1)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
    if rng.random() < p["p_blur"]:
        x = _gauss_blur(x, float(rng.uniform(*p["sigma_range"])))
    if rng.random() < p["p_jpeg"]:
        x = _jpeg_ste(x, int(rng.integers(p["quality_range"][0], p["quality_range"][1] + 1)))
    k = np.asarray(kps, dtype=np.float64)
    c = k.mean(0)
    th = math.radians(float(rng.uniform(-p["rot_jitter_deg"], p["rot_jitter_deg"])))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    k = (k - c) @ R.T + c + rng.normal(0.0, p["kps_jitter_px"], size=k.shape)
    return x, k.astype(np.float32)


def _face_mask(shape: tuple[int, int], bbox: np.ndarray, pad: float = 0.3) -> torch.Tensor:
    H, W = shape
    x1, y1, x2, y2 = (float(v) for v in bbox)
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    xa, xb = max(0, math.floor(x1 - px)), min(W, math.ceil(x2 + px))
    ya, yb = max(0, math.floor(y1 - py)), min(H, math.ceil(y2 + py))
    m = torch.zeros(1, 1, H, W)
    m[..., ya:yb, xa:xb] = 1.0
    return m


# ----------------------------------------------------------------------------- Cloaker


#: Number of fixed EOT transform samples used to score candidate iterates (see :meth:`Cloaker.cloak`).
EOT_SCORE_SAMPLES = 8


class Cloaker:
    """Projected-gradient (L-inf, sign-step) identity-feature attack.

    EOT is *stochastic with one transform sample per PGD step* (not the average over several samples).
    Because the step gradients are noisy, the returned iterate is chosen by a separate, deterministic
    score: every 10 steps the candidate is scored by its mean loss over ``EOT_SCORE_SAMPLES`` (8)
    transform samples drawn from a fixed seeded RNG stream, the SAME 8 samples for every candidate, so
    candidates are compared on equal footing and on the transformed images the attack is meant to
    survive (without EOT the clean loss is used).
    """

    def __init__(
        self,
        cfg: CloakConfig,
        recognizers: dict[str, Any],
        lpips_model: Any = None,
        detector_torch: Any = None,
        device: str = "cpu",
    ):
        bad = (set(recognizers) | set(cfg.models)) & EVAL_ONLY_NAMES
        if bad:
            raise ValueError(f"evaluation-only models cannot be used in the attack: {sorted(bad)}")
        missing = set(cfg.models) - set(recognizers)
        if missing:
            raise ValueError(f"config models without a recognizer: {sorted(missing)}")
        if cfg.lpips_weight > 0 and lpips_model is None:
            raise ValueError("lpips_weight > 0 requires lpips_model")
        if cfg.det_weight > 0 and detector_torch is None:
            raise ValueError("det_weight > 0 requires detector_torch")
        self.cfg = cfg
        self.recs = {k: recognizers[k] for k in cfg.models}
        self.lpips = lpips_model
        self.det = detector_torch
        self.device = device
        self.eot = {**DEFAULT_EOT, **cfg.eot} if cfg.eot is not None else None

    def _loss(self, x, x0, kps, clean, decoy, eot_rng=None) -> torch.Tensor:
        cfg = self.cfg
        xt, k = (x, kps) if eot_rng is None else _eot_apply(x, kps, self.eot, eot_rng)
        total = x.new_zeros(())
        for name, rec in self.recs.items():
            w = cfg.models[name]
            e = rec.embed_image(xt, k)[0]
            if cfg.mode in ("untargeted", "both"):
                total = total + w * (e * clean[name]).sum()
            if cfg.mode in ("targeted", "both"):
                tw = cfg.targeted_weight if cfg.mode == "both" else 1.0
                total = total - w * tw * (e * decoy[name]).sum()
        if cfg.lpips_weight > 0:
            total = total + cfg.lpips_weight * self.lpips(x * 2 - 1, x0 * 2 - 1).mean()
        if cfg.det_weight > 0:
            from facecloak.det_torch import det_loss

            total = total + cfg.det_weight * det_loss(self.det, x)
        return total

    def cloak(
        self,
        img_uint8: np.ndarray,
        kps: np.ndarray,
        bbox: np.ndarray | None = None,
        decoy_embs: dict[str, torch.Tensor] | None = None,
    ) -> CloakResult:
        cfg, dev = self.cfg, self.device
        t0 = time.time()
        kps = np.asarray(kps, dtype=np.float32)
        if cfg.mode in ("targeted", "both") and (not decoy_embs or any(k not in decoy_embs for k in self.recs)):
            raise ValueError("targeted/both mode needs decoy_embs for every attack model")
        if cfg.roi == "face" and bbox is None:
            raise ValueError("roi='face' needs bbox")
        decoy = {k: F.normalize(v.detach().to(dev).reshape(-1), dim=0) for k, v in (decoy_embs or {}).items()}

        x0 = _to01(img_uint8, dev)
        H, W = x0.shape[-2:]
        mask = _face_mask((H, W), bbox).to(dev) if cfg.roi == "face" else torch.ones(1, 1, H, W, device=dev)
        with torch.no_grad():
            clean = {k: r.embed_image(x0, kps)[0] for k, r in self.recs.items()}

        eps, alpha = cfg.eps / 255.0, cfg.alpha / 255.0
        gen = torch.Generator().manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        if cfg.random_start:
            delta = ((torch.rand(x0.shape, generator=gen) * 2 - 1) * eps).to(dev)
        else:
            delta = torch.zeros_like(x0)
        delta = (delta * mask)
        delta = (x0 + delta).clamp(0, 1) - x0

        def score(d: torch.Tensor) -> float:
            """Lower is better. EOT: mean loss over K fixed seeded transform samples; else clean loss."""
            x = (x0 + d).clamp(0, 1)
            with torch.no_grad():
                if self.eot is None:
                    return float(self._loss(x, x0, kps, clean, decoy))
                vals = [
                    float(self._loss(x, x0, kps, clean, decoy, np.random.default_rng([cfg.seed, 7919, k])))
                    for k in range(EOT_SCORE_SAMPLES)
                ]
            return float(np.mean(vals))

        best_delta, best_val = delta.clone(), score(delta)
        history: list[float] = []
        for step in range(cfg.steps):
            delta = delta.detach().requires_grad_(True)
            x = (x0 + delta).clamp(0, 1)
            loss = self._loss(x, x0, kps, clean, decoy, rng if self.eot is not None else None)
            (grad,) = torch.autograd.grad(loss, delta)
            history.append(float(loss.detach()))
            with torch.no_grad():
                delta = delta - alpha * grad.sign()
                delta = delta.clamp(-eps, eps) * mask
                delta = (x0 + delta).clamp(0, 1) - x0
            if (step + 1) % 10 == 0 or step + 1 == cfg.steps:
                v = score(delta)
                if v < best_val:
                    best_val, best_delta = v, delta.detach().clone()

        out = (x0 + best_delta).mul(255).round().clamp(0, 255)[0].permute(1, 2, 0).byte().cpu().numpy()
        linf = int(np.abs(out.astype(np.int16) - img_uint8.astype(np.int16)).max())
        assert linf <= int(cfg.eps), f"L-inf {linf} exceeds eps {cfg.eps}"
        if cfg.roi == "face":
            assert not (np.abs(out.astype(np.int16) - img_uint8) * (1 - mask[0, 0].cpu().numpy())[..., None]).any()

        xq = _to01(out, dev)
        with torch.no_grad():
            final = {k: float((r.embed_image(xq, kps)[0] * clean[k]).sum()) for k, r in self.recs.items()}
            fdec = {k: float((r.embed_image(xq, kps)[0] * decoy[k]).sum()) for k, r in self.recs.items()} if decoy else {}
        return CloakResult(out, linf, history, final, cfg.steps, time.time() - t0, fdec)
