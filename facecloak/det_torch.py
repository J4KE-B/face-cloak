"""Differentiable SCRFD (buffalo_l det_10g) face detector for the detector-suppression loss.

Preprocessing replicates insightface/model_zoo/retinaface.py (``detect`` + ``forward``):
aspect-preserving resize into a 640x640 canvas (top-left, zero padded in *pixel* space), then
``(x - 127.5) / 128`` with swapRB (our tensors are already RGB, so no swap is needed).
The det_10g ONNX has 9 outputs: [0:3] sigmoid scores for strides 8/16/32, [3:6] bbox, [6:9] kps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from facecloak.paths import WEIGHTS_DIR

INPUT_SIZE = 640
MEAN = 127.5
STD = 128.0
STRIDES = (8, 16, 32)


def letterbox01(img01: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    """(N,3,H,W) RGB [0,1] -> (N,3,size,size) [0,1], resized (bilinear) top-left, zero padded.

    Mirrors retinaface.detect: new size uses int() truncation; cv2.resize INTER_LINEAR equals
    F.interpolate(bilinear, align_corners=False) without antialiasing (half-pixel centres).
    """
    H, W = img01.shape[-2:]
    im_ratio = H / W
    if im_ratio > 1.0:  # model_ratio == 1
        new_h, new_w = size, int(size / im_ratio)
    else:
        new_w, new_h = size, int(size * im_ratio)
    x = F.interpolate(img01, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return F.pad(x, (0, size - new_w, 0, size - new_h), value=0.0)


def to_blob(img01: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    """Letterbox + normalise to the network input blob."""
    return (letterbox01(img01, size) * 255.0 - MEAN) / STD


class SCRFDTorch(nn.Module):
    """Frozen torch SCRFD. ``forward(img01)`` returns the three stride score tensors."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net.eval().requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def raw_outputs(self, blob: torch.Tensor) -> list[torch.Tensor]:
        out = self.net(blob)
        return list(out) if isinstance(out, (tuple, list)) else [out]

    def forward(self, img01: torch.Tensor) -> list[torch.Tensor]:
        return self.raw_outputs(to_blob(img01))[: len(STRIDES)]


def load_scrfd_torch(path: str | None = None) -> SCRFDTorch:
    from onnx2torch import convert

    from facecloak.models import _configure_torch

    _configure_torch()
    p = path or str(WEIGHTS_DIR / "models" / "buffalo_l" / "det_10g.onnx")
    return SCRFDTorch(convert(p))


def det_loss(det: SCRFDTorch, img01: torch.Tensor, k: int = 50, thresh: float | None = None) -> torch.Tensor:
    """Detection confidence to *minimise*: mean of top-k scores over all strides (default), or, if
    ``thresh`` is given, the sum of scores above it."""
    s = torch.cat([o.reshape(-1) for o in det(img01)])
    if thresh is not None:
        return s[s > thresh].sum()
    return s.topk(min(k, s.numel())).values.mean()
