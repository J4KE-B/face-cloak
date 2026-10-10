"""Differentiable face alignment (torch) plus cv2 reference implementations.

All images in this module are RGB. Torch images are float (N,3,H,W) in [0,1]; numpy images are
uint8 (H,W,3). ``M`` is always the cv2-style 2x3 *forward* matrix (source pixel -> crop pixel).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Same 5-point template as insightface.utils.face_align.arcface_dst (112x112 crop):
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def template_for(size: int) -> np.ndarray:
    """ArcFace template scaled to a ``size`` x ``size`` crop (float32, (5,2)).

    ArcFace uses size=112 (identical to insightface for multiples of 112). FaceNet (160 px) has no
    official 5-point template; we scale the ArcFace one by 160/112 so the face occupies the same
    fraction of the crop. This is a deliberate approximation (FaceNet/MTCNN crops are looser).
    """
    return (ARCFACE_DST.astype(np.float64) * (size / 112.0)).astype(np.float32)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (3x3) mapping src -> dst (Umeyama 1991).

    Numerically equivalent to skimage ``SimilarityTransform.estimate`` (which insightface uses)
    without relying on its deprecated API.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    var_s = (sc**2).sum() / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / var_s
    T = np.eye(3)
    T[:2, :2] = scale * R
    T[:2, 2] = mu_d - scale * R @ mu_s
    return T


def estimate_norm(kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """2x3 forward similarity matrix mapping detected 5-pt ``kps`` onto the scaled ArcFace template."""
    kps = np.asarray(kps)
    assert kps.shape == (5, 2), kps.shape
    return _umeyama(kps, template_for(image_size))[:2, :]


def align_crop_cv2(img_uint8_rgb: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Reference crop == insightface ``norm_crop`` (cv2.warpAffine, bilinear, zero border). RGB in/out."""
    M = estimate_norm(kps, size)
    return cv2.warpAffine(img_uint8_rgb, M, (size, size), borderValue=0.0)


def warp_torch(img: torch.Tensor, M: np.ndarray | torch.Tensor, out_size: int | tuple[int, int]) -> torch.Tensor:
    """Differentiable ``cv2.warpAffine(img, M, out_size)`` (INTER_LINEAR, zero border).

    Derivation (all in 3x3 homogeneous form, x = column index, y = row index):
      * cv2.warpAffine inverts M (forward: src px -> dst px) and, for every integer dst pixel
        index q, samples the source at p = M^-1 q, where integer coordinates are pixel *centres*.
      * grid_sample(align_corners=False) uses normalized coords u = 2*(p + 0.5)/W - 1, i.e. pixel
        centre i sits at u = (2i+1)/W - 1, and u = -1 / +1 are the outer image edges.
      * Let Nin  (src px -> src norm): u = (2/Ws) p + (1/Ws - 1)          (same for y with Hs)
        Let Nout (dst px -> dst norm): v = (2/Wd) q + (1/Wd - 1)
      * affine_grid(theta) maps output normalized coords v to input normalized coords u:
            u = theta @ [v, 1],  theta = Nin @ M^-1 @ Nout^-1   (top two rows).
      * With align_corners=False affine_grid generates v at dst pixel centres
        (2j+1)/Wd - 1, matching q = j in cv2.
    Out-of-image samples are 0 (padding_mode='zeros'); note cv2 INTER_LINEAR with borderValue=0 also
    blends towards 0 at the edge, so the two agree to within rounding.
    """
    if isinstance(out_size, int):
        out_size = (out_size, out_size)
    Hd, Wd = out_size
    N, _, Hs, Ws = img.shape
    M3 = np.eye(3, dtype=np.float64)
    M3[:2, :] = M.detach().cpu().double().numpy() if isinstance(M, torch.Tensor) else np.asarray(M, np.float64)
    Minv = np.linalg.inv(M3)
    n_in = np.array([[2.0 / Ws, 0, 1.0 / Ws - 1], [0, 2.0 / Hs, 1.0 / Hs - 1], [0, 0, 1]])
    n_out = np.array([[2.0 / Wd, 0, 1.0 / Wd - 1], [0, 2.0 / Hd, 1.0 / Hd - 1], [0, 0, 1]])
    theta = (n_in @ Minv @ np.linalg.inv(n_out))[:2]
    theta_t = torch.from_numpy(theta).to(device=img.device, dtype=img.dtype).unsqueeze(0).expand(N, 2, 3)
    grid = F.affine_grid(theta_t, (N, img.shape[1], Hd, Wd), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def align_crop_torch(img: torch.Tensor, kps: np.ndarray, size: int = 112) -> torch.Tensor:
    """Differentiable aligned crop (N,3,size,size) of ``img`` (N,3,H,W) in [0,1]. kps: (5,2) in pixels.

    The same landmarks/matrix are used for every image in the batch; ``kps`` carries no gradient.
    """
    return warp_torch(img, estimate_norm(kps, size), size)
