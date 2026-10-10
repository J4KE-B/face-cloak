"""Face detector and recognizers: differentiable torch attack models, ONNX evaluation models.

Public APIs take/return RGB uint8 HxWx3 numpy images (insightface itself wants BGR; we convert
internally) or RGB float tensors in [0,1].

ATTACK_MODELS may be used inside the PGD loop. EVAL_MODELS (glintr100, w600k_mbf, and the ONNX
w600k_r50 used for held-out parity checks) are for *evaluation only*: glintr100 and w600k_mbf must
NEVER be used to compute attack gradients/losses, otherwise the black-box transfer results are void.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from facecloak.align import align_crop_cv2, align_crop_torch
from facecloak.paths import WEIGHTS_DIR


def num_threads() -> int:
    """Thread budget: env FACECLOAK_THREADS (default 8, capped at 8)."""
    return max(1, min(8, int(os.environ.get("FACECLOAK_THREADS", "8"))))


def ort_providers() -> list[str]:
    """onnxruntime providers: CUDA first when env FACECLOAK_ORT_GPU=1 and it is available, else CPU.

    Off by default so laptop runs stay on CPU; the Kaggle notebooks set it so every result in a run
    is computed with the same provider.
    """
    if os.environ.get("FACECLOAK_ORT_GPU") == "1":
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _configure_torch() -> None:
    torch.set_num_threads(num_threads())

# ----------------------------------------------------------------------------- detection


def load_face_app(det_size: tuple[int, int] = (640, 640), providers: list[str] | None = None):
    """insightface FaceAnalysis with only the SCRFD detector (buffalo_l det_10g, gives 5-pt kps)."""
    from insightface.app import FaceAnalysis

    _configure_torch()
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(WEIGHTS_DIR),
        allowed_modules=["detection"],
        providers=providers or ort_providers(),
    )
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


def detect_largest(app, img_rgb_uint8: np.ndarray) -> dict | None:
    """Largest-area face as dict(bbox (4,), kps (5,2) float32, det_score) or None. Input RGB uint8."""
    bgr = np.ascontiguousarray(img_rgb_uint8[:, :, ::-1])
    faces = app.get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return {
        "bbox": np.asarray(f.bbox, dtype=np.float32),
        "kps": np.asarray(f.kps, dtype=np.float32),
        "det_score": float(f.det_score),
    }


# ----------------------------------------------------------------------------- torch recognizers


class TorchRecognizer(nn.Module):
    """Frozen differentiable recognizer.

    Input crops are RGB floats in [0,1]; ``(x*255 - mean) / std`` is applied internally.
    ArcFace: mean=std=127.5, FaceNet: mean=127.5, std=128 (fixed_image_standardization).
    """

    def __init__(self, backbone: nn.Module, input_size: int, mean: float, std: float, channel_order: str = "rgb"):
        super().__init__()
        assert channel_order in ("rgb", "bgr")
        self.backbone = backbone.eval().requires_grad_(False)
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.channel_order = channel_order
        self.eval()

    def train(self, mode: bool = True):  # always stay in eval mode (BN, dropout)
        return super().train(False)

    def embed_crop(self, crop01: torch.Tensor) -> torch.Tensor:
        """(N,3,s,s) RGB [0,1] -> L2-normalized (N,512) embedding."""
        x = crop01[:, [2, 1, 0]] if self.channel_order == "bgr" else crop01
        x = (x * 255.0 - self.mean) / self.std
        return F.normalize(self.raw_features(x), dim=1)

    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone output on already-standardized input (unnormalized for ArcFace)."""
        return self.backbone(x)

    def embed_image(self, img01: torch.Tensor, kps: np.ndarray) -> torch.Tensor:
        """Differentiable align (full image -> crop) then embed. img01: (N,3,H,W) in [0,1]."""
        return self.embed_crop(align_crop_torch(img01, kps, self.input_size))


def _onnx_path(name: str, pack: str) -> Path:
    p = WEIGHTS_DIR / "models" / pack / f"{name}.onnx"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `uv run python -m facecloak.weights --fetch {pack}`")
    return p


def onnx_input_norm(path: Path) -> tuple[float, float]:
    """(mean, std) exactly as insightface ArcFaceONNX infers them (model_zoo/arcface_onnx.py:25-41)."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    find_sub = find_mul = False
    for node in graph.node[:8]:
        if node.name.startswith("Sub") or node.name.startswith("_minus"):
            find_sub = True
        if node.name.startswith("Mul") or node.name.startswith("_mul"):
            find_mul = True
    return (0.0, 1.0) if (find_sub and find_mul) else (127.5, 127.5)


def load_arcface_torch(name: str = "w600k_r50", pack: str = "buffalo_l") -> TorchRecognizer:
    """onnx2torch conversion of an insightface ArcFace ONNX (attack model, differentiable).

    Preprocessing (verified in insightface/model_zoo/arcface_onnx.py get_feat, l.82-83): the aligned
    BGR crop goes through cv2.dnn.blobFromImages(scale=1/std, mean=(mean,)*3, swapRB=True) so the
    network consumes RGB, (x - 127.5)/127.5. Output is the raw (unnormalized) 512-d feature.
    """
    from onnx2torch import convert

    _configure_torch()

    path = _onnx_path(name, pack)
    mean, std = onnx_input_norm(path)
    net = convert(str(path))
    return TorchRecognizer(net, 112, mean, std, "rgb")


def load_facenet_torch() -> TorchRecognizer:
    """FaceNet InceptionResnetV1 (VGGFace2), RGB 160x160, fixed_image_standardization (x-127.5)/128."""
    from facecloak.third_party.facenet import InceptionResnetV1

    _configure_torch()
    net = InceptionResnetV1(pretrained="vggface2", classify=False)
    return TorchRecognizer(net, 160, 127.5, 128.0, "rgb")


# ----------------------------------------------------------------------------- ONNX evaluators


class ORTRecognizer:
    """onnxruntime recognizer reproducing insightface's ArcFaceONNX preprocessing. Evaluation only."""

    def __init__(self, name: str, pack: str, providers: list[str] | None = None):
        import onnxruntime as ort

        self.name = name
        path = _onnx_path(name, pack)
        self.mean, self.std = onnx_input_norm(path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = num_threads()
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), so, providers=providers or ort_providers())
        cfg = self.session.get_inputs()[0]
        self.input_name = cfg.name
        self.input_size = int(cfg.shape[2])

    def raw_features(self, crops_rgb_uint8: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Unnormalized features for aligned RGB uint8 crops (N,s,s,3). Mirrors ArcFaceONNX.get_feat."""
        imgs = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops_rgb_uint8]  # to BGR as insightface has
        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / self.std, (self.input_size, self.input_size), (self.mean,) * 3, swapRB=True
        )
        return self.session.run(None, {self.input_name: blob})[0].astype(np.float32)

    def embed_crops(self, batch: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """L2-normalized (N,512) float32 embeddings of aligned RGB uint8 crops."""
        f = self.raw_features(batch)
        return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)

    def embed(self, img_rgb_uint8: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """L2-normalized (512,) float32 embedding of the face given by 5-pt ``kps``."""
        return self.embed_crops([align_crop_cv2(img_rgb_uint8, kps, self.input_size)])[0]


_EVAL_SPECS = {"glintr100": "antelopev2", "w600k_mbf": "buffalo_s", "w600k_r50": "buffalo_l"}


def load_eval_recognizers() -> dict[str, ORTRecognizer]:
    """Held-out evaluators: glintr100 (antelopev2), w600k_mbf (buffalo_s), w600k_r50 (ONNX)."""
    return {n: ORTRecognizer(n, p) for n, p in _EVAL_SPECS.items()}


#: Models allowed inside the attack loop (differentiable torch).
ATTACK_MODELS: dict[str, Callable[[], TorchRecognizer]] = {
    "w600k_r50": load_arcface_torch,
    "facenet_vggface2": load_facenet_torch,
}
#: Evaluation-only models. NEVER use these to compute attack gradients/losses.
EVAL_MODELS: dict[str, Callable[[], ORTRecognizer]] = {
    n: (lambda n=n, p=p: ORTRecognizer(n, p)) for n, p in _EVAL_SPECS.items()
}
