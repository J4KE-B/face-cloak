"""Face-swap attackers used to evaluate cloaked identity photos.

Threat model: we cloak the SOURCE identity photo; the attacker feeds the cloaked photo to a face-swap
tool as the identity source and pastes that identity onto a target image. Both attackers here are
black-box with respect to the cloak (no gradients flow through them) and re-run face detection and
recognition on whatever image they receive, as a real attacker would.

* :class:`InswapperSwap`  - insightface ``inswapper_128`` (ONNX) driven by buffalo_l detection and
  the buffalo_l ArcFace embedding. Runs on CPU or GPU.
* :class:`SimSwapSwap`    - official SimSwap 224 "people" model (PyTorch), which uses its OWN ArcFace
  network for the identity latent, so it is a genuine transfer attacker.

All public APIs take RGB uint8 HxWx3 numpy images and return RGB uint8 (insightface/cv2 want BGR,
conversion happens internally).

SECURITY (SimSwap): ``arcface_checkpoint.tar`` is a fully pickled ``nn.Module`` (not a state_dict),
so ``torch.load(weights_only=True)`` cannot read it and plain ``weights_only=False`` would allow
arbitrary code execution from a tampered file. We therefore (1) refuse to load unless its SHA-256
matches the pin in :mod:`facecloak.weights`, and (2) unpickle through an allow-list unpickler
(:class:`_RestrictedUnpickler`) that only resolves the exact classes found in that file
(``models.arcface_models.{ResNet,IRBlock,SEBlock}``, ``torch.nn`` layers, tensor rebuild helpers).
The generator checkpoint is a plain state_dict, loaded with ``weights_only=True`` (also SHA-pinned).
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from facecloak.paths import WEIGHTS_DIR

#: Pinned commit of https://github.com/neuralchen/SimSwap (HEAD of main, 2024-08-07).
SIMSWAP_COMMIT = "bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780"
SIMSWAP_REPO = WEIGHTS_DIR / "simswap_repo"
SIMSWAP_GEN = WEIGHTS_DIR / "simswap" / "checkpoints" / "people" / "latest_net_G.pth"
SIMSWAP_ARC = WEIGHTS_DIR / "simswap" / "arcface_checkpoint.tar"
INSWAPPER_PATH = WEIGHTS_DIR / "inswapper_128.onnx"

_FALLBACK_DET_SIZE = (320, 320)


class NoFaceError(ValueError):
    """Raised when the *target* image contains no detectable face (nothing to swap onto)."""


@dataclass
class SwapResult:
    """Outcome of one swap.

    Attributes:
        image: Swapped target, RGB uint8 (same shape as the target), or None if the source face was
            not detected by the attacker.
        source_detected: Whether the attacker found a face in the (possibly cloaked) source.
        source_det_score: Detector confidence of that face (0.0 if none).
        source_embedding: The attacker's identity embedding of the source, L2-normalised float32
            (buffalo_l ArcFace for inswapper, SimSwap's own ArcFace latent for SimSwap); None if
            the face was not detected.
    """

    image: np.ndarray | None
    source_detected: bool
    source_det_score: float
    source_embedding: np.ndarray | None


# ----------------------------------------------------------------------------- shared helpers


def _bgr(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.dtype != np.uint8 or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB uint8 HxWx3, got {img_rgb.dtype} {img_rgb.shape}")
    return np.ascontiguousarray(img_rgb[:, :, ::-1])


def _default_providers() -> list[str]:
    import onnxruntime as ort

    avail = ort.get_available_providers()
    return (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else []) + ["CPUExecutionProvider"]


def _capped_session(path: Path | str, providers: list[str]):
    """ORT session with intra-op threads capped by ``facecloak.models.num_threads`` (<= 8)."""
    import onnxruntime as ort

    from facecloak.models import num_threads

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = num_threads()
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=providers)


def _faces(app, bgr: np.ndarray) -> list:
    """Detect faces at the app's det_size; if none, retry at 320x320.

    LFW images are 250x250. SCRFD at 640x640 upsamples them 2.56x, which can push the face outside the
    anchor scale range and fail on some (esp. cloaked) images; 320x320 matches their native scale.
    The attacker would equally retry with a different det_size, so we model that.
    """
    faces = app.get(bgr)
    if faces:
        return faces
    det = app.det_model
    old = det.input_size
    try:
        det.input_size = _FALLBACK_DET_SIZE
        return app.get(bgr)
    finally:
        det.input_size = old


def _largest(faces: list):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _image_key(img: np.ndarray) -> str:
    return hashlib.blake2b(img.tobytes(), digest_size=16, salt=str(img.shape).encode()[:16]).hexdigest()


# ----------------------------------------------------------------------------- inswapper


@dataclass
class PreparedTarget:
    """Target image (BGR) with its pre-detected face, reusable across many sources."""

    bgr: np.ndarray
    face: Any


class InswapperSwap:
    """insightface ``inswapper_128`` attacker (detect + embed with buffalo_l, then swap).

    Args:
        providers: ORT providers; default = CUDA if available, then CPU.
        app: Optional pre-built ``FaceAnalysis`` (must include the ``recognition`` module, i.e. NOT the
            detection-only app of ``facecloak.models.load_face_app``) to share memory with other code.
    """

    def __init__(self, providers: list[str] | None = None, app: Any = None) -> None:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo.inswapper import INSwapper

        self.providers = providers or _default_providers()
        gpu = "CUDAExecutionProvider" in self.providers
        if app is None:
            app = FaceAnalysis(
                name="buffalo_l",
                root=str(WEIGHTS_DIR),
                allowed_modules=["detection", "recognition"],
                providers=self.providers,
            )
            # FaceAnalysis builds default (uncapped) sessions; replace them by thread-capped ones.
            for m in app.models.values():
                m.session = _capped_session(m.model_file, self.providers)
            app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
        if "recognition" not in app.models:
            raise ValueError("shared FaceAnalysis must include the 'recognition' module")
        self.app = app
        if not INSWAPPER_PATH.exists():
            raise FileNotFoundError(f"{INSWAPPER_PATH} missing; run `python -m facecloak.weights --fetch inswapper_128`")
        # model_zoo.get_model() would build an uncapped session; INSwapper accepts our own.
        self.swapper = INSwapper(model_file=str(INSWAPPER_PATH), session=_capped_session(INSWAPPER_PATH, self.providers))
        self._cache: OrderedDict[str, PreparedTarget] = OrderedDict()

    def prepare_target(self, target_rgb: np.ndarray) -> PreparedTarget:
        """Detect the largest face in the target once and cache it (LRU of 8)."""
        key = _image_key(target_rgb)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        bgr = _bgr(target_rgb)
        faces = _faces(self.app, bgr)
        if not faces:
            raise NoFaceError("no face detected in target image")
        prepared = PreparedTarget(bgr=bgr, face=_largest(faces))
        self._cache[key] = prepared
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return prepared

    def swap(self, source_rgb: np.ndarray, target: np.ndarray | PreparedTarget) -> SwapResult:
        """Paste the identity of ``source_rgb`` onto the largest face of ``target``."""
        prepared = target if isinstance(target, PreparedTarget) else self.prepare_target(target)
        faces = _faces(self.app, _bgr(source_rgb))
        if not faces:
            return SwapResult(None, False, 0.0, None)
        src = _largest(faces)
        out_bgr = self.swapper.get(prepared.bgr, prepared.face, src, paste_back=True)
        return SwapResult(
            image=np.ascontiguousarray(out_bgr[:, :, ::-1]),
            source_detected=True,
            source_det_score=float(src.det_score),
            source_embedding=np.asarray(src.normed_embedding, dtype=np.float32),
        )


# ----------------------------------------------------------------------------- SimSwap

# SimSwap alignment templates for crop_mode 'None' (224 "people" model): 5 pose templates defined on a
# 112 px grid (insightface_func/utils/face_align_ffhqandnewarc.py, src_map); the best-fitting one (min
# summed landmark error after a similarity fit) is used. Order: left eye, right eye, nose, mouth L, R.
_SIMSWAP_TEMPLATES_112 = np.array(
    [
        [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]],  # left profile
        [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]],  # left
        [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]],  # frontal
        [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]],  # right
        [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]],  # right profile
    ],
    dtype=np.float32,
)
_SIMSWAP_CROP = 224


def simswap_estimate_norm(kps: np.ndarray, size: int = _SIMSWAP_CROP) -> np.ndarray:
    """2x3 forward matrix mapping ``kps`` (5,2) onto the best SimSwap 'None'-mode template.

    Port of ``estimate_norm(lmk, image_size, mode='None')`` with skimage's SimilarityTransform replaced by
    the equivalent least-squares Umeyama fit already used in :mod:`facecloak.align` (skimage is not a
    runtime dependency, and its ``.estimate`` is deprecated).
    """
    from facecloak.align import _umeyama

    kps = np.asarray(kps, dtype=np.float64)
    ones = np.ones((5, 1))
    best, best_err = None, np.inf
    for tmpl in _SIMSWAP_TEMPLATES_112.astype(np.float64) * (size / 112.0):
        M = _umeyama(kps, tmpl)[:2, :]
        err = np.sqrt(((np.hstack([kps, ones]) @ M.T - tmpl) ** 2).sum(1)).sum()
        if err < best_err:
            best, best_err = M, err
    assert best is not None
    return best


#: (module, name) pairs the allow-list unpickler may resolve while reading arcface_checkpoint.tar.
#: Derived by statically disassembling the file (pickletools, no execution).
_ALLOWED_GLOBALS = {
    ("models.arcface_models", "ResNet"),
    ("models.arcface_models", "IRBlock"),
    ("models.arcface_models", "SEBlock"),
    ("collections", "OrderedDict"),
    ("__builtin__", "set"),
    ("builtins", "set"),
    ("torch", "FloatStorage"),
    ("torch", "LongStorage"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
}
_ALLOWED_NN_PREFIX = "torch.nn.modules."


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler resolving only globals on the allow-list (blocks ``os.system`` & friends)."""

    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED_GLOBALS or module.startswith(_ALLOWED_NN_PREFIX):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} in checkpoint")


def _restricted_pickle_module() -> types.ModuleType:
    """Duck-typed ``pickle`` module for ``torch.load(pickle_module=...)``."""
    m = types.ModuleType("facecloak_restricted_pickle")
    m.Unpickler = _RestrictedUnpickler  # type: ignore[attr-defined]
    m.UnpicklingError = pickle.UnpicklingError  # type: ignore[attr-defined]
    m.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()  # type: ignore[attr-defined]
    return m


def _require_sha(path: Path, expected: str | None, what: str) -> None:
    from facecloak.weights import sha256_file

    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m facecloak.weights --fetch {what} --force-kaggle-only`")
    if expected is None:
        raise RuntimeError(f"no pinned sha256 for {what}; refusing to load an unverified checkpoint")
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"sha256 mismatch for {path}: got {got}, pinned {expected}")


def _import_simswap(repo: Path):
    """Import ``models.fs_networks`` / ``models.arcface_models`` from the cloned repo (read-only)."""
    if not (repo / "models" / "fs_networks.py").exists():
        raise FileNotFoundError(f"SimSwap repo missing at {repo}; git clone https://github.com/neuralchen/SimSwap there")
    existing = sys.modules.get("models")
    if existing is not None and Path(getattr(existing, "__file__", "") or "").parent != repo / "models":
        raise RuntimeError("a different top-level 'models' package is already imported; cannot load SimSwap")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return importlib.import_module("models.fs_networks"), importlib.import_module("models.arcface_models")


class SimSwapSwap:
    """SimSwap (224, 'people') black-box attacker.

    Pipeline (mirrors ``test_wholeimage_swapsingle.py`` with ``--crop_size 224``, ``--use_mask`` off):
      1. Detect with buffalo_l SCRFD (OUR detector's 5-pt landmarks; SimSwap's own detector is an
         insightface 'antelope' pack we do not ship).
      2. Align source and target to 224 px with SimSwap's 'None' templates (:func:`simswap_estimate_norm`).
      3. Source crop -> ToTensor + ImageNet normalise -> ``F.interpolate`` to 112 (default *nearest*, as in
         the original) -> SimSwap's ArcFace -> L2-normalise = identity latent.
      4. Generator on the target crop (RGB in [0,1], no normalisation), output in [0,1].
      5. Paste back with the inverse affine and a feathered mask. The original uses a BiSeNet face-parsing
         model for the optional mask; its weights (79999_iter.pth) are NOT in checkpoints.zip and are
         off by default upstream, so we use upstream's default non-parsing path (eroded + Gaussian
         blurred warped-square mask), with kernel sizes scaled by mask size instead of fixed 40/41 px
         (upstream constants are tuned for large photos and would erase the mask on 250 px LFW faces).

    Shims for torch 2.14 / numpy 2 (the cloned repo is never modified):
      * We do not import ``util/reverse2original.py`` / ``test_*.py`` (they use removed ``np.float`` and
        ``fractions.gcd``, hard-coded ``.cuda()``); paste-back is re-implemented here with numpy/cv2.
      * ``fs_model.py`` is not imported either (``SpecificNorm`` hard-codes ``.cuda()`` and the file needs
        training-only deps); the generator is built directly from ``fs_networks``.
      * ``arcface_checkpoint.tar`` is loaded through :class:`_RestrictedUnpickler` (see module docstring).
      * ``models/config.py`` calls ``torch.cuda.is_available()`` at import - harmless on CPU.
      * Conv2d layers of the unpickled ArcFace gain ``_reversed_padding_repeated_twice`` if the pickle
        predates it (defensive; no-op for current files).

    Args:
        device: ``'cuda'`` / ``'cpu'`` (default: cuda if available).
        app: Optional pre-built FaceAnalysis (detection is enough) to share with other code.
        repo: Path of the cloned SimSwap repository.
    """

    def __init__(self, device: str | None = None, app: Any = None, repo: Path | str = SIMSWAP_REPO) -> None:
        import torch

        from facecloak.models import load_face_app, num_threads
        from facecloak.weights import MANIFEST

        torch.set_num_threads(num_threads())
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.app = app if app is not None else load_face_app()
        repo = Path(repo)
        fsn, arc = _import_simswap(repo)

        _require_sha(SIMSWAP_GEN, MANIFEST["simswap_checkpoints"].members["latest_net_G.pth"], "simswap_checkpoints")
        _require_sha(SIMSWAP_ARC, MANIFEST["simswap_arcface"].sha256, "simswap_arcface")

        gen = fsn.Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        state = torch.load(SIMSWAP_GEN, map_location="cpu", weights_only=True)
        gen.load_state_dict(state, strict=True)
        self.gen = gen.eval().requires_grad_(False).to(self.device)

        net = torch.load(
            SIMSWAP_ARC, map_location="cpu", weights_only=False, pickle_module=_restricted_pickle_module()
        )
        if not isinstance(net, arc.ResNet):
            raise TypeError(f"unexpected arcface checkpoint object {type(net)}")
        for m in net.modules():
            if isinstance(m, torch.nn.Conv2d) and not hasattr(m, "_reversed_padding_repeated_twice"):
                m._reversed_padding_repeated_twice = torch.nn.modules.utils._reverse_repeat_tuple(m.padding, 2)
        self.arc = net.eval().requires_grad_(False).to(self.device)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    # -- pieces
    @staticmethod
    def _crop(bgr: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = simswap_estimate_norm(kps)
        return cv2.warpAffine(bgr, M, (_SIMSWAP_CROP, _SIMSWAP_CROP), borderValue=0.0), M

    def _identity_latent(self, crop_bgr: np.ndarray):
        torch = self.torch
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        x = (x - self._mean) / self._std
        x = torch.nn.functional.interpolate(x, size=(112, 112))  # nearest, as upstream
        return torch.nn.functional.normalize(self.arc(x), p=2, dim=1)

    @staticmethod
    def _paste_back(fake_rgb01: np.ndarray, M: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """Inverse-affine the 224 swap into the target with a feathered mask. Returns BGR uint8."""
        h, w = target_bgr.shape[:2]
        IM = cv2.invertAffineTransform(M)
        fake = cv2.warpAffine(fake_rgb01[:, :, ::-1].astype(np.float32), IM, (w, h))  # BGR [0,1]
        white = cv2.warpAffine(np.full((_SIMSWAP_CROP,) * 2, 255, np.float32), IM, (w, h))
        white[white > 20] = 255
        ys, xs = np.where(white == 255)
        if len(ys) == 0:
            return target_bgr.copy()
        size = int(np.sqrt((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)))
        k_er = max(3, round(0.18 * size))  # upstream: 40 px on a ~224 px crop
        k_bl = 2 * max(1, round(0.09 * size)) + 1  # upstream: 41 px
        mask = cv2.erode(white, np.ones((k_er, k_er), np.uint8))
        mask = cv2.GaussianBlur(mask, (k_bl, k_bl), 0)[:, :, None] / 255.0
        out = mask * (fake * 255.0) + (1 - mask) * target_bgr.astype(np.float32)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def swap(self, source_rgb: np.ndarray, target_rgb: np.ndarray) -> SwapResult:
        """Paste the identity of ``source_rgb`` onto the largest face of ``target_rgb``."""
        torch = self.torch
        src_faces = _faces(self.app, _bgr(source_rgb))
        if not src_faces:
            return SwapResult(None, False, 0.0, None)
        tgt_bgr = _bgr(target_rgb)
        tgt_faces = _faces(self.app, tgt_bgr)
        if not tgt_faces:
            raise NoFaceError("no face detected in target image")
        src, tgt = _largest(src_faces), _largest(tgt_faces)
        src_crop, _ = self._crop(_bgr(source_rgb), np.asarray(src.kps))
        tgt_crop, M = self._crop(tgt_bgr, np.asarray(tgt.kps))
        with torch.no_grad():
            latent = self._identity_latent(src_crop)
            t = (
                torch.from_numpy(np.ascontiguousarray(tgt_crop[:, :, ::-1]))
                .permute(2, 0, 1)
                .float()
                .div(255)
                .unsqueeze(0)
                .to(self.device)
            )
            fake = self.gen(t, latent)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        out_bgr = self._paste_back(fake, M, tgt_bgr)
        return SwapResult(
            image=np.ascontiguousarray(out_bgr[:, :, ::-1]),
            source_detected=True,
            source_det_score=float(src.det_score),
            source_embedding=latent[0].cpu().numpy().astype(np.float32),
        )


def get_swapper(name: str, **kwargs: Any) -> InswapperSwap | SimSwapSwap:
    """Factory: ``'inswapper'`` or ``'simswap'``."""
    if name == "inswapper":
        return InswapperSwap(**kwargs)
    if name == "simswap":
        return SimSwapSwap(**kwargs)
    raise ValueError(f"unknown swapper {name!r}; expected 'inswapper' or 'simswap'")
