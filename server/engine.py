"""Inference engine abstraction.

``Engine`` is the only thing the HTTP layer / job worker know about, so tests inject ``StubEngine`` and never
load model weights. ``RealEngine`` (in ``server.real_engine``) wraps the facecloak package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from server.schema import Target


@dataclass
class Face:
    bbox: np.ndarray  # (4,) float32 x1,y1,x2,y2
    kps: np.ndarray  # (5,2) float32
    score: float = 1.0


class JobCancelled(Exception):
    """Raised from the progress callback to abort a running cloak."""


class Engine(Protocol):
    device: str
    tau: float
    default_steps: int

    def targets(self) -> list[Target]: ...

    def has_target(self, target_id: str) -> bool: ...

    def detect(self, img_rgb: np.ndarray) -> Face | None:
        """Largest face of an RGB uint8 image (must be thread-safe: called from request threads)."""

    def embed_judge(self, img_rgb: np.ndarray, face: Face) -> np.ndarray:
        """Identity embedding under the held-out judge (glintr100), L2-normalised."""

    def cloak(
        self,
        img_rgb: np.ndarray,
        face: Face,
        eps: int,
        steps: int,
        mode: str,
        on_step: Callable[[int, int], None],
    ) -> np.ndarray:
        """Return the cloaked uint8 image. ``on_step(step, total)`` is called once per PGD step."""

    def swap(self, src_rgb: np.ndarray, target_id: str) -> np.ndarray | None:
        """Face-swap attack: identity of ``src_rgb`` onto the target. None if the source face is not found."""

    def judge_swap(self, swapped_rgb: np.ndarray, target_id: str, ref_emb: np.ndarray) -> float:
        """cosine(judge embedding of the swapped face, ref_emb)."""

    def quality(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float | None]:
        """(psnr, ssim, lpips-or-None) between two uint8 RGB images."""


class StubEngine:
    """Instant fake engine (tests, schema export): clean swaps match (0.6), swaps of cloaked images do not."""

    device = "cpu"
    tau = 0.144
    default_steps = 30
    models_loaded = True

    def __init__(
        self, detect_ok: bool = True, clean_matches: bool = True, cloaked_undetected: bool = False
    ) -> None:
        self.detect_ok = detect_ok
        self.clean_matches = clean_matches
        self.cloaked_undetected = cloaked_undetected
        self.cloak_calls = 0
        self._cloaked: list[np.ndarray] = []

    def targets(self) -> list[Target]:
        px = (
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////"
            "////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAB"
            "PxA="
        )
        return [Target(id="t1", name="Stub Target", thumbnail=px, source="https://example.org", license="PD")]

    def has_target(self, target_id: str) -> bool:
        return target_id == "t1"

    def detect(self, img_rgb: np.ndarray) -> Face | None:
        if not self.detect_ok:
            return None
        h, w = img_rgb.shape[:2]
        return Face(np.array([0, 0, w, h], np.float32), np.zeros((5, 2), np.float32))

    def embed_judge(self, img_rgb: np.ndarray, face: Face) -> np.ndarray:
        return np.ones(4, np.float32) / 2.0

    def cloak(self, img_rgb, face, eps, steps, mode, on_step):
        self.cloak_calls += 1
        for s in range(1, steps + 1):
            on_step(s, steps)
        out = img_rgb.astype(np.int16).copy()
        out[::2, ::2] += eps
        out = np.clip(out, 0, 255).astype(np.uint8)
        self._cloaked.append(out)
        return out

    def swap(self, src_rgb, target_id):
        is_cloaked = any(src_rgb is c for c in self._cloaked)
        if is_cloaked and self.cloaked_undetected:
            return None
        return np.full((128, 128, 3), 255 if is_cloaked else 0, np.uint8)

    def judge_swap(self, swapped_rgb, target_id, ref_emb):
        clean = int(swapped_rgb[0, 0, 0]) == 0
        if clean:
            return 0.6 if self.clean_matches else 0.05
        return -0.2

    def quality(self, a, b):
        return 40.0, 0.99, 0.001
