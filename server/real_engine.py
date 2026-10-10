"""Engine backed by the facecloak package (loads model weights; instantiate once at startup)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from facecloak.cloak import CloakConfig, Cloaker
from facecloak.metrics import LPIPSMetric, psnr, ssim
from facecloak.models import (
    detect_largest,
    load_arcface_torch,
    load_face_app,
    load_facenet_torch,
    num_threads,
)
from facecloak.paths import RESULTS_DIR
from facecloak.swap import InswapperSwap, PreparedTarget
from server import imaging
from server.engine import Face
from server.schema import Target

log = logging.getLogger("facecloak.server")
ASSETS = Path(__file__).parent / "assets" / "targets"


class _StepCounter:
    """Transparent recognizer proxy that reports true PGD progress without touching ``Cloaker``.

    ``Cloaker.cloak`` calls ``rec.embed_image`` once per model per PGD step with autograd enabled, and only
    under ``torch.no_grad`` for its (initial / every-10-steps) evaluation passes. Counting the grad-enabled
    calls therefore counts steps exactly.
    """

    def __init__(self, rec, counter: dict) -> None:
        self._rec, self._c = rec, counter

    def embed_image(self, img01, kps):
        if torch.is_grad_enabled():
            self._c["calls"] += 1
            self._c["on_call"]()
        return self._rec.embed_image(img01, kps)

    def __getattr__(self, name):
        return getattr(self._rec, name)


def _load_tau() -> float:
    for name in ("calibration.json", "calibration_quick.json"):
        p = RESULTS_DIR / name
        if p.exists():
            return float(json.loads(p.read_text())["glintr100"]["tau_far1"])
    raise FileNotFoundError(f"no calibration.json / calibration_quick.json in {RESULTS_DIR}")


class RealEngine:
    def __init__(self) -> None:
        from facecloak.models import ORTRecognizer

        t0 = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_steps = 100 if self.device == "cuda" else 30
        torch.set_num_threads(num_threads())
        self.tau = _load_tau()
        self.swapper = InswapperSwap()
        # detection-only SCRFD for request-time face finding (swapper.app also runs the recognition model)
        self.det_app = load_face_app()
        self.judge = ORTRecognizer("glintr100", "antelopev2")
        self.arc = load_arcface_torch().to(self.device)
        self.facenet = load_facenet_torch().to(self.device)
        self.lpips = LPIPSMetric(self.device)
        self._targets: dict[str, tuple[Target, np.ndarray, PreparedTarget]] = {}
        self._load_targets()
        log.info(
            "RealEngine ready on %s in %.1fs (tau=%.3f, %d targets)",
            self.device,
            time.time() - t0,
            self.tau,
            len(self._targets),
        )

    # ---- targets
    def _load_targets(self) -> None:
        import cv2
        from skimage import data

        meta = json.loads((ASSETS / "targets.json").read_text())
        entries = []
        for m in meta:
            bgr = cv2.imread(str(ASSETS / m["file"]))
            if bgr is None:
                log.warning("target %s unreadable", m["id"])
                continue
            entries.append((m, np.ascontiguousarray(bgr[:, :, ::-1])))
        entries.append(
            (
                {
                    "id": "collins",
                    "name": "Eileen Collins (NASA)",
                    "source": "https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut",
                    "license": "Public domain (US federal government / NASA)",
                },
                np.ascontiguousarray(data.astronaut()),
            )
        )
        for m, rgb in entries:
            try:
                prep = self.swapper.prepare_target(rgb)
            except ValueError:
                log.warning("no face in target %s; skipped", m["id"])
                continue
            t = Target(
                id=m["id"],
                name=m["name"],
                thumbnail=imaging.jpeg_data_url(rgb, 256),
                source=m["source"],
                license=m["license"],
            )
            self._targets[m["id"]] = (t, rgb, prep)

    def targets(self) -> list[Target]:
        return [t for t, _, _ in self._targets.values()]

    def has_target(self, target_id: str) -> bool:
        return target_id in self._targets

    # ---- detection / judge
    def detect(self, img_rgb: np.ndarray) -> Face | None:
        d = detect_largest(self.det_app, img_rgb)  # detection only; largest face
        return None if d is None else Face(d["bbox"], d["kps"], d["det_score"])

    def embed_judge(self, img_rgb: np.ndarray, face: Face) -> np.ndarray:
        return self.judge.embed(img_rgb, face.kps)

    # ---- cloak
    def cloak(self, img_rgb, face, eps, steps, mode, on_step: Callable[[int, int], None]) -> np.ndarray:
        if mode == "robust":
            recs, models, eot = (
                {"w600k_r50": self.arc, "facenet_vggface2": self.facenet},
                {"w600k_r50": 1.0, "facenet_vggface2": 1.0},
                {},
            )
        else:
            recs, models, eot = {"w600k_r50": self.arc}, {"w600k_r50": 1.0}, None
        counter = {"calls": 0, "on_call": lambda: None}
        n_models = len(models)

        def tick() -> None:
            on_step(min(steps, (counter["calls"] - 1) // n_models + 1), steps)  # may raise JobCancelled

        counter["on_call"] = tick
        wrapped = {k: _StepCounter(r, counter) for k, r in recs.items()}
        cfg = CloakConfig(name=f"demo_{mode}", eps=float(eps), steps=steps, models=models, eot=eot)
        try:
            res = Cloaker(cfg, wrapped, device=self.device).cloak(img_rgb, face.kps, face.bbox)
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return res.image

    # ---- attack + judge
    def swap(self, src_rgb: np.ndarray, target_id: str) -> np.ndarray | None:
        return self.swapper.swap(src_rgb, self._targets[target_id][2]).image

    def judge_swap(self, swapped_rgb: np.ndarray, target_id: str, ref_emb: np.ndarray) -> float:
        import cv2

        prep = self._targets[target_id][2]
        tc = (prep.face.bbox[:2] + prep.face.bbox[2:]) / 2
        faces = self.swapper.app.get(cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR))
        # the swapped identity sits on the target's face: pick the detection nearest to the target face centre
        kps = (
            min(faces, key=lambda f: float(np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - tc))).kps
            if faces
            else prep.face.kps
        )
        return float(self.judge.embed(swapped_rgb, np.asarray(kps, np.float32)) @ ref_emb)

    def quality(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float | None]:
        try:
            lp: float | None = float(self.lpips(a, b))
        except Exception:  # noqa: BLE001  # e.g. AlexNet backbone download blocked
            lp = None
        return float(psnr(a, b)), float(ssim(a, b)), lp
