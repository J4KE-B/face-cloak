"""Experiment runner: (attack x source x target x transform x swapper) grid with caching and resume.

Design notes
------------
* Everything heavy (models, swappers, cloaker) comes from an injectable *factory* that returns a
  :class:`Components`; the factory is only called when there is work left, so a resumed run with
  nothing pending never touches model weights. Tests inject fakes.
* Two result tables per config, both :class:`facecloak.io.ResumableCSV` (truncation-safe):
  ``<config>_cloak.csv`` keyed (attack, src_id) and ``<config>_swap.csv`` keyed
  (attack, src_id, tgt_id, transform, swapper).
* Cloaked images are cached losslessly as PNG (``<cache>/<config>/<attack>/<src_id>.png``) so that
  transforms / swappers / judges reuse them and a resume skips PGD.
* Evaluator-leakage guard: held-out judges must never be attack models. A judge may coincide with an
  attack model only if it is explicitly listed in ``whitebox_judges`` (reported as white-box).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from facecloak.cloak import EVAL_ONLY_NAMES, CloakConfig
from facecloak.data import load_manifest, shard
from facecloak.io import ResumableCSV, read_rgb, write_png
from facecloak.metrics import psnr, ssim
from facecloak.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR
from facecloak.transforms import ROBUSTNESS

NAN = float("nan")
SWAPPERS = ("inswapper", "simswap")
ATTACK_MODEL_NAMES = ("w600k_r50", "facenet_vggface2")  # columns for final white-box cosines
CLOAK_KEY = ["attack", "src_id"]
SWAP_KEY = ["attack", "src_id", "tgt_id", "transform", "swapper"]


# ----------------------------------------------------------------------------- config
@dataclass
class ExpConfig:
    name: str
    manifest: Path
    attacks: dict[str, dict[str, Any]]
    swappers: list[str]
    transforms: list[str]
    judges: list[str]
    whitebox_judges: list[str] = field(default_factory=list)
    n_targets_per_src: int | None = None
    seed: int = 0
    device: str = "auto"
    lfw_root: Path | None = None
    calibration: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_judge(self) -> str:
        held_out = [j for j in self.judges if j not in self.whitebox_judges]
        return (held_out or self.judges)[0]

    def attack_kind(self, name: str) -> str:
        return name if name in ("clean", "noise") else "pgd"

    def cloak_config(self, name: str) -> CloakConfig:
        d = dict(self.attacks[name])
        d.setdefault("seed", self.seed)
        return CloakConfig.from_dict({"name": name, **d})

    def attack_models(self, name: str) -> list[str]:
        return list(self.cloak_config(name).models) if self.attack_kind(name) == "pgd" else []


def _repo_path(p: str | os.PathLike | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(src: str | os.PathLike | dict[str, Any]) -> ExpConfig:
    """Load a YAML config (path) or an already-parsed dict into a validated :class:`ExpConfig`."""
    if isinstance(src, dict):
        raw = dict(src)
    else:
        import yaml

        raw = yaml.safe_load(Path(src).read_text())
        raw.setdefault("name", Path(src).stem)
    cfg = ExpConfig(
        name=raw["name"],
        manifest=_repo_path(raw["manifest"]),
        attacks={k: dict(v or {}) for k, v in raw["attacks"].items()},
        swappers=list(raw["swappers"]),
        transforms=list(raw["transforms"]),
        judges=list(raw["judges"]),
        whitebox_judges=list(raw.get("whitebox_judges", [])),
        n_targets_per_src=raw.get("n_targets_per_src"),
        seed=int(raw.get("seed", 0)),
        device=str(raw.get("device", "auto")),
        lfw_root=_repo_path(raw.get("lfw_root")),
        calibration=_repo_path(raw.get("calibration")),
        raw=raw,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExpConfig) -> None:
    bad_sw = set(cfg.swappers) - set(SWAPPERS)
    if bad_sw:
        raise ValueError(f"unknown swappers {sorted(bad_sw)}")
    bad_tf = set(cfg.transforms) - set(ROBUSTNESS)
    if bad_tf:
        raise ValueError(f"unknown transforms {sorted(bad_tf)}; known: {sorted(ROBUSTNESS)}")
    if "none" not in cfg.transforms:
        raise ValueError("transforms must include 'none'")
    if not cfg.attacks or not cfg.judges or not cfg.swappers:
        raise ValueError("attacks, judges and swappers must be non-empty")
    for name in cfg.attacks:
        if cfg.attack_kind(name) == "pgd":
            cfg.cloak_config(name)  # raises on unknown keys / bad mode
    check_evaluator_leakage(cfg)


def check_evaluator_leakage(cfg: ExpConfig) -> None:
    """Held-out judges must not be attack models; a white-box judge must be declared as such."""
    wb = set(cfg.whitebox_judges)
    for name in cfg.attacks:
        models = set(cfg.attack_models(name))
        leaked = models & EVAL_ONLY_NAMES
        if leaked:
            raise ValueError(f"attack {name!r} uses evaluation-only model(s) {sorted(leaked)}")
        undeclared = (models & set(cfg.judges)) - wb
        if undeclared:
            raise ValueError(
                f"evaluator leakage: judge(s) {sorted(undeclared)} are attack models of {name!r} but not "
                "listed in whitebox_judges"
            )
    if not [j for j in cfg.judges if j not in wb]:
        raise ValueError("need at least one held-out (non-white-box) judge")


# ----------------------------------------------------------------------------- plan
def select_sources(manifest: dict, shard_spec: str = "0/1", limit: int | None = None) -> list[dict]:
    srcs = shard(list(manifest["sources"]), shard_spec)
    return srcs[:limit] if limit else srcs


def targets_by_source(manifest: dict, n: int | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sid, tid in manifest["pairs"]:
        out.setdefault(sid, []).append(tid)
    return {s: (t[:n] if n else t) for s, t in out.items()}


def expected_keys(
    cfg: ExpConfig, manifest: dict, shard_spec: str = "0/1", limit: int | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, ...]]]:
    """All (attack, src) cloak keys and (attack, src, tgt, transform, swapper) swap keys, in run order."""
    srcs = select_sources(manifest, shard_spec, limit)
    tg = targets_by_source(manifest, cfg.n_targets_per_src)
    cloaks, swaps = [], []
    for a in cfg.attacks:
        for s in srcs:
            cloaks.append((a, s["id"]))
            for t in tg.get(s["id"], []):
                for tf in cfg.transforms:
                    for sw in cfg.swappers:
                        swaps.append((a, s["id"], t, tf, sw))
    return cloaks, swaps


# ----------------------------------------------------------------------------- csv helpers
def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a results CSV without modifying it; a half-written last line is dropped."""
    data = Path(path).read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
    lines = data.decode().splitlines()
    if not lines:
        return [], []
    reader = csv.DictReader(lines)
    return list(reader.fieldnames or []), list(reader)


def key_cols_for(filename: str) -> list[str] | None:
    if filename.endswith("_cloak.csv"):
        return CLOAK_KEY
    if filename.endswith("_swap.csv"):
        return SWAP_KEY
    return None


def done_keys(paths: Iterable[Path], key_cols: list[str]) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for p in paths:
        if Path(p).exists():
            keys |= {tuple(r.get(k, "") for k in key_cols) for r in read_csv_rows(p)[1]}
    return keys


# ----------------------------------------------------------------------------- merge / ingest
@dataclass
class MergeReport:
    rows_in: dict[str, int] = field(default_factory=dict)
    rows_out: dict[str, int] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)
    pngs_copied: int = 0
    pngs_skipped: int = 0

    def lines(self) -> list[str]:
        out = []
        for f in sorted(self.rows_out):
            out.append(f"{f}: {self.rows_in[f]} rows in -> {self.rows_out[f]} unique ({self.duplicates[f]} duplicates)")
        out.append(f"cached PNGs: {self.pngs_copied} copied, {self.pngs_skipped} already present")
        return out


def _scan(root: Path) -> tuple[list[Path], list[Path], list[tuple[Path, Path]]]:
    """(csvs, metas, [(png, path-relative-to-a-'cloaked'-dir)]) below ``root``.

    For a ``.../raw`` directory the sibling ``../cache/cloaked`` is scanned too.
    """
    csvs, metas, pngs = [], [], []
    for p in sorted(root.rglob("*")):
        if p.name.endswith(("_cloak.csv", "_swap.csv")):
            csvs.append(p)
        elif p.name.endswith("_meta.json"):
            metas.append(p)
        elif p.suffix == ".png":
            parts = p.relative_to(root).parts
            if "cloaked" in parts:
                i = parts.index("cloaked")
                pngs.append((p, Path(*parts[i + 1 :])))
    sib = root.parent / "cache" / "cloaked"
    if root.name == "raw" and sib.is_dir():
        pngs += [(p, p.relative_to(sib)) for p in sorted(sib.rglob("*.png"))]
    return csvs, metas, pngs


def merge_into(inputs: Iterable[str | os.PathLike], out_dir: Path, cache_dir: Path | None = None) -> MergeReport:
    """Concatenate result CSVs from zips / directories into ``out_dir``, de-duplicating on keys.

    First occurrence wins (inputs are consulted in the given order, rows in file order). Cached cloaked
    PNGs are copied into ``cache_dir`` when absent. ``out_dir`` may itself be one of the inputs.
    """
    out_dir = Path(out_dir)
    report = MergeReport()
    rows: dict[str, list[dict[str, str]]] = {}
    fields: dict[str, list[str]] = {}
    seen: dict[str, set[tuple[str, ...]]] = {}
    metas: dict[str, dict[str, Any]] = {}
    png_jobs: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix="facecloak_merge_") as tmp:
        roots: list[Path] = []
        for i, inp in enumerate(inputs):
            p = Path(inp)
            if p.is_dir():
                roots.append(p)
            elif zipfile.is_zipfile(p):
                dest = Path(tmp) / f"zip{i}"
                with zipfile.ZipFile(p) as z:
                    z.extractall(dest)
                roots.append(dest)
            elif p.exists():
                raise ValueError(f"{p} is neither a directory nor a zip file")
        for root in roots:
            csvs, metas_found, pngs = _scan(root)
            for c in csvs:
                keys = key_cols_for(c.name)
                hdr, rs = read_csv_rows(c)
                f = c.name
                fields.setdefault(f, [])
                fields[f] += [h for h in hdr if h not in fields[f]]
                rows.setdefault(f, [])
                seen.setdefault(f, set())
                report.rows_in[f] = report.rows_in.get(f, 0) + len(rs)
                report.duplicates.setdefault(f, 0)
                for r in rs:
                    k = tuple(r.get(x, "") for x in keys)
                    if k in seen[f]:
                        report.duplicates[f] += 1
                        continue
                    seen[f].add(k)
                    rows[f].append(r)
            for m in metas_found:
                try:
                    d = json.loads(m.read_text())
                except json.JSONDecodeError:
                    continue
                if m.name in metas:
                    metas[m.name]["runs"] = metas[m.name].get("runs", []) + d.get("runs", [])
                else:
                    metas[m.name] = d
            if cache_dir is not None:
                png_jobs += [(png, Path(cache_dir) / rel) for png, rel in pngs]
        out_dir.mkdir(parents=True, exist_ok=True)
        for f, rs in rows.items():
            tmp_path = out_dir / (f + ".tmp")
            with open(tmp_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields[f], restval="")
                w.writeheader()
                w.writerows(rs)
            tmp_path.replace(out_dir / f)
            report.rows_out[f] = len(rs)
        for name, d in metas.items():
            (out_dir / name).write_text(json.dumps(d, indent=1))
        for src, dst in png_jobs:
            if dst.exists() or src.resolve() == dst.resolve():
                report.pngs_skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            report.pngs_copied += 1
    return report


# ----------------------------------------------------------------------------- components
@dataclass
class Components:
    """Everything the runner needs from the (heavy) outside world."""

    device: str
    judges: dict[str, Any]  # name -> .embed(img_rgb_uint8, kps) -> L2-normalised vector
    swappers: dict[str, Any]  # name -> .swap(src_rgb, tgt_rgb) -> SwapResult; .app.get(bgr) -> faces
    make_cloaker: Callable[[CloakConfig], Any]  # -> object with .cloak(img, kps, bbox, decoy_embs)
    decoy_fn: Callable[[np.ndarray, np.ndarray, list[str]], tuple[str, dict]]  # -> (decoy tgt id, embs)
    lpips: Callable[[np.ndarray, np.ndarray], float]
    detect: Callable[[np.ndarray], float]  # attacker detection score of the largest face (0 if none)


Factory = Callable[[ExpConfig, str, dict], Components]


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def lfw_root_for(cfg: ExpConfig) -> Path:
    return cfg.lfw_root or Path(os.environ.get("FACECLOAK_LFW") or DATA_DIR / "lfw")


def default_factory(cfg: ExpConfig, device: str, manifest: dict) -> Components:
    """Real models. Heavy; loads each model exactly once."""
    from facecloak.cloak import Cloaker, decoy_embeddings, embed_uint8, select_decoy
    from facecloak.metrics import LPIPSMetric
    from facecloak.models import ATTACK_MODELS, load_eval_recognizers
    from facecloak.swap import _faces, _largest, get_swapper

    root = lfw_root_for(cfg)
    all_judges = load_eval_recognizers()
    judges = {j: all_judges[j] for j in cfg.judges}
    swappers = {}
    for name in cfg.swappers:
        swappers[name] = get_swapper(name, device=device) if name == "simswap" else get_swapper(name)

    recs: dict[str, Any] = {}

    def rec(name: str):
        if name not in recs:
            recs[name] = ATTACK_MODELS[name]().to(device).eval().requires_grad_(False)
        return recs[name]

    shared: dict[str, Any] = {}

    def make_cloaker(cc: CloakConfig):
        lp = det = None
        if cc.lpips_weight > 0:
            if "lpips" not in shared:
                import lpips

                shared["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)
            lp = shared["lpips"]
        if cc.det_weight > 0:
            if "det" not in shared:
                from facecloak.det_torch import load_scrfd_torch

                shared["det"] = load_scrfd_torch().to(device).eval().requires_grad_(False)
            det = shared["det"]
        return Cloaker(cc, {m: rec(m) for m in cc.models}, lp, det, device)

    pool: dict[str, Any] = {}

    def decoy_fn(img: np.ndarray, kps: np.ndarray, models: list[str]):
        import torch

        if not pool:
            ids, imgs = [], {}
            for t in manifest["targets"]:
                ids.append(t["id"])
                imgs[t["id"]] = (read_rgb(str(root / t["path"])), np.asarray(t["kps"], np.float32))
            pool["ids"], pool["imgs"], pool["embs"] = ids, imgs, {}
        for m in set(models) | {"w600k_r50"}:  # selection always uses the white-box r50, never a judge
            if m not in pool["embs"]:
                pool["embs"][m] = torch.stack(
                    [embed_uint8(rec(m), *pool["imgs"][i], device) for i in pool["ids"]]
                )
        clean = embed_uint8(rec("w600k_r50"), img, kps, device)
        idx = select_decoy(clean.cpu(), pool["embs"]["w600k_r50"].cpu())
        tid = pool["ids"][idx]
        return tid, decoy_embeddings({m: rec(m) for m in models}, *pool["imgs"][tid], device)

    first_app = next(iter(swappers.values())).app

    def detect(img: np.ndarray) -> float:
        faces = _faces(first_app, np.ascontiguousarray(img[:, :, ::-1]))
        return float(_largest(faces).det_score) if faces else 0.0

    return Components(device, judges, swappers, make_cloaker, decoy_fn, LPIPSMetric(device), detect)


# ----------------------------------------------------------------------------- per-item logic
def locate_swapped_face(app: Any, swapped_rgb: np.ndarray, tgt_bbox: Iterable[float]):
    """Face in the swap output whose bbox centre is nearest the target's bbox centre (None if none
    lies within half the target face size, e.g. the swap destroyed the face)."""
    faces = app.get(np.ascontiguousarray(swapped_rgb[:, :, ::-1]))
    if not faces:
        return None
    x1, y1, x2, y2 = (float(v) for v in tgt_bbox)
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def dist(f) -> float:
        b = np.asarray(f.bbox, dtype=float)
        return float(np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - c))

    best = min(faces, key=dist)
    return best if dist(best) <= 0.5 * max(x2 - x1, y2 - y1) else None


def swap_and_score(
    swapper: Any,
    judges: dict[str, Any],
    cloaked_rgb: np.ndarray,
    tgt_rgb: np.ndarray,
    tgt_bbox: Iterable[float],
    ref_embs: dict[str, np.ndarray],
) -> tuple[Any, bool, dict[str, float]]:
    """Run the swap and score the swapped face against the reference with each judge.

    Returns (SwapResult-or-None, face_found, {judge: cosine or NaN}).
    """
    from facecloak.swap import NoFaceError

    try:
        res = swapper.swap(cloaked_rgb, tgt_rgb)
    except NoFaceError:
        res = None
    sims = dict.fromkeys(judges, NAN)
    if res is None or res.image is None:
        return res, False, sims
    face = locate_swapped_face(swapper.app, res.image, tgt_bbox)
    if face is None:
        return res, False, sims
    kps = np.asarray(face.kps, dtype=np.float32)
    for j, judge in judges.items():
        sims[j] = float(np.dot(judge.embed(res.image, kps), ref_embs[j]))
    return res, True, sims


def cloak_columns() -> list[str]:
    return (
        ["attack", "src_id", "decoy_id", "seconds", "psnr", "ssim", "lpips", "linf"]
        + [f"final_cos_{m}" for m in ATTACK_MODEL_NAMES]
        + ["det_score_after", "detected_after"]
    )


def swap_columns(judges: list[str]) -> list[str]:
    cols = ["attack", "src_id", "tgt_id", "transform", "swapper", "source_detected", "swap_face_found"]
    for j in judges:
        cols += [f"id_sim_swap_{j}", f"id_sim_direct_{j}", f"protected_{j}"]
    return cols + ["seconds"]


def load_tau(cfg: ExpConfig) -> tuple[dict[str, float], str]:
    """FAR=1% thresholds per judge; prefers results/calibration.json over calibration_quick.json."""
    cands = [cfg.calibration] if cfg.calibration else [RESULTS_DIR / "calibration.json", RESULTS_DIR / "calibration_quick.json"]
    for p in cands:
        if p and Path(p).exists():
            d = json.loads(Path(p).read_text())
            return {j: float(d[j]["tau_far1"]) for j in cfg.judges}, Path(p).name
    raise FileNotFoundError("no calibration file (results/calibration.json or calibration_quick.json)")


def _git_info() -> dict[str, Any]:
    def run(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], check=False, capture_output=True, text=True, timeout=20).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:  # noqa: BLE001 - git may be absent on Kaggle
        return {"commit": "unknown", "dirty": None}


def _versions() -> dict[str, str]:
    from importlib import metadata

    out = {}
    for pkg in ("torch", "numpy", "onnxruntime", "insightface", "opencv-python-headless", "lpips", "scikit-image", "onnx2torch"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "n/a"
    import sys

    out["python"] = sys.version.split()[0]
    return out


def _write_meta(path: Path, cfg: ExpConfig, run: dict[str, Any]) -> None:
    meta = json.loads(path.read_text()) if path.exists() else {"name": cfg.name, "config": cfg.raw, "runs": []}
    meta["runs"] = [r for r in meta["runs"] if r.get("run_id") != run["run_id"]] + [run]
    path.write_text(json.dumps(meta, indent=1, default=str))


def _noise_seed(cfg: ExpConfig, src_id: str) -> int:
    import zlib

    return (zlib.crc32(src_id.encode()) ^ cfg.seed) & 0x7FFFFFFF


# ----------------------------------------------------------------------------- the runner
def run_experiment(
    cfg: ExpConfig,
    *,
    out: Path,
    cache: Path,
    shard_spec: str = "0/1",
    limit: int | None = None,
    resume_from: Path | str | None = None,
    factory: Factory | None = None,
    tau: dict[str, float] | None = None,
    manifest: dict | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run (or resume) one config. Returns counts of work done / skipped."""
    check_evaluator_leakage(cfg)
    out, cache = Path(out), Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    if resume_from:
        rep = merge_into([out, resume_from], out, cache)
        print("resume-from merged:\n  " + "\n  ".join(rep.lines()))
    manifest = manifest or load_manifest(cfg.manifest)
    srcs = {s["id"]: s for s in manifest["sources"]}
    tgts = {t["id"]: t for t in manifest["targets"]}
    tg = targets_by_source(manifest, cfg.n_targets_per_src)
    ckeys, skeys = expected_keys(cfg, manifest, shard_spec, limit)

    cloak_csv = ResumableCSV(str(out / f"{cfg.name}_cloak.csv"), CLOAK_KEY)
    swap_csv = ResumableCSV(str(out / f"{cfg.name}_swap.csv"), SWAP_KEY)
    done_c, done_s = cloak_csv.done_keys(), swap_csv.done_keys()
    pending_swaps: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for k in skeys:
        if k not in done_s:
            pending_swaps.setdefault((k[0], k[1]), []).append(k)
    work = [k for k in ckeys if k not in done_c or k in pending_swaps]
    summary = {
        "cloaks_total": len(ckeys), "swaps_total": len(skeys),
        "cloaks_pending": sum(k not in done_c for k in ckeys),
        "swaps_pending": sum(len(v) for v in pending_swaps.values()),
        "cloaks_computed": 0, "swaps_written": 0,
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if not work:
        print(f"[{cfg.name}] nothing to do ({len(ckeys)} cloaks, {len(skeys)} swap rows complete)")
        return summary

    tau_src = "explicit"
    if tau is None:
        tau, tau_src = load_tau(cfg)
    device = resolve_device(cfg.device)
    meta_path = out / f"{cfg.name}_meta.json"
    run_meta = {
        "run_id": run_id, "started": run_id, "shard": shard_spec, "limit": limit, "device": device,
        "git": _git_info(), "versions": _versions(), "tau": tau, "tau_source": tau_src,
        "judges": cfg.judges, "whitebox_judges": cfg.whitebox_judges, "pending": summary.copy(),
    }
    if device == "cuda":
        import torch

        run_meta["gpu"] = torch.cuda.get_device_name(0)
    _write_meta(meta_path, cfg, run_meta)

    comps = (factory or default_factory)(cfg, device, manifest)
    if set(comps.judges) != set(cfg.judges):
        raise ValueError("factory judges do not match config")
    cloakers: dict[str, Any] = {}
    img_cache: dict[str, np.ndarray] = {}
    ref_embs: dict[str, dict[str, np.ndarray]] = {}

    def tgt_img(tid: str) -> np.ndarray:
        if tid not in img_cache:
            img_cache[tid] = read_rgb(str(lfw_root_for(cfg) / tgts[tid]["path"]))
        return img_cache[tid]

    def refs(sid: str) -> dict[str, np.ndarray]:
        if sid not in ref_embs:
            s = srcs[sid]
            B = read_rgb(str(lfw_root_for(cfg) / s["ref_path"]))
            kb = np.asarray(s["ref_kps"], np.float32)
            ref_embs[sid] = {j: comps.judges[j].embed(B, kb) for j in cfg.judges}
        return ref_embs[sid]

    cc_cols = cloak_columns()
    sw_cols = swap_columns(cfg.judges)
    it: Iterable = work
    if progress:
        from tqdm import tqdm

        it = tqdm(work, desc=cfg.name, unit="cloak")
    t_start = time.time()
    for attack, sid in it:
        s = srcs[sid]
        A = read_rgb(str(lfw_root_for(cfg) / s["src_path"]))
        kps = np.asarray(s["src_kps"], np.float32)
        png = cache / cfg.name / attack / f"{sid}.png"
        kind = cfg.attack_kind(attack)
        secs, cos, decoy_id = NAN, {}, ""
        if kind == "clean":
            cloaked, secs = A, 0.0
        elif png.exists():
            cloaked = read_rgb(str(png))
        else:
            t0 = time.time()
            if kind == "noise":
                from facecloak.cloak import noise_baseline

                cloaked = noise_baseline(A, float(cfg.attacks[attack].get("eps", 8.0)), _noise_seed(cfg, sid))
            else:
                cc = cfg.cloak_config(attack)
                if attack not in cloakers:
                    cloakers[attack] = comps.make_cloaker(cc)
                embs = None
                if cc.mode in ("targeted", "both"):
                    decoy_id, embs = comps.decoy_fn(A, kps, list(cc.models))
                res = cloakers[attack].cloak(A, kps, np.asarray(s["src_bbox"], np.float32), embs)
                cloaked, cos = res.image, res.final_cos
            secs = time.time() - t0
            write_png(str(png), cloaked)
            summary["cloaks_computed"] += 1
        if (attack, sid) not in done_c:
            score = comps.detect(cloaked)
            row = {
                "attack": attack, "src_id": sid, "decoy_id": decoy_id, "seconds": round(secs, 3),
                "psnr": psnr(A, cloaked), "ssim": ssim(A, cloaked), "lpips": comps.lpips(A, cloaked),
                "linf": int(np.abs(A.astype(np.int16) - cloaked.astype(np.int16)).max()),
                "det_score_after": round(score, 4), "detected_after": int(score > 0),
            }
            for m in ATTACK_MODEL_NAMES:
                row[f"final_cos_{m}"] = cos.get(m, NAN)
            cloak_csv.append({c: row[c] for c in cc_cols})
            done_c.add((attack, sid))

        pend = pending_swaps.get((attack, sid), [])
        by_tf: dict[str, list[tuple[str, ...]]] = {}
        for k in pend:
            by_tf.setdefault(k[3], []).append(k)
        eB = refs(sid) if pend else {}
        for tf, ks in by_tf.items():
            timg = ROBUSTNESS[tf](cloaked)
            direct = {j: float(np.dot(comps.judges[j].embed(timg, kps), eB[j])) for j in cfg.judges}
            for _, _, tid, _, swn in ks:
                t0 = time.time()
                res, found, sims = swap_and_score(
                    comps.swappers[swn], comps.judges, timg, tgt_img(tid), tgts[tid]["bbox"], eB
                )
                row = {
                    "attack": attack, "src_id": sid, "tgt_id": tid, "transform": tf, "swapper": swn,
                    "source_detected": int(bool(res is not None and res.source_detected)),
                    "swap_face_found": int(found),
                }
                for j in cfg.judges:
                    row[f"id_sim_swap_{j}"] = round(sims[j], 6) if found else NAN
                    row[f"id_sim_direct_{j}"] = round(direct[j], 6)
                    row[f"protected_{j}"] = int(sims[j] < tau[j]) if found else NAN
                row["seconds"] = round(time.time() - t0, 3)
                swap_csv.append({c: row[c] for c in sw_cols})
                summary["swaps_written"] += 1
        assert all(t in tg.get(sid, []) for _, _, t, _, _ in pend)
    run_meta["finished"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_meta["elapsed_s"] = round(time.time() - t_start, 1)
    run_meta["done"] = {k: summary[k] for k in ("cloaks_computed", "swaps_written")}
    _write_meta(meta_path, cfg, run_meta)
    return summary


# ----------------------------------------------------------------------------- remaining work
def remaining_work(
    cfg: ExpConfig, manifest: dict, results: Iterable[str | os.PathLike], n_shards: int = 2
) -> dict[str, Any]:
    """Which cloaks / swap rows are missing given the CSVs inside ``results`` (zips or dirs)."""
    with tempfile.TemporaryDirectory(prefix="facecloak_rem_") as tmp:
        merge_into(results, Path(tmp) / "raw", None)
        raw = Path(tmp) / "raw"
        dc = done_keys([raw / f"{cfg.name}_cloak.csv"], CLOAK_KEY)
        ds = done_keys([raw / f"{cfg.name}_swap.csv"], SWAP_KEY)
    ckeys, skeys = expected_keys(cfg, manifest)
    miss_c = [k for k in ckeys if k not in dc]
    miss_s = [k for k in skeys if k not in ds]
    shards = {}
    for i in range(n_shards):
        spec = f"{i}/{n_shards}"
        ids = {s["id"] for s in select_sources(manifest, spec)}
        shards[spec] = {
            "sources": len(ids),
            "cloaks": sum(k[1] in ids for k in miss_c),
            "swaps": sum(k[1] in ids for k in miss_s),
        }
    return {
        "cloaks_total": len(ckeys), "swaps_total": len(skeys),
        "missing_cloaks": miss_c, "missing_swaps": miss_s, "shards": shards,
    }
