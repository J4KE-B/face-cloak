"""Qualitative grid: original | cloaked | |delta| x10 | swap(original) | swap(cloaked), with id-sim captions.

Reads cached cloaked PNGs (results/cache/cloaked/<config>/<attack>/<src>.png). Swap panels are cached under
results/cache/qualitative/ and the swapper / judge are only loaded if a panel is missing (heavy; a few images).

Example:
    uv run python scripts/make_qualitative.py --config configs/main.yaml --attack pgd_ens_eot_lpips \
        --sources s0000 s0007 s0021 --swapper inswapper
"""

from __future__ import annotations

import os

_T = str(max(1, min(8, int(os.environ.get("FACECLOAK_THREADS", "8")))))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from facecloak.data import load_manifest
from facecloak.experiment import lfw_root_for, load_config, swap_and_score, targets_by_source
from facecloak.figures import FULL_W, save_fig, set_style
from facecloak.io import read_rgb, write_png
from facecloak.paths import RESULTS_DIR

COLS = ["Original", "Cloaked", r"$|\delta|\times 10$", "Swap (original)", "Swap (cloaked)"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--attack", default="pgd_ens_eot_lpips")
    ap.add_argument("--sources", nargs="*", default=None, help="source ids (default: first 4)")
    ap.add_argument("--swapper", default="inswapper")
    ap.add_argument("--judge", default=None, help="default: the config's held-out primary judge")
    ap.add_argument("--cache", default=str(RESULTS_DIR / "cache" / "cloaked"))
    ap.add_argument("--panel-cache", default=str(RESULTS_DIR / "cache" / "qualitative"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "figs"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    judge_name = a.judge or cfg.primary_judge
    manifest = load_manifest(cfg.manifest)
    srcs = {s["id"]: s for s in manifest["sources"]}
    tgts = {t["id"]: t for t in manifest["targets"]}
    tg = targets_by_source(manifest, 1)
    ids = a.sources or sorted(srcs)[:4]
    root = lfw_root_for(cfg)
    pdir = Path(a.panel_cache) / cfg.name / a.attack / a.swapper
    pdir.mkdir(parents=True, exist_ok=True)

    lazy: dict = {}

    def models():
        if not lazy:
            from facecloak.models import load_eval_recognizers
            from facecloak.swap import get_swapper

            lazy["judge"] = load_eval_recognizers()[judge_name]
            lazy["swapper"] = get_swapper(a.swapper)
        return lazy["judge"], lazy["swapper"]

    def panel(sid: str, tag: str, img: np.ndarray, tid: str):
        png, js = pdir / f"{sid}_{tag}.png", pdir / f"{sid}_{tag}.json"
        if png.exists() and js.exists():
            return read_rgb(str(png)), json.loads(js.read_text())["sim"]
        judge, swapper = models()
        s = srcs[sid]
        eB = {judge_name: judge.embed(read_rgb(str(root / s["ref_path"])), np.asarray(s["ref_kps"], np.float32))}
        res, found, sims = swap_and_score(
            swapper, {judge_name: judge}, img, read_rgb(str(root / tgts[tid]["path"])), tgts[tid]["bbox"], eB
        )
        if res is None or res.image is None:
            return None, None
        write_png(str(png), res.image)
        sim = sims[judge_name] if found else None
        js.write_text(json.dumps({"sim": sim}))
        return res.image, sim

    rows = []
    for sid in ids:
        cpng = Path(a.cache) / cfg.name / a.attack / f"{sid}.png"
        if sid not in srcs or not cpng.exists():
            print(f"notice: skipping {sid} (unknown source or no cached cloak at {cpng})")
            continue
        A = read_rgb(str(root / srcs[sid]["src_path"]))
        C = read_rgb(str(cpng))
        tid = tg[sid][0]
        so, sim_o = panel(sid, "orig", A, tid)
        sc, sim_c = panel(sid, "cloak", C, tid)
        diff = np.clip(np.abs(A.astype(np.int16) - C.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
        rows.append((sid, [A, C, diff, so, sc], ["", "", f"Linf={int(np.abs(A.astype(int) - C).max())}",
                                                  _cap(sim_o), _cap(sim_c)]))
    if not rows:
        print("notice: no panels available; nothing written")
        return
    set_style()
    fig, axs = plt.subplots(len(rows), 5, figsize=(FULL_W, FULL_W / 5 * len(rows) + 0.3), squeeze=False)
    for r, (sid, imgs, caps) in enumerate(rows):
        for c in range(5):
            ax = axs[r][c]
            ax.axis("off")
            if imgs[c] is not None:
                ax.imshow(imgs[c])
            else:
                ax.text(0.5, 0.5, "no face\ndetected", ha="center", va="center", transform=ax.transAxes, fontsize=6)
            if caps[c]:
                ax.set_title(caps[c], fontsize=6, y=-0.16)
            if r == 0:
                ax.text(0.5, 1.04, COLS[c], ha="center", va="bottom", transform=ax.transAxes, fontsize=7)
    fig.subplots_adjust(wspace=0.02, hspace=0.25)
    save_fig(fig, Path(a.out), "qualitative")


def _cap(sim) -> str:
    return "" if sim is None else f"id-sim {sim:.2f}"


if __name__ == "__main__":
    main()
