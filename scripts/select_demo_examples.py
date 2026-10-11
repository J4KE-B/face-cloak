"""Pick built-in demo "victim" photos whose face swap is unmistakable, and check the cloak defeats each one.

Input: a folder of public-domain portraits named ``<slug>_a.jpg`` (the photo that gets swapped / cloaked) and
``<slug>_b.jpg`` (a second photo of the same person, used as the judge's reference), plus ``sources.json``.

For every victim x bundled target it runs the demo's own pipeline (``server.real_engine.RealEngine``):
swap(victim_a -> target), then scores it with the held-out judge (glintr100):

    sim_victim = cos(judge(swap), judge(victim_b))     high  = the swap carries the victim's identity
    sim_target = cos(judge(swap), judge(target))       low   = the swap no longer reads as the target person
    score      = sim_victim - sim_target

The best pair per victim is then cloaked (fast mode, the demo defaults) and attacked again; a pair is kept only
if the clean swap clears tau by ``--margin`` and the cloaked swap falls below tau by the same margin.
Writes ``<out>/selection.json`` and a private contact sheet ``<out>/contact_sheet.jpg`` for a visual check.

CPU heavy: run it alone, inside the capped scope (see server/run_local.sh).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def load_rgb(p: Path, max_side: int = 1024) -> np.ndarray:
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"unreadable: {p}")
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def tile(img: np.ndarray, size: int = 220, label: str = "") -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.full((size + 22, size, 3), 255, np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y : y + small.shape[0], x : x + small.shape[1]] = small
    cv2.putText(out, label[:34], (3, size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--victims", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--cloak-top", type=int, default=8, help="cloak the best pair of this many victims")
    args = ap.parse_args()

    from server.real_engine import RealEngine

    eng = RealEngine()
    args.out.mkdir(parents=True, exist_ok=True)
    tau = eng.tau
    targets = {t.id: eng._targets[t.id] for t in eng.targets()}
    target_emb = {}
    for tid, (_, rgb, _) in targets.items():
        f = eng.detect(rgb)
        target_emb[tid] = eng.embed_judge(rgb, f)

    sources = {s["slug"]: s for s in json.loads((args.victims / "sources.json").read_text())}
    victims = []
    for a in sorted(args.victims.glob("*_a.jpg")):
        slug = a.name[: -len("_a.jpg")]
        b = args.victims / f"{slug}_b.jpg"
        if not b.exists():
            continue
        img_a, img_b = load_rgb(a), load_rgb(b)
        fa, fb = eng.detect(img_a), eng.detect(img_b)
        if fa is None or fb is None:
            print(f"skip {slug}: no face in {'a' if fa is None else 'b'}")
            continue
        ref = eng.embed_judge(img_b, fb)
        ab = float(eng.embed_judge(img_a, fa) @ ref)  # same person, two photos: should be well above tau
        victims.append({"slug": slug, "a": img_a, "fa": fa, "ref": ref, "a_vs_b": ab})
        print(f"{slug}: a-vs-b judge sim {ab:.3f}")

    rows = []
    t0 = time.time()
    for v in victims:
        for tid in targets:
            sw = eng.swap(v["a"], tid)
            if sw is None:
                continue
            sv = eng.judge_swap(sw, tid, v["ref"])
            st = eng.judge_swap(sw, tid, target_emb[tid])
            rows.append({"slug": v["slug"], "target": tid, "sim_victim": sv, "sim_target": st, "score": sv - st})
    print(f"{len(rows)} clean swaps in {time.time() - t0:.0f}s")

    best = {}
    for r in sorted(rows, key=lambda r: -r["score"]):
        best.setdefault(r["slug"], r)
    ranked = sorted(best.values(), key=lambda r: -r["score"])[: args.cloak_top]

    vmap = {v["slug"]: v for v in victims}
    sheet_rows, kept = [], []
    for r in ranked:
        v = vmap[r["slug"]]
        t1 = time.time()
        cl = eng.cloak(v["a"], v["fa"], args.eps, args.steps, "fast", lambda i, n: None)
        sw_clean = eng.swap(v["a"], r["target"])
        sw_cloak = eng.swap(cl, r["target"])
        r["sim_victim_cloaked"] = None if sw_cloak is None else eng.judge_swap(sw_cloak, r["target"], v["ref"])
        r["psnr"] = eng.quality(v["a"], cl)[0]
        r["cloak_seconds"] = round(time.time() - t1, 1)
        r["a_vs_b"] = v["a_vs_b"]
        r["ok"] = bool(
            r["sim_victim"] >= tau + args.margin
            and r["sim_victim_cloaked"] is not None
            and r["sim_victim_cloaked"] <= tau - args.margin
        )
        print(
            f"{r['slug']:>12} -> {r['target']:<9} clean {r['sim_victim']:.3f}  cloaked "
            f"{r['sim_victim_cloaked'] if r['sim_victim_cloaked'] is None else round(r['sim_victim_cloaked'], 3)}"
            f"  target-sim {r['sim_target']:.3f}  psnr {r['psnr']:.1f}  ok={r['ok']}"
        )
        if r["ok"]:
            kept.append(r)
        tgt_rgb = targets[r["target"]][1]
        blank = np.zeros_like(v["a"])
        sheet_rows.append(
            np.hstack(
                [
                    tile(v["a"], label=f"victim {r['slug']}"),
                    tile(tgt_rgb, label=f"target {r['target']}"),
                    tile(sw_clean if sw_clean is not None else blank, label=f"swap clean {r['sim_victim']:.2f}"),
                    tile(
                        sw_cloak if sw_cloak is not None else blank,
                        label=f"swap cloaked {r['sim_victim_cloaked'] or 0:.2f}",
                    ),
                ]
            )
        )

    if sheet_rows:
        cv2.imwrite(str(args.out / "contact_sheet.jpg"), np.vstack(sheet_rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    out = {
        "tau": tau,
        "steps": args.steps,
        "eps": args.eps,
        "margin": args.margin,
        "kept": kept,
        "ranked": ranked,
        "all_pairs": rows,
        "sources": {k: sources.get(k) for k in vmap},
    }
    (args.out / "selection.json").write_text(json.dumps(out, indent=2))
    print(f"kept {len(kept)}/{len(ranked)}; wrote {args.out / 'selection.json'}")


if __name__ == "__main__":
    main()
