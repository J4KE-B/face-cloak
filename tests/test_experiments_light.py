"""Light tests of the experiment pipeline with fake models (no weights, no PGD)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from facecloak import experiment as ex
from facecloak.cloak import CloakResult
from facecloak.swap import SwapResult

H = W = 48
KPS = [[14, 18], [34, 18], [24, 28], [16, 38], [32, 38]]
BBOX = [8, 8, 40, 42]


class FakeJudge:
    def embed(self, img, kps):
        v = cv2.resize(img.astype(np.float32), (4, 4), interpolation=cv2.INTER_AREA).ravel() - 100.0
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class FakeApp:
    def get(self, bgr):
        return [SimpleNamespace(bbox=np.array([8, 8, 40, 42], float), kps=np.array(KPS, np.float32), det_score=0.99)]


class FakeSwapper:
    app = FakeApp()

    def __init__(self, name):
        self.name = name

    def swap(self, src, tgt):
        if src.std() < 1.0:  # flat image = "undetectable" source
            return SwapResult(None, False, 0.0, None)
        out = np.clip(0.5 * tgt.astype(np.float32) + 0.5 * src.astype(np.float32), 0, 255).astype(np.uint8)
        return SwapResult(out, True, 0.9, np.ones(4, np.float32))


class FakeCloaker:
    calls = 0

    def __init__(self, cc):
        self.cc = cc

    def cloak(self, img, kps, bbox=None, decoy_embs=None):
        type(self).calls += 1
        out = np.clip(img.astype(np.int16) + 8, 0, 255).astype(np.uint8)
        return CloakResult(out, 8, [], {"w600k_r50": 0.1}, 2, 0.01)


def fake_factory_builder():
    made = {"n": 0}

    def factory(cfg, device, manifest):
        made["n"] += 1
        judges = {j: FakeJudge() for j in cfg.judges}
        sw = {s: FakeSwapper(s) for s in cfg.swappers}
        return ex.Components(
            device="cpu", judges=judges, swappers=sw, make_cloaker=FakeCloaker,
            decoy_fn=lambda img, kps, models: ("t0", {}), lpips=lambda a, b: 0.05, detect=lambda img: 0.9,
        )

    factory.made = made
    return factory


@pytest.fixture()
def world(tmp_path):
    rng = np.random.default_rng(0)
    lfw = tmp_path / "lfw"
    (lfw / "lfw").mkdir(parents=True)

    def save(rel, arr):
        (lfw / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(lfw / rel), arr)
        return rel

    sources = []
    for i in range(2):
        a = save(f"lfw/S{i}/a.png", rng.integers(60, 200, (H, W, 3), dtype=np.uint8))
        b = save(f"lfw/S{i}/b.png", rng.integers(60, 200, (H, W, 3), dtype=np.uint8))
        sources.append({"id": f"s{i:04d}", "identity": f"S{i}", "src_path": a, "ref_path": b,
                        "src_kps": KPS, "src_bbox": BBOX, "ref_kps": KPS})
    targets = []
    for i in range(2):
        p = save(f"lfw/T{i}/a.png", rng.integers(60, 200, (H, W, 3), dtype=np.uint8))
        targets.append({"id": f"t{i:04d}", "identity": f"T{i}", "path": p, "kps": KPS, "bbox": BBOX})
    pairs = [[s["id"], t["id"]] for s in sources for t in targets]
    man = {"version": 1, "sources": sources, "targets": targets, "pairs": pairs}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(man))
    raw = {
        "name": "tiny", "manifest": str(mpath), "lfw_root": str(lfw), "n_targets_per_src": 2, "seed": 0,
        "swappers": ["inswapper", "simswap"], "transforms": ["none", "jpeg75"],
        "judges": ["glintr100", "w600k_r50"], "whitebox_judges": ["w600k_r50"],
        "attacks": {"clean": {}, "noise": {"eps": 8},
                    "pgd_r50": {"eps": 8, "steps": 2, "models": {"w600k_r50": 1.0}}},
    }
    return SimpleNamespace(cfg=ex.load_config(raw), man=man, tmp=tmp_path, raw=raw)


TAU = {"glintr100": 0.2, "w600k_r50": 0.2}


def run(w, out, shard="0/1", factory=None, **kw):
    f = factory or fake_factory_builder()
    s = ex.run_experiment(w.cfg, out=w.tmp / out, cache=w.tmp / "cache" / "cloaked", shard_spec=shard,
                          factory=f, tau=TAU, manifest=w.man, progress=False, **kw)
    return s, f


def keys(df, cols):
    return sorted(map(tuple, df[cols].astype(str).to_numpy().tolist()))


def test_columns_rows_and_nan_safety(world):
    s, _ = run(world, "raw")
    sw = pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")
    cl = pd.read_csv(world.tmp / "raw" / "tiny_cloak.csv")
    assert list(sw.columns) == ex.swap_columns(["glintr100", "w600k_r50"])
    assert list(cl.columns) == ex.cloak_columns()
    assert len(sw) == 3 * 2 * 2 * 2 * 2 == s["swaps_written"]
    assert len(cl) == 6
    assert not sw.duplicated(ex.SWAP_KEY).any()
    assert set(sw["protected_glintr100"].dropna()) <= {0.0, 1.0}
    assert (world.tmp / "cache" / "cloaked" / "tiny" / "pgd_r50" / "s0000.png").exists()
    assert not (world.tmp / "cache" / "cloaked" / "tiny" / "clean").exists()
    meta = json.loads((world.tmp / "raw" / "tiny_meta.json").read_text())
    assert meta["runs"][-1]["tau"] == TAU and meta["config"]["name"] == "tiny"


def test_undetected_source_gives_nan_not_dropped(world):
    # a flat (std<1) cloak makes the fake attacker miss the face
    class Flat(FakeCloaker):
        def cloak(self, img, kps, bbox=None, decoy_embs=None):
            r = super().cloak(img, kps, bbox, decoy_embs)
            r.image[:] = 7
            return r

    f = fake_factory_builder()
    orig = f

    def factory(cfg, device, manifest):
        c = orig(cfg, device, manifest)
        c.make_cloaker = Flat
        return c

    run(world, "raw", factory=factory)
    sw = pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")
    p = sw[sw["attack"] == "pgd_r50"]
    assert len(p) == 16 and (p["source_detected"] == 0).all() and p["id_sim_swap_glintr100"].isna().all()
    assert p["id_sim_direct_glintr100"].notna().all()
    assert (sw[sw["attack"] == "clean"]["source_detected"] == 1).all()


def test_rerun_is_noop_and_does_not_build_models(world):
    FakeCloaker.calls = 0
    run(world, "raw")
    n_calls = FakeCloaker.calls
    assert n_calls == 2  # two sources, one pgd attack
    before = (world.tmp / "raw" / "tiny_swap.csv").read_bytes()
    s, f = run(world, "raw")
    assert f.made["n"] == 0 and s["swaps_written"] == 0 and FakeCloaker.calls == n_calls
    assert (world.tmp / "raw" / "tiny_swap.csv").read_bytes() == before


def test_shards_merge_equals_unsharded(world):
    run(world, "full")
    run(world, "sh0", shard="0/2")
    run(world, "sh1", shard="1/2")
    rep = ex.merge_into([world.tmp / "sh0", world.tmp / "sh1"], world.tmp / "merged", None)
    assert sum(rep.duplicates.values()) == 0
    full = pd.read_csv(world.tmp / "full" / "tiny_swap.csv")
    merged = pd.read_csv(world.tmp / "merged" / "tiny_swap.csv")
    assert len(full) == len(merged) and not merged.duplicated(ex.SWAP_KEY).any()
    cols = ex.SWAP_KEY + ["id_sim_swap_glintr100", "id_sim_direct_glintr100", "source_detected"]
    a = full[cols].sort_values(ex.SWAP_KEY).reset_index(drop=True)
    b = merged[cols].sort_values(ex.SWAP_KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert len(pd.read_csv(world.tmp / "merged" / "tiny_cloak.csv")) == 6
    # merging the same data twice reports (and drops) duplicates
    rep2 = ex.merge_into([world.tmp / "full", world.tmp / "merged"], world.tmp / "m2", None)
    assert rep2.duplicates["tiny_swap.csv"] == len(full)
    assert len(pd.read_csv(world.tmp / "m2" / "tiny_swap.csv")) == len(full)


def test_truncated_line_is_redone_and_resume_from_zip(world):
    run(world, "raw")
    p = world.tmp / "raw" / "tiny_swap.csv"
    data = p.read_bytes()
    n_full = len(pd.read_csv(p))
    p.write_bytes(data[:-15])  # cut the last row mid-line
    # resume from a zip that only holds the truncated CSV + cache: fresh out dir
    z = world.tmp / "res.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(p, "raw/tiny_swap.csv")
        zf.write(world.tmp / "raw" / "tiny_cloak.csv", "raw/tiny_cloak.csv")
        for png in (world.tmp / "cache" / "cloaked").rglob("*.png"):
            zf.write(png, "cache/cloaked/" + str(png.relative_to(world.tmp / "cache" / "cloaked")))
    FakeCloaker.calls = 0
    s, _ = run(world, "raw2", resume_from=z)
    assert s["swaps_written"] == 1 and FakeCloaker.calls == 0  # only the cut row; no PGD rerun
    out = pd.read_csv(world.tmp / "raw2" / "tiny_swap.csv")
    assert len(out) == n_full and not out.duplicated(ex.SWAP_KEY).any()
    # truncation in place is also handled by ResumableCSV
    p.write_bytes(data[:-15])
    s, _ = run(world, "raw")
    assert s["swaps_written"] == 1 and len(pd.read_csv(p)) == n_full


def test_remaining_work_lists_exactly_missing(world):
    run(world, "raw")
    swap_p = world.tmp / "raw" / "tiny_swap.csv"
    cloak_p = world.tmp / "raw" / "tiny_cloak.csv"
    sw = pd.read_csv(swap_p, dtype=str)
    drop = sw[(sw["attack"] == "noise") & (sw["src_id"] == "s0001")]
    sw[~sw.index.isin(drop.index)].to_csv(swap_p, index=False)
    cl = pd.read_csv(cloak_p, dtype=str)
    gone = cl[(cl["attack"] == "pgd_r50") & (cl["src_id"] == "s0000")]
    cl[~cl.index.isin(gone.index)].to_csv(cloak_p, index=False)
    r = ex.remaining_work(world.cfg, world.man, [world.tmp / "raw"], 2)
    assert r["missing_cloaks"] == [("pgd_r50", "s0000")]
    assert sorted(r["missing_swaps"]) == sorted(map(tuple, drop[ex.SWAP_KEY].to_numpy().tolist()))
    assert r["shards"]["1/2"]["swaps"] == len(drop) and r["shards"]["0/2"]["cloaks"] == 1
    # complete results -> nothing missing
    run(world, "raw")
    r = ex.remaining_work(world.cfg, world.man, [world.tmp / "raw"])
    assert not r["missing_cloaks"] and not r["missing_swaps"]


def test_leakage_guard(world):
    bad = dict(world.raw)
    bad["judges"] = ["glintr100", "w600k_r50"]
    bad["whitebox_judges"] = []
    with pytest.raises(ValueError, match="leakage"):
        ex.load_config(bad)
    bad2 = json.loads(json.dumps(world.raw))
    bad2["attacks"]["pgd_r50"]["models"] = {"glintr100": 1.0}
    with pytest.raises(ValueError, match="evaluation-only"):
        ex.load_config(bad2)
    bad3 = dict(world.raw)
    bad3["transforms"] = ["jpeg75"]
    with pytest.raises(ValueError, match="none"):
        ex.load_config(bad3)


def test_make_figures_and_web_export(world, tmp_path):
    from facecloak import figures

    run(world, "raw")
    figs, tabs = tmp_path / "figs", tmp_path / "tabs"
    T = figures.make_all(world.tmp / "raw", figs, tabs)
    assert {"main", "robustness", "transfer"} <= set(T)
    assert (tabs / "main.tex").read_text().count(r"\toprule") == 1
    assert (tabs / "main.csv").exists() and (figs / "main_psr.png").exists() and (figs / "robustness.pdf").exists()
    assert (figs / "id_sim_distributions.png").exists()
    main = T["main"]
    assert len(main) == 6  # 3 attacks x 2 swappers
    # white-box judge column is labelled
    assert "white-box" in (tabs / "main.tex").read_text()
    # PSR matches a direct recomputation from the raw CSV
    sw = pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")
    g = sw[(sw["attack"] == "noise") & (sw["swapper"] == "inswapper") & (sw["transform"] == "none")]
    r = main[(main["attack"] == "noise") & (main["swapper"] == "inswapper")].iloc[0]
    assert r["psr"] == pytest.approx(g["protected_glintr100"].mean())
    # absent ablation data is skipped, not invented
    assert "ablation_eps" not in T
    import subprocess
    import sys

    web = tmp_path / "web"
    res = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "scripts" / "export_web_results.py"),
         "--raw", str(world.tmp / "raw"), "--figs", str(figs), "--web", str(web)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    summ = json.loads((web / "summary.json").read_text())
    assert "main" in summ["tables"] and "main_psr.png" in summ["figures"] and (web / "main_psr.png").exists()
