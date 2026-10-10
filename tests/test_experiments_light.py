"""Light tests of the experiment pipeline with fake models (no weights, no PGD)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pandas as pd
import pytest

from facecloak import experiment as ex
from facecloak.cloak import CloakResult
from facecloak.swap import NoFaceError, SwapResult

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
    calls: ClassVar[list] = []  # (swapper name, source_hint, target_hint) of every swap
    no_face_targets: ClassVar[set] = set()  # target image pixel sums that raise NoFaceError

    def __init__(self, name):
        self.name = name

    def swap(self, src, tgt, source_hint=None, target_hint=None):
        type(self).calls.append((self.name, source_hint, target_hint))
        if int(tgt.astype(np.int64).sum()) in type(self).no_face_targets:
            raise NoFaceError("no face in target")
        if src.std() < 1.0:  # flat image = "undetectable" source
            return SwapResult(None, False, 0.0, None)
        out = np.clip(0.5 * tgt.astype(np.float32) + 0.5 * src.astype(np.float32), 0, 255).astype(np.uint8)
        return SwapResult(out, True, 0.9, np.ones(4, np.float32), np.array(BBOX, np.float32))


class FakeCloaker:
    calls = 0

    def __init__(self, cc):
        self.cc = cc

    def cloak(self, img, kps, bbox=None, decoy_embs=None):
        type(self).calls += 1
        e = int(self.cc.eps)
        out = np.clip(img.astype(np.int16) + e, 0, 255).astype(np.uint8)
        return CloakResult(out, e, [], {"w600k_r50": 0.1}, 2, 0.01)


DECOY_CALLS: list = []


def decoy_fn(img, kps, models, exclude):
    DECOY_CALLS.append(list(exclude))
    return "t0", {}


def fake_factory_builder():
    made = {"n": 0}

    def factory(cfg, device, manifest):
        made["n"] += 1
        judges = {j: FakeJudge() for j in cfg.judges}
        sw = {s: FakeSwapper(s) for s in cfg.swappers}
        return ex.Components(
            device="cpu", judges=judges, swappers=sw, make_cloaker=FakeCloaker,
            decoy_fn=decoy_fn, lpips=lambda a, b: 0.05, detect=lambda img: 0.9,
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
        "attacks": {"clean": {"kind": "clean"}, "noise": {"kind": "noise", "eps": 8},
                    "pgd_r50": {"kind": "pgd", "eps": 8, "steps": 2, "models": {"w600k_r50": 1.0}}},
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
    assert {"main", "robustness_inswapper", "robustness_simswap", "transfer"} <= set(T)
    assert (tabs / "main.tex").read_text().count(r"\toprule") == 1
    assert (tabs / "main.csv").exists() and (figs / "main_psr.png").exists()
    for sw in ("inswapper", "simswap"):  # one robustness / distribution figure per swapper
        assert (figs / f"robustness_{sw}.pdf").exists() and (figs / f"id_sim_distributions_{sw}.png").exists()
        assert (tabs / f"robustness_{sw}.tex").exists()
    main = T["main"]
    assert len(main) == 6  # 3 attacks x 2 swappers
    # white-box judge column is labelled
    tex = (tabs / "main.tex").read_text()
    assert "white-box" in tex and "inswapper (white-box encoder)" in tex and "SimSwap (black-box)" in tex
    for col in ("n\\_valid", "PSR fail=prot.", "Src not det.", "Swap face not found"):
        assert col in tex
    rob = (tabs / "robustness_inswapper.tex").read_text()
    assert "n\\_valid" in rob and "PSR fail=prot." in rob and "Src not det." in rob
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


# ----------------------------------------------------------------------------- review-fix tests
def test_swappers_get_manifest_hints_and_target_bbox_is_used(world):
    FakeSwapper.calls = []
    run(world, "raw")
    assert FakeSwapper.calls
    for _, sh, th in FakeSwapper.calls:
        np.testing.assert_allclose(sh, BBOX)
        np.testing.assert_allclose(th, BBOX)
    # locate_swapped_face works around the bbox reported by the swapper, even if the manifest bbox is off
    judges = {"glintr100": FakeJudge()}
    ref = {"glintr100": FakeJudge().embed(np.zeros((H, W, 3), np.uint8), None)}
    src = np.full((H, W, 3), 100, np.uint8)
    src[::2] = 150
    sc = ex.swap_and_score_ex(FakeSwapper("inswapper"), judges, src, src, [100, 100, 120, 120], ref, BBOX)
    assert sc.target_found and sc.face_found  # manifest bbox (far away) ignored in favour of target_bbox


def test_target_without_face_is_not_a_source_failure(world):
    FakeSwapper.no_face_targets = {read_sum(world, "T1/a.png")}
    try:
        run(world, "raw")
    finally:
        FakeSwapper.no_face_targets = set()
    sw = pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")
    bad = sw[sw["tgt_id"] == "t0001"]
    assert len(bad) and (bad["target_detected"] == 0).all() and bad["source_detected"].isna().all()
    assert (bad["swap_face_found"] == 0).all()
    ok = sw[sw["tgt_id"] == "t0000"]
    assert (ok["target_detected"] == 1).all() and ok["source_detected"].notna().all()
    from facecloak import figures

    st = figures.psr_stats(bad, "glintr100")
    assert st["src_not_detected"] == 0.0 and st["swap_face_not_found"] == 1.0


def read_sum(world, rel):
    return int(cv2.imread(str(world.tmp / "lfw" / "lfw" / rel)).astype(np.int64).sum())


def test_config_hash_invalidates_rows_and_png_and_sidecar_resumes(world):
    import warnings

    run(world, "raw")
    cl = pd.read_csv(world.tmp / "raw" / "tiny_cloak.csv")
    h = ex.attack_hash(world.cfg, "pgd_r50", world.man)
    assert len(h) == 8 and set(cl.loc[cl.attack == "pgd_r50", "config_hash"]) == {h}
    png = world.tmp / "cache" / "cloaked" / "tiny" / "pgd_r50" / "s0000.png"
    side = json.loads(png.with_suffix(".json").read_text())
    assert side["config_hash"] == h and side["cos"] == {"w600k_r50": 0.1} and side["decoy_id"] == ""
    assert not list(png.parent.glob("*.tmp*"))  # atomic writes leave no temp files

    # killed between PNG and CSV row: the row is rebuilt from the sidecar, PGD is not re-run
    keep = cl[~((cl.attack == "pgd_r50") & (cl.src_id == "s0000"))]
    keep.to_csv(world.tmp / "raw" / "tiny_cloak.csv", index=False)
    sw = pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")
    FakeCloaker.calls = 0
    run(world, "raw")
    assert FakeCloaker.calls == 0
    again = pd.read_csv(world.tmp / "raw" / "tiny_cloak.csv")
    r = again[(again.attack == "pgd_r50") & (again.src_id == "s0000")].iloc[0]
    assert r["final_cos_w600k_r50"] == pytest.approx(0.1)
    assert len(pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")) == len(sw)

    # change the attack config -> new hash -> stale rows/PNGs are not reused
    raw2 = json.loads(json.dumps(world.raw))
    raw2["attacks"]["pgd_r50"]["eps"] = 4
    cfg2 = ex.load_config(raw2)
    assert ex.attack_hash(cfg2, "pgd_r50", world.man) != h
    assert ex.attack_hash(cfg2, "noise", world.man) == ex.attack_hash(world.cfg, "noise", world.man)
    world2 = SimpleNamespace(cfg=cfg2, man=world.man, tmp=world.tmp, raw=raw2)
    FakeCloaker.calls = 0
    with pytest.warns(UserWarning, match="config_hash"):
        s, _ = run(world2, "raw")
    assert FakeCloaker.calls == 2 and s["cloaks_computed"] == 2
    cl2 = pd.read_csv(world.tmp / "raw" / "tiny_cloak.csv")
    assert set(cl2.loc[cl2.attack == "pgd_r50", "config_hash"]) == {ex.attack_hash(cfg2, "pgd_r50", world.man)}
    assert list((world.tmp / "raw").glob("tiny_cloak.csv.stale-*")) and list((world.tmp / "raw").glob("tiny_swap.csv.stale-*"))
    assert cl2.loc[cl2.attack == "pgd_r50", "linf"].eq(4).all()
    # untouched attacks keep their rows
    assert len(cl2) == 6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s, _ = run(world2, "raw")  # now a clean no-op
    assert s["swaps_written"] == 0


def test_legacy_schema_is_set_aside_not_mixed(world):
    out = world.tmp / "raw"
    out.mkdir()
    (out / "tiny_swap.csv").write_text("attack,src_id,tgt_id,transform,swapper\nclean,s0000,t0000,none,inswapper\n")
    with pytest.warns(UserWarning, match="schema"):
        run(world, "raw")
    assert list(out.glob("tiny_swap.csv.legacy-*"))
    assert list(pd.read_csv(out / "tiny_swap.csv").columns) == ex.swap_columns(["glintr100", "w600k_r50"])


def test_decoy_excludes_paired_targets(world):
    raw = json.loads(json.dumps(world.raw))
    raw["attacks"]["pgd_tgt"] = {"kind": "pgd", "eps": 8, "steps": 2, "models": {"w600k_r50": 1.0}, "mode": "both"}
    w = SimpleNamespace(cfg=ex.load_config(raw), man=world.man, tmp=world.tmp, raw=raw)
    DECOY_CALLS.clear()
    run(w, "raw")
    assert len(DECOY_CALLS) == 2 and all(sorted(c) == ["t0000", "t0001"] for c in DECOY_CALLS)
    cl = pd.read_csv(world.tmp / "raw" / "tiny_cloak.csv")
    assert set(cl.loc[cl.attack == "pgd_tgt", "decoy_id"]) == {"t0"}  # decoy_id recorded


def test_attack_kind_must_be_explicit(world):
    import warnings

    raw = json.loads(json.dumps(world.raw))
    raw["attacks"]["mystery"] = {"eps": 8, "steps": 2, "models": {"w600k_r50": 1.0}}
    with pytest.raises(ValueError, match="kind"):
        ex.load_config(raw)
    raw["attacks"]["mystery"] = {"kind": "pgd", "eps": 8}
    with pytest.raises(ValueError, match="models"):
        ex.load_config(raw)
    raw["attacks"]["mystery"] = {"kind": "wat"}
    with pytest.raises(ValueError, match="bad kind"):
        ex.load_config(raw)
    legacy = json.loads(json.dumps(world.raw))
    legacy["attacks"]["clean"] = {}
    with pytest.warns(DeprecationWarning, match="kind"):
        cfg = ex.load_config(legacy)
    assert cfg.attack_kind("clean") == "clean"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ex.load_config(world.raw)  # explicit kinds: no warning


def test_quick_calibration_refused_unless_allowed(world, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RESULTS_DIR", tmp_path)
    cal = {"glintr100": {"tau_far1": 0.3}, "w600k_r50": {"tau_far1": 0.4}}
    (tmp_path / "calibration_quick.json").write_text(json.dumps(cal))
    with pytest.raises(FileNotFoundError, match="calibration"):
        ex.load_tau(world.cfg)
    explicit = ex.load_config({**world.raw, "calibration": str(tmp_path / "calibration_quick.json")})
    with pytest.raises(ValueError, match="quick"):
        ex.load_tau(explicit)
    smoke = ex.load_config({**world.raw, "allow_quick_calibration": True})
    assert ex.load_tau(smoke) == ({"glintr100": 0.3, "w600k_r50": 0.4}, "calibration_quick.json")
    (tmp_path / "calibration.json").write_text(json.dumps({"glintr100": {"tau_far1": 0.5}, "w600k_r50": {"tau_far1": 0.6}}))
    assert ex.load_tau(world.cfg)[1] == "calibration.json"
    assert ex.load_tau(smoke)[1] == "calibration.json"  # full calibration always preferred


def test_shipped_configs(world):
    root = Path(__file__).parent.parent / "configs"
    for f in ("main", "ablation_eps", "ablation_steps", "smoke"):
        cfg = ex.load_config(root / f"{f}.yaml")
        assert cfg.allow_quick_calibration == (f == "smoke")
        if f != "smoke":
            assert cfg.swappers == ["inswapper", "simswap"]


def test_figures_recompute_protected_with_one_tau(world, tmp_path):
    from facecloak import figures

    run(world, "raw")
    p = world.tmp / "raw" / "tiny_swap.csv"
    sw = pd.read_csv(p)
    sw["protected_glintr100"] = 1.0  # corrupt the stored flag
    sw.to_csv(p, index=False)
    R = figures.load_raw(world.tmp / "raw")
    exp = np.where(R.swap["id_sim_swap_glintr100"].isna(), np.nan, R.swap["id_sim_swap_glintr100"] < TAU["glintr100"])
    np.testing.assert_array_equal(R.swap["protected_glintr100"].to_numpy(), exp)
    assert R.tau("tiny", "glintr100") == TAU["glintr100"]

    # runs that disagree on tau: warn and fall back to calibration.json
    mp = world.tmp / "raw" / "tiny_meta.json"
    meta = json.loads(mp.read_text())
    meta["runs"].append({**meta["runs"][-1], "run_id": "x", "started": "z", "tau": {"glintr100": 0.9, "w600k_r50": 0.2}})
    mp.write_text(json.dumps(meta))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"glintr100": {"tau_far1": 0.25}}))
    with pytest.warns(UserWarning, match="different thresholds"):
        R = figures.load_raw(world.tmp / "raw", cal)
    assert R.tau_used == {"glintr100": 0.25, "w600k_r50": 0.2}


def test_cluster_bootstrap_and_wilson_and_paired(world):
    from facecloak import figures
    from facecloak.metrics import bootstrap_ci, cluster_bootstrap_ci, wilson_interval

    # 20 sources x 5 rows, rows within a source identical: iid bootstrap is overconfident
    rng = np.random.default_rng(0)
    src = np.repeat(np.arange(20), 5)
    val = np.repeat((rng.random(20) < 0.5).astype(float), 5)
    m, lo, hi = cluster_bootstrap_ci(val, src)
    m2, lo2, hi2 = bootstrap_ci(val)
    assert m == m2 and (hi - lo) > (hi2 - lo2)
    assert np.isnan(cluster_bootstrap_ci([np.nan], [0])[0])
    lo_w, hi_w = wilson_interval(0, 50)
    assert lo_w == 0.0 and 0.05 < hi_w < 0.1
    assert wilson_interval(50, 50)[1] == 1.0

    # all-protected -> degenerate cluster CI -> Wilson fallback, flagged
    df = pd.DataFrame({"src_id": np.repeat(["a", "b", "c"], 4), "protected_j": 1.0, "id_sim_swap_j": 0.0,
                       "source_detected": 1, "swap_face_found": 1})
    st = figures.psr_stats(df, "j")
    assert st["psr"] == 1.0 and st["psr_ci"] == "wilson" and st["psr_lo"] < 1.0 and st["psr_hi"] == 1.0
    df["protected_j"] = [1.0] * 4 + [0.0] * 4 + [1.0, 0.0, 1.0, 0.0]
    assert figures.psr_stats(df, "j")["psr_ci"] == "cluster-bootstrap"

    # main table carries paired-vs-clean columns
    run(world, "raw")
    T = figures.build_tables(figures.load_raw(world.tmp / "raw"))
    main = T["main"]
    for c in ("psr_ci", "d_id_sim", "d_id_sim_lo", "d_id_sim_hi", "n_paired_src", "psr_failasprot",
              "src_not_detected", "swap_face_not_found", "n", "n_valid"):
        assert c in main.columns
    clean = main[main.attack == "clean"].iloc[0]
    assert np.isnan(clean["d_id_sim"])
    noise = main[(main.attack == "noise") & (main.swapper == "inswapper")].iloc[0]
    sw = pd.read_csv(world.tmp / "raw" / "tiny_swap.csv")
    b = sw[(sw["transform"] == "none") & (sw["swapper"] == "inswapper")]
    per = b.groupby(["src_id", "attack"])["id_sim_swap_glintr100"].mean().unstack("attack")
    assert noise["d_id_sim"] == pytest.approx((per["noise"] - per["clean"]).mean())
    assert noise["n_paired_src"] == 2
