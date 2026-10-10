"""Tables and figures from ``results/raw`` (shared by make_figures.py and export_web_results.py).

Numbers only ever come from the CSVs. A table / figure whose data is absent is skipped with a printed
notice. PSR (protection success rate) = fraction of valid swaps whose identity similarity to the
reference photo falls below the judge's calibrated threshold; rows where the swap failed (no source face /
no swapped face found) are excluded from PSR and counted separately (``src_not_detected``,
``swap_face_not_found``; ``psr_failasprot`` counts every failure as "protected").

Statistics: rows sharing a ``src_id`` are dependent, so CIs are percentile CIs from a CLUSTER bootstrap
over source ids; where that interval is degenerate (width 0, e.g. PSR exactly 0 or 1) a Wilson score
interval over the valid rows is reported instead (``psr_ci == "wilson"``, marked ``*`` in the tables).
``d_id_sim`` is the paired per-source mean id-sim difference attack - clean (negative = protective).

A single threshold per judge is used: ``protected_<j>`` is RECOMPUTED here from ``id_sim_swap_<j>`` with
one tau (all run metas must agree; otherwise results/calibration.json wins and a warning is printed).

Swapper labelling: inswapper's identity encoder is buffalo_l w600k_r50, which is also an attack model, so
its rows are white-box on the encoder; SimSwap (its own ArcFace) is the black-box transfer test.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from facecloak.metrics import bootstrap_ci, cluster_bootstrap_ci, wilson_interval
from facecloak.paths import RESULTS_DIR
from facecloak.transforms import ROBUSTNESS

SWAPPER_LABELS = {"inswapper": "inswapper (white-box encoder)", "simswap": "SimSwap (black-box)"}

COL_W = 3.5  # IEEE single column, inches
FULL_W = 7.16  # IEEE double column
ATTACK_ORDER = ["clean", "noise", "pgd_r50", "pgd_ens", "pgd_ens_targeted", "pgd_ens_eot",
                "pgd_ens_eot_lpips", "pgd_ens_det"]
LABELS = {
    "clean": "Clean", "noise": "Random noise", "pgd_r50": "PGD (R50)", "pgd_ens": "PGD (ens.)",
    "pgd_ens_targeted": "PGD ens. + decoy", "pgd_ens_eot": "PGD ens. + EOT",
    "pgd_ens_eot_lpips": "PGD ens. + EOT + LPIPS", "pgd_ens_det": "PGD ens. + det.",
}
COLORS = {
    "clean": "#7f7f7f", "noise": "#bcbd22", "pgd_r50": "#1f77b4", "pgd_ens": "#2ca02c",
    "pgd_ens_targeted": "#9467bd", "pgd_ens_eot": "#ff7f0e", "pgd_ens_eot_lpips": "#d62728",
    "pgd_ens_det": "#17becf",
}
_FALLBACK = ["#8c564b", "#e377c2", "#393b79", "#637939", "#843c39", "#7b4173"]
HEADLINE = ["pgd_ens_eot_lpips", "pgd_ens_eot", "pgd_ens", "pgd_ens_targeted", "pgd_r50"]


def label(a: str) -> str:
    return LABELS.get(a, a)


def sw_label(sw: str) -> str:
    return SWAPPER_LABELS.get(sw, sw)


def color(a: str) -> str:
    return COLORS.get(a) or _FALLBACK[sum(map(ord, a)) % len(_FALLBACK)]


def set_style() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
    })


def save_fig(fig: plt.Figure, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    print(f"wrote {out / name}.pdf/.png")
    return paths


# ----------------------------------------------------------------------------- loading
@dataclass
class Raw:
    swap: pd.DataFrame
    cloak: pd.DataFrame
    metas: dict[str, dict[str, Any]] = field(default_factory=dict)
    tau_used: dict[str, float] = field(default_factory=dict)
    tau_source: str = ""

    @property
    def configs(self) -> list[str]:
        return sorted(self.swap["config"].unique()) if len(self.swap) else []

    def judges(self, df: pd.DataFrame | None = None) -> list[str]:
        d = self.swap if df is None else df
        return [c[len("id_sim_swap_"):] for c in d.columns if c.startswith("id_sim_swap_")]

    def whitebox(self) -> set[str]:
        out: set[str] = set()
        for m in self.metas.values():
            out |= set(m.get("config", {}).get("whitebox_judges", []))
        return out

    def primary_judge(self, judges: list[str] | None = None) -> str | None:
        js = judges if judges is not None else self.judges()
        if not js:
            return None
        wb = self.whitebox()
        pref = [j for j in js if j == "glintr100"] or [j for j in js if j not in wb] or js
        return pref[0]

    def tau(self, cfg: str, judge: str) -> float | None:
        """The single threshold used for ``judge`` (``cfg`` kept for API compatibility)."""
        if judge in self.tau_used:
            return self.tau_used[judge]
        runs = self.metas.get(cfg, {}).get("runs", [])
        for r in reversed(runs):
            if judge in r.get("tau", {}):
                return float(r["tau"][judge])
        return None


def resolve_tau(metas: dict[str, dict[str, Any]], calibration: Path | None = None) -> tuple[dict[str, float], str]:
    """ONE tau per judge over all runs of all configs.

    All ``runs[*].tau`` must agree; if they do not, warn and use ``calibration`` (default
    results/calibration.json) for the judges involved, else the newest run's value.
    """
    seen: dict[str, list[tuple[str, float]]] = {}
    for m in metas.values():
        for r in m.get("runs", []):
            for j, t in (r.get("tau") or {}).items():
                seen.setdefault(j, []).append((str(r.get("started", "")), float(t)))
    out: dict[str, float] = {}
    sources: set[str] = set()
    cal_path = Path(calibration) if calibration else RESULTS_DIR / "calibration.json"
    for j, items in seen.items():
        vals = {round(t, 9) for _, t in items}
        if len(vals) == 1:
            out[j] = items[0][1]
            sources.add("run metas")
            continue
        cal = None
        if cal_path.exists():
            try:
                cal = float(json.loads(cal_path.read_text())[j]["tau_far1"])
            except (KeyError, ValueError, json.JSONDecodeError):
                cal = None
        if cal is not None:
            out[j] = cal
            sources.add(cal_path.name)
        else:
            out[j] = max(items)[1]
            sources.add("newest run (INCONSISTENT)")
        warnings.warn(
            f"judge {j}: runs used different thresholds {sorted(vals)}; using {out[j]:.6f} from "
            f"{'calibration file' if cal is not None else 'the newest run'}"
        )
    return out, ", ".join(sorted(sources))


def apply_tau(swap: pd.DataFrame, tau: dict[str, float]) -> None:
    """Recompute ``protected_<j>`` = id_sim_swap_<j> < tau[j] (NaN where the swap failed), in place."""
    for j, t in tau.items():
        col = f"id_sim_swap_{j}"
        if col in swap.columns:
            sims = swap[col].to_numpy(dtype=float)
            swap[f"protected_{j}"] = np.where(np.isnan(sims), np.nan, (sims < t).astype(float))


def load_raw(raw_dir: Path, calibration: Path | None = None) -> Raw:
    raw_dir = Path(raw_dir)
    swaps, cloaks = [], []
    for p in sorted(raw_dir.glob("*_swap.csv")):
        d = pd.read_csv(p)
        d["config"] = p.name[: -len("_swap.csv")]
        swaps.append(d)
    for p in sorted(raw_dir.glob("*_cloak.csv")):
        d = pd.read_csv(p)
        d["config"] = p.name[: -len("_cloak.csv")]
        cloaks.append(d)
    metas = {}
    for p in sorted(raw_dir.glob("*_meta.json")):
        try:
            metas[p.name[: -len("_meta.json")]] = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"notice: unreadable {p.name}, ignored")
    swap = pd.concat(swaps, ignore_index=True) if swaps else pd.DataFrame(columns=["config"])
    cloak = pd.concat(cloaks, ignore_index=True) if cloaks else pd.DataFrame(columns=["config"])
    tau, tau_src = resolve_tau(metas, calibration)
    if len(swap):
        apply_tau(swap, tau)
    for c in cloak["config"].unique() if "config_hash" in cloak.columns else []:
        per = cloak[cloak["config"] == c].groupby("attack")["config_hash"].nunique()
        for a in per[per > 1].index:
            warnings.warn(f"{c}/{a}: rows from {per[a]} different config hashes were merged")
    return Raw(swap, cloak, metas, tau, tau_src)


def pick_main(R: Raw) -> str | None:
    cfgs = R.configs
    if "main" in cfgs:
        return "main"
    rest = [c for c in cfgs if not c.startswith("ablation")]
    # prefer the config with the most rows
    return max(rest, key=lambda c: (R.swap["config"] == c).sum()) if rest else None


def order_attacks(attacks: list[str]) -> list[str]:
    known = [a for a in ATTACK_ORDER if a in attacks]
    return known + sorted(a for a in attacks if a not in known)


def order_transforms(ts: list[str]) -> list[str]:
    known = [t for t in ROBUSTNESS if t in ts]
    return known + sorted(t for t in ts if t not in known)


def swappers_in(df: pd.DataFrame) -> list[str]:
    """Swappers present, inswapper first (figures / tables are produced for EACH of them)."""
    sw = list(df["swapper"].unique()) if len(df) else []
    return sorted(sw, key=lambda x: (x != "inswapper", x))


# ----------------------------------------------------------------------------- statistics
def psr_stats(df: pd.DataFrame, judge: str) -> dict[str, Any]:
    n = len(df)
    prot = df[f"protected_{judge}"].to_numpy(dtype=float) if n else np.array([])
    ok = ~np.isnan(prot)
    valid = prot[ok]
    mean, lo, hi = (np.nan, np.nan, np.nan)
    ci = "none"
    if len(valid):
        mean, lo, hi = cluster_bootstrap_ci(valid, df["src_id"].to_numpy()[ok])
        ci = "cluster-bootstrap"
        if not (hi - lo > 0):  # degenerate (all valid rows equal): fall back to Wilson on the valid rows
            lo, hi = wilson_interval(float(valid.sum()), len(valid))
            ci = "wilson"
    sims = df[f"id_sim_swap_{judge}"].to_numpy(dtype=float) if n else np.array([])
    sd = df["source_detected"].to_numpy(dtype=float) if n else np.array([])
    sf = df["swap_face_found"].to_numpy(dtype=float) if n else np.array([])
    return {
        "n": n, "n_valid": len(valid), "psr": mean, "psr_lo": lo, "psr_hi": hi, "psr_ci": ci,
        "psr_failasprot": 1.0 - float(np.sum(valid == 0)) / n if n else np.nan,
        "mean_id_sim": float(np.nanmean(sims)) if np.any(~np.isnan(sims)) else np.nan,
        "det_fail": 1.0 - float(np.nanmean(sd)) if n and np.any(~np.isnan(sd)) else np.nan,
        # source not found by the attacker's detector (target-side failures are NOT counted here)
        "src_not_detected": float(np.sum(sd == 0)) / n if n else np.nan,
        # swapped face not located in the output (incl. rows where the target had no face)
        "swap_face_not_found": float(np.sum((sf == 0) & (sd != 0))) / n if n else np.nan,
    }


def paired_vs_clean(g: pd.DataFrame, g_clean: pd.DataFrame, judge: str) -> dict[str, float]:
    """Per-source mean id-sim(attack) - id-sim(clean), cluster (= source) bootstrap CI. Negative = protective."""
    col = f"id_sim_swap_{judge}"
    if not len(g) or not len(g_clean):
        return {"d_id_sim": np.nan, "d_id_sim_lo": np.nan, "d_id_sim_hi": np.nan, "n_paired_src": 0}
    d = (g.groupby("src_id")[col].mean() - g_clean.groupby("src_id")[col].mean()).dropna().to_numpy()
    m, lo, hi = bootstrap_ci(d)
    return {"d_id_sim": m, "d_id_sim_lo": lo, "d_id_sim_hi": hi, "n_paired_src": len(d)}


def _finite_mean(s: pd.Series) -> float:
    v = s.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def cloak_quality(R: Raw, cfg: str) -> pd.DataFrame:
    c = R.cloak[R.cloak["config"] == cfg]
    if not len(c):
        return pd.DataFrame(columns=["attack", "psnr", "ssim", "lpips", "cloak_seconds"])
    g = c.groupby("attack")
    return pd.DataFrame({
        "psnr": g["psnr"].apply(_finite_mean), "ssim": g["ssim"].apply(_finite_mean),
        "lpips": g["lpips"].apply(_finite_mean), "cloak_seconds": g["seconds"].apply(_finite_mean),
    }).reset_index()


def build_tables(R: Raw) -> dict[str, pd.DataFrame]:
    """All numeric tables (empty dict entries are omitted).

    Keys: ``main``, ``transfer``, and per swapper ``robustness_<sw>``, ``ablation_eps_<sw>``,
    ``ablation_steps_<sw>``.
    """
    out: dict[str, pd.DataFrame] = {}
    main = pick_main(R)
    if main is None:
        return out
    S = R.swap[R.swap["config"] == main]
    judges = R.judges(S)
    pj = R.primary_judge(judges)
    wb = R.whitebox()
    base = S[S["transform"] == "none"]

    # main table: attack x swapper
    rows = []
    q = cloak_quality(R, main).set_index("attack") if len(R.cloak) else pd.DataFrame()
    for (a, sw), g in base.groupby(["attack", "swapper"]):
        st = psr_stats(g, pj)
        clean = base[(base["attack"] == "clean") & (base["swapper"] == sw)]
        pair = paired_vs_clean(g, clean, pj) if a != "clean" else paired_vs_clean(g.iloc[:0], clean, pj)
        row = {"attack": a, "swapper": sw, "judge": pj, **st, **pair}
        for j in judges:
            if j != pj:
                s2 = psr_stats(g, j)
                row[f"psr_{j}"] = s2["psr"]
                row[f"psr_{j}_lo"], row[f"psr_{j}_hi"] = s2["psr_lo"], s2["psr_hi"]
                row[f"psr_{j}_ci"] = s2["psr_ci"]
                row[f"mean_id_sim_{j}"] = s2["mean_id_sim"]
        for k in ("psnr", "ssim", "lpips", "cloak_seconds"):
            row[k] = q.loc[a, k] if len(q) and a in q.index else np.nan
        row["whitebox_judges"] = ",".join(sorted(wb & set(judges)))
        rows.append(row)
    if rows:
        t = pd.DataFrame(rows)
        t["_o"] = t["attack"].map({a: i for i, a in enumerate(order_attacks(list(t["attack"].unique())))})
        t["_s"] = t["swapper"].map({x: i for i, x in enumerate(swappers_in(t))})
        out["main"] = t.sort_values(["_s", "_o"]).drop(columns=["_o", "_s"]).reset_index(drop=True)

    # robustness: attack x transform, one table per swapper
    for sw in swappers_in(S):
        R0 = S[S["swapper"] == sw]
        if R0["transform"].nunique() <= 1:
            continue
        rows = []
        for (a, tf), g in R0.groupby(["attack", "transform"]):
            rows.append({"attack": a, "transform": tf, "swapper": sw, "judge": pj, **psr_stats(g, pj)})
        t = pd.DataFrame(rows)
        t["_o"] = t["attack"].map({a: i for i, a in enumerate(order_attacks(list(t["attack"].unique())))})
        t["_t"] = t["transform"].map({x: i for i, x in enumerate(order_transforms(list(t["transform"].unique())))})
        out[f"robustness_{sw}"] = t.sort_values(["_o", "_t"]).drop(columns=["_o", "_t"]).reset_index(drop=True)

    # transfer: white-box-encoder swapper (inswapper) vs black-box swapper (SimSwap)
    if base["swapper"].nunique() > 1 and "main" in out:
        out["transfer"] = out["main"][["attack", "swapper", "judge", "n", "n_valid", "psr", "psr_lo", "psr_hi",
                                       "psr_ci", "mean_id_sim", "det_fail"]].copy()

    # ablations, one table per swapper
    for name, key, xcol in (("ablation_eps", "eps", "eps"), ("ablation_steps", "steps", "steps")):
        if name not in R.configs:
            continue
        A = R.swap[R.swap["config"] == name]
        acfg = R.metas.get(name, {}).get("config", {}).get("attacks", {})
        if not acfg:
            print(f"notice: {name}_meta.json missing; cannot map attacks to {key}, skipping {name}")
            continue
        aj = R.primary_judge(R.judges(A))
        q = cloak_quality(R, name).set_index("attack")
        for asw in swappers_in(A):
            rows = []
            for (a, tf), g in A[A["swapper"] == asw].groupby(["attack", "transform"]):
                if a not in acfg:
                    continue
                st = psr_stats(g, aj)
                rows.append({"attack": a, xcol: acfg[a][key], "transform": tf, "swapper": asw, "judge": aj, **st,
                             "lpips": q.loc[a, "lpips"] if a in q.index else np.nan,
                             "psnr": q.loc[a, "psnr"] if a in q.index else np.nan})
            if rows:
                out[f"{name}_{asw}"] = pd.DataFrame(rows).sort_values([xcol, "transform"]).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- table writers
def _tex_escape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_pct(x: float, lo: float | None = None, hi: float | None = None, ci: str | None = None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{100 * x:.1f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]" + ("*" if ci == "wilson" else "")
    return s


def fmt_num(x: float, d: int = 3) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def fmt_int(x: float) -> str:
    return "--" if x is None or not np.isfinite(x) else str(int(x))


CAPTION = (
    "PSR (valid) = share of valid swaps whose id-sim to the reference is below the judge's FAR=1% threshold, "
    "conditional on the swap succeeding (n_valid of n rows); PSR (fail=prot.) counts every failed swap as "
    "protected. 95% CI: percentile cluster bootstrap over source ids; * = degenerate interval (width 0, "
    "e.g. PSR 0 or 100%) replaced by the Wilson score interval over the n_valid rows (treats rows as "
    "independent, so it is optimistic). Src not det. = attacker found no face in the (cloaked) source; "
    "Swap face not found = swapped face not located in the output or target had no face. "
    "d id-sim = paired per-source mean id-sim(attack) - id-sim(clean) [cluster bootstrap CI]; negative = "
    "protective. inswapper (white-box encoder): its identity encoder is the attacked w600k_r50; "
    "SimSwap (black-box): own ArcFace, transfer test."
)


def write_tex(path: Path, header: list[str], rows: list[list[str]], align: str | None = None,
              caption: str | None = None) -> None:
    align = align or "ll" + "r" * (len(header) - 2)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(_tex_escape(h) for h in header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(_tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    if caption:
        lines.insert(0, f"% caption: {caption}")
        path.with_name(path.stem + ".caption.tex").write_text(_tex_escape(caption) + "\n")
    path.write_text("\n".join(lines))


def _count_cols(r: pd.Series) -> list[str]:
    return [fmt_int(r["n"]), fmt_int(r["n_valid"])]


def _fail_cols(r: pd.Series) -> list[str]:
    return [fmt_pct(r["psr_failasprot"]), fmt_pct(r["src_not_detected"]), fmt_pct(r["swap_face_not_found"])]


def write_tables(R: Raw, T: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wb = R.whitebox()
    for name, t in T.items():
        t.to_csv(out / f"{name}.csv", index=False)
    if "main" in T:
        t = T["main"]
        pj = t["judge"].iloc[0]
        others = [c[4:] for c in t.columns if c.startswith("psr_") and not c.endswith(("_lo", "_hi", "_ci", "_failasprot"))
                  and c not in ("psr",)]
        hdr = ["Attack", "Swapper", "n", "n_valid", f"PSR@tau {pj}, valid (%)", "PSR fail=prot. (%)"] + [
            f"PSR {j}{' (white-box)' if j in wb else ''} (%)" for j in others
        ] + [f"Mean id-sim {pj}", "d id-sim vs clean", "PSNR", "SSIM", "LPIPS", "Src not det. (%)",
             "Swap face not found (%)"]
        rows = []
        for _, r in t.iterrows():
            rows.append([label(r["attack"]), sw_label(r["swapper"])] + _count_cols(r)
                        + [fmt_pct(r["psr"], r["psr_lo"], r["psr_hi"], r["psr_ci"]), fmt_pct(r["psr_failasprot"])]
                        + [fmt_pct(r[f"psr_{j}"], r[f"psr_{j}_lo"], r[f"psr_{j}_hi"], r[f"psr_{j}_ci"]) for j in others]
                        + [fmt_num(r["mean_id_sim"]),
                           "--" if r["attack"] == "clean" else
                           f"{fmt_num(r['d_id_sim'])} [{fmt_num(r['d_id_sim_lo'])}, {fmt_num(r['d_id_sim_hi'])}]",
                           fmt_num(r["psnr"], 1), fmt_num(r["ssim"]), fmt_num(r["lpips"]),
                           fmt_pct(r["src_not_detected"]), fmt_pct(r["swap_face_not_found"])])
        write_tex(out / "main.tex", hdr, rows, caption=CAPTION)
    for name, t in T.items():
        if not name.startswith("robustness_"):
            continue
        sw = t["swapper"].iloc[0]
        hdr = ["Attack", "Transform", "n", "n_valid", f"PSR@tau {t['judge'].iloc[0]}, valid (%)", "PSR fail=prot. (%)",
               "Src not det. (%)", "Swap face not found (%)"]
        rows = []
        for a in order_attacks(list(t["attack"].unique())):
            g = t[t["attack"] == a].set_index("transform")
            for x in order_transforms(list(g.index)):
                r = g.loc[x]
                rows.append([label(a), x, fmt_int(r["n"]), fmt_int(r["n_valid"]),
                             fmt_pct(r["psr"], r["psr_lo"], r["psr_hi"], r["psr_ci"]), fmt_pct(r["psr_failasprot"]),
                             fmt_pct(r["src_not_detected"]), fmt_pct(r["swap_face_not_found"])])
        write_tex(out / f"{name}.tex", hdr, rows, "llrrrrrr", caption=f"{sw_label(sw)}. " + CAPTION)
        tfs = order_transforms(list(t["transform"].unique()))
        mrows = []
        for a in order_attacks(list(t["attack"].unique())):
            g = t[t["attack"] == a].set_index("transform")
            mrows.append([label(a)] + [fmt_pct(g.loc[x, "psr"]) if x in g.index else "--" for x in tfs])
        write_tex(out / f"{name}_matrix.tex", ["Attack"] + tfs, mrows, "l" + "r" * len(tfs),
                  caption=f"{sw_label(sw)}. PSR (valid, %) by transform; see robustness_{sw}.tex for n, CIs and failures.")
    if "transfer" in T:
        t = T["transfer"]
        sws = swappers_in(t)
        hdr = ["Attack"] + [f"PSR {sw_label(s)} (%)" for s in sws] + [f"Mean id-sim {sw_label(s)}" for s in sws]
        rows = []
        for a in order_attacks(list(t["attack"].unique())):
            g = t[t["attack"] == a].set_index("swapper")
            rows.append([label(a)]
                        + [fmt_pct(g.loc[s, "psr"], g.loc[s, "psr_lo"], g.loc[s, "psr_hi"], g.loc[s, "psr_ci"])
                           if s in g.index else "--" for s in sws]
                        + [fmt_num(g.loc[s, "mean_id_sim"]) if s in g.index else "--" for s in sws])
        write_tex(out / "transfer.tex", hdr, rows, "l" + "r" * (2 * len(sws)), caption=CAPTION)
    print(f"wrote tables {sorted(T)} to {out}")


# ----------------------------------------------------------------------------- figures
def fig_robustness(T: dict[str, pd.DataFrame], out: Path) -> None:
    keys = [k for k in T if k.startswith("robustness_")]
    if not keys:
        print("notice: no multi-transform data; skipping robustness figure")
        return
    for key in keys:
        t = T[key]
        sw = t["swapper"].iloc[0]
        attacks = order_attacks(list(t["attack"].unique()))
        tfs = order_transforms(list(t["transform"].unique()))
        fig, ax = plt.subplots(figsize=(FULL_W, 2.3))
        w = 0.8 / len(attacks)
        for i, a in enumerate(attacks):
            g = t[t["attack"] == a].set_index("transform")
            y = np.array([100 * g.loc[x, "psr"] if x in g.index else np.nan for x in tfs])
            lo = np.array([100 * g.loc[x, "psr_lo"] if x in g.index else np.nan for x in tfs])
            hi = np.array([100 * g.loc[x, "psr_hi"] if x in g.index else np.nan for x in tfs])
            ax.bar(np.arange(len(tfs)) + (i - (len(attacks) - 1) / 2) * w, y, w, color=color(a), label=label(a),
                   yerr=[np.clip(np.nan_to_num(y - lo), 0, None), np.clip(np.nan_to_num(hi - y), 0, None)], error_kw={"lw": 0.4, "capsize": 0.8})
        ax.set_xticks(range(len(tfs)), tfs, rotation=30, ha="right")
        ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}]")
        ax.set_title(sw_label(sw), loc="left")
        ax.set_ylim(0, 105)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
        save_fig(fig, out, f"robustness_{sw}")


def fig_ablation(T: dict[str, pd.DataFrame], out: Path) -> None:
    found = False
    for name, xcol, xl in (("ablation_eps", "eps", r"$\epsilon$ (/255)"), ("ablation_steps", "steps", "PGD steps")):
        for key in [k for k in T if k.startswith(name + "_")]:
            found = True
            t = T[key]
            sw = t["swapper"].iloc[0]
            fig, ax = plt.subplots(figsize=(COL_W, 2.2))
            for tf, mk in zip(order_transforms(list(t["transform"].unique())), "os^vD", strict=False):
                g = t[t["transform"] == tf].sort_values(xcol)
                ax.errorbar(g[xcol], 100 * g["psr"],
                            yerr=[np.clip(100 * (g["psr"] - g["psr_lo"]), 0, None), np.clip(100 * (g["psr_hi"] - g["psr"]), 0, None)],
                            marker=mk, ms=3, capsize=1.5, label=tf)
            ax.set_xlabel(xl)
            ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}]")
            ax.set_title(sw_label(sw), loc="left")
            ax.set_ylim(0, 105)
            ax.legend(title="transform", frameon=False)
            save_fig(fig, out, key)
            if name == "ablation_eps" and t["lpips"].notna().any():
                fig, ax = plt.subplots(figsize=(COL_W, 2.2))
                for tf in order_transforms(list(t["transform"].unique())):
                    g = t[t["transform"] == tf].sort_values("lpips")
                    ax.plot(g["lpips"], 100 * g["psr"], marker="o", ms=3, label=tf)
                    if tf == "none":
                        for _, r in g.iterrows():
                            ax.annotate(f"{r['eps']:g}", (r["lpips"], 100 * r["psr"]), textcoords="offset points",
                                        xytext=(3, -8), fontsize=5)
                ax.set_xlabel("LPIPS (lower = less visible)")
                ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}]")
                ax.set_title(sw_label(sw), loc="left")
                ax.set_ylim(0, 105)
                ax.legend(title="transform", frameon=False)
                save_fig(fig, out, f"ablation_psr_vs_lpips_{sw}")
    if not found:
        print("notice: no ablation data; skipping")


def fig_distributions(R: Raw, out: Path) -> None:
    main = pick_main(R)
    if main is None:
        print("notice: no data; skipping id-sim distribution figure")
        return
    S0 = R.swap[(R.swap["config"] == main) & (R.swap["transform"] == "none")]
    pj = R.primary_judge(R.judges(S0))
    for sw in swappers_in(S0):
        _fig_distribution_one(R, S0[S0["swapper"] == sw], sw, pj, main, out)


def _fig_distribution_one(R: Raw, S: pd.DataFrame, sw: str, pj: str, main: str, out: Path) -> None:
    present = set(S["attack"])
    groups = [a for a in ("clean", "noise") if a in present]
    if "pgd_r50" in present:
        groups.append("pgd_r50")
    head = next((a for a in HEADLINE if a in present and a not in groups), None)
    if head:
        groups.append(head)
    data = [(a, S.loc[S["attack"] == a, f"id_sim_swap_{pj}"].dropna().to_numpy()) for a in groups]
    data = [(a, d) for a, d in data if len(d)]
    if len(data) < 2:
        print(f"notice: need >= 2 attack groups with valid swaps; skipping id-sim distribution figure ({sw})")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    rng = np.random.default_rng(0)
    for i, (a, d) in enumerate(data):
        if len(d) >= 5 and np.std(d) > 0:
            vp = ax.violinplot(d, positions=[i], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color(a))
                b.set_alpha(0.5)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=2, color=color(a), alpha=0.6, lw=0)
        ax.hlines(np.mean(d), i - 0.3, i + 0.3, color="k", lw=1)
    tau = R.tau(main, pj)
    if tau is not None:
        ax.axhline(tau, color="k", ls="--", lw=0.7)
        ax.text(len(data) - 0.5, tau, r" $\tau$", va="bottom", ha="right", fontsize=6)
    ax.set_xticks(range(len(data)), [label(a) for a, _ in data], rotation=20, ha="right")
    ax.set_ylabel(f"id-sim(swap, ref) [{pj}]")
    ax.set_title(sw_label(sw), loc="left")
    ax.grid(axis="x", visible=False)
    save_fig(fig, out, f"id_sim_distributions_{sw}")


def fig_main_bar(T: dict[str, pd.DataFrame], out: Path) -> None:
    if "main" not in T:
        return
    t = T["main"]
    sws = swappers_in(t)
    attacks = order_attacks(list(t["attack"].unique()))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    w = 0.8 / len(sws)
    for k, s in enumerate(sws):
        g = t[t["swapper"] == s].set_index("attack")
        y = np.array([100 * g.loc[a, "psr"] if a in g.index else np.nan for a in attacks])
        lo = np.array([100 * g.loc[a, "psr_lo"] if a in g.index else np.nan for a in attacks])
        hi = np.array([100 * g.loc[a, "psr_hi"] if a in g.index else np.nan for a in attacks])
        ax.bar(np.arange(len(attacks)) + (k - (len(sws) - 1) / 2) * w, y, w, label=sw_label(s),
               color=[color(a) for a in attacks], alpha=1.0 if k == 0 else 0.55,
               yerr=[np.clip(np.nan_to_num(y - lo), 0, None), np.clip(np.nan_to_num(hi - y), 0, None)], error_kw={"lw": 0.5, "capsize": 1})
    ax.set_xticks(range(len(attacks)), [label(a) for a in attacks], rotation=30, ha="right")
    ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}]")
    ax.set_ylim(0, 105)
    if len(sws) > 1:
        ax.legend(title="swapper (solid / faded)", frameon=False)
    save_fig(fig, out, "main_psr")


def make_all(raw_dir: Path, figs: Path, tables: Path, calibration: Path | None = None) -> dict[str, pd.DataFrame]:
    set_style()
    R = load_raw(raw_dir, calibration)
    if not len(R.swap):
        print(f"notice: no *_swap.csv in {raw_dir}; nothing to plot")
        return {}
    T = build_tables(R)
    write_tables(R, T, tables)
    fig_main_bar(T, figs)
    fig_robustness(T, figs)
    fig_ablation(T, figs)
    fig_distributions(R, figs)
    if "transfer" not in T:
        print("notice: only one swapper present; no transfer table")
    return T
