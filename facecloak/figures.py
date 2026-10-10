"""Tables and figures from ``results/raw`` (shared by make_figures.py and export_web_results.py).

Numbers only ever come from the CSVs. A table / figure whose data is absent is skipped with a printed
notice. PSR (protection success rate) = fraction of valid swaps whose identity similarity to the
reference photo falls below the judge's calibrated threshold (protected_<judge> column); rows where the
swap failed (no source face / no swapped face found) are excluded from PSR and counted separately
(``det_fail`` and ``psr_failasprot``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from facecloak.metrics import bootstrap_ci
from facecloak.transforms import ROBUSTNESS

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
        runs = self.metas.get(cfg, {}).get("runs", [])
        for r in reversed(runs):
            if judge in r.get("tau", {}):
                return float(r["tau"][judge])
        return None


def load_raw(raw_dir: Path) -> Raw:
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
    return Raw(swap, cloak, metas)


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


def pick_swapper(df: pd.DataFrame) -> str | None:
    sw = list(df["swapper"].unique()) if len(df) else []
    return "inswapper" if "inswapper" in sw else (min(sw) if sw else None)


# ----------------------------------------------------------------------------- statistics
def psr_stats(df: pd.DataFrame, judge: str) -> dict[str, float]:
    n = len(df)
    prot = df[f"protected_{judge}"].to_numpy(dtype=float) if n else np.array([])
    valid = prot[~np.isnan(prot)]
    mean, lo, hi = bootstrap_ci(valid) if len(valid) else (np.nan, np.nan, np.nan)
    sims = df[f"id_sim_swap_{judge}"].to_numpy(dtype=float) if n else np.array([])
    return {
        "n": n, "n_valid": len(valid), "psr": mean, "psr_lo": lo, "psr_hi": hi,
        "psr_failasprot": 1.0 - float(np.sum(valid == 0)) / n if n else np.nan,
        "mean_id_sim": float(np.nanmean(sims)) if np.any(~np.isnan(sims)) else np.nan,
        "det_fail": 1.0 - float(df["source_detected"].mean()) if n else np.nan,
    }


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
    """All numeric tables (empty dict entries are omitted)."""
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
        row = {"attack": a, "swapper": sw, "judge": pj, **st}
        for j in judges:
            if j != pj:
                s2 = psr_stats(g, j)
                row[f"psr_{j}"] = s2["psr"]
                row[f"psr_{j}_lo"], row[f"psr_{j}_hi"] = s2["psr_lo"], s2["psr_hi"]
                row[f"mean_id_sim_{j}"] = s2["mean_id_sim"]
        for k in ("psnr", "ssim", "lpips", "cloak_seconds"):
            row[k] = q.loc[a, k] if len(q) and a in q.index else np.nan
        row["whitebox_judges"] = ",".join(sorted(wb & set(judges)))
        rows.append(row)
    if rows:
        t = pd.DataFrame(rows)
        t["_o"] = t["attack"].map({a: i for i, a in enumerate(order_attacks(list(t["attack"].unique())))})
        out["main"] = t.sort_values(["swapper", "_o"]).drop(columns="_o").reset_index(drop=True)

    # robustness: attack x transform (one swapper)
    sw0 = pick_swapper(S)
    R0 = S[S["swapper"] == sw0]
    if R0["transform"].nunique() > 1:
        rows = []
        for (a, tf), g in R0.groupby(["attack", "transform"]):
            st = psr_stats(g, pj)
            rows.append({"attack": a, "transform": tf, "swapper": sw0, "judge": pj, **st})
        t = pd.DataFrame(rows)
        t["_o"] = t["attack"].map({a: i for i, a in enumerate(order_attacks(list(t["attack"].unique())))})
        t["_t"] = t["transform"].map({x: i for i, x in enumerate(order_transforms(list(t["transform"].unique())))})
        out["robustness"] = t.sort_values(["_o", "_t"]).drop(columns=["_o", "_t"]).reset_index(drop=True)

    # transfer: inswapper vs simswap
    if base["swapper"].nunique() > 1 and "main" in out:
        out["transfer"] = out["main"][["attack", "swapper", "judge", "n", "n_valid", "psr", "psr_lo", "psr_hi",
                                       "mean_id_sim", "det_fail"]].copy()

    # ablations
    for name, key, xcol in (("ablation_eps", "eps", "eps"), ("ablation_steps", "steps", "steps")):
        if name not in R.configs:
            continue
        A = R.swap[R.swap["config"] == name]
        acfg = R.metas.get(name, {}).get("config", {}).get("attacks", {})
        if not acfg:
            print(f"notice: {name}_meta.json missing; cannot map attacks to {key}, skipping {name}")
            continue
        aj = R.primary_judge(R.judges(A))
        asw = pick_swapper(A)
        q = cloak_quality(R, name).set_index("attack")
        rows = []
        for (a, tf), g in A[A["swapper"] == asw].groupby(["attack", "transform"]):
            if a not in acfg:
                continue
            st = psr_stats(g, aj)
            rows.append({"attack": a, xcol: acfg[a][key], "transform": tf, "swapper": asw, "judge": aj, **st,
                         "lpips": q.loc[a, "lpips"] if a in q.index else np.nan,
                         "psnr": q.loc[a, "psnr"] if a in q.index else np.nan})
        if rows:
            out[name] = pd.DataFrame(rows).sort_values([xcol, "transform"]).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------- table writers
def _tex_escape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt_pct(x: float, lo: float | None = None, hi: float | None = None) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{100 * x:.1f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s


def fmt_num(x: float, d: int = 3) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def write_tex(path: Path, header: list[str], rows: list[list[str]], align: str | None = None) -> None:
    align = align or "ll" + "r" * (len(header) - 2)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(_tex_escape(h) for h in header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(_tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def write_tables(R: Raw, T: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wb = R.whitebox()
    for name, t in T.items():
        t.to_csv(out / f"{name}.csv", index=False)
    if "main" in T:
        t = T["main"]
        pj = t["judge"].iloc[0]
        others = [c[4:] for c in t.columns if c.startswith("psr_") and not c.endswith(("_lo", "_hi", "_failasprot"))
                  and c not in ("psr",)]
        hdr = ["Attack", "Swapper", f"PSR@tau {pj} (%)"] + [
            f"PSR {j}{' (white-box)' if j in wb else ''} (%)" for j in others
        ] + [f"Mean id-sim {pj}", "PSNR", "SSIM", "LPIPS", "Det-fail (%)"]
        rows = []
        for _, r in t.iterrows():
            rows.append([label(r["attack"]), r["swapper"], fmt_pct(r["psr"], r["psr_lo"], r["psr_hi"])]
                        + [fmt_pct(r[f"psr_{j}"], r[f"psr_{j}_lo"], r[f"psr_{j}_hi"]) for j in others]
                        + [fmt_num(r["mean_id_sim"]), fmt_num(r["psnr"], 1), fmt_num(r["ssim"]),
                           fmt_num(r["lpips"]), fmt_pct(r["det_fail"])])
        write_tex(out / "main.tex", hdr, rows)
    if "robustness" in T:
        t = T["robustness"]
        tfs = order_transforms(list(t["transform"].unique()))
        hdr = ["Attack"] + tfs
        rows = []
        for a in order_attacks(list(t["attack"].unique())):
            g = t[t["attack"] == a].set_index("transform")
            rows.append([label(a)] + [fmt_pct(g.loc[x, "psr"]) if x in g.index else "--" for x in tfs])
        write_tex(out / "robustness.tex", hdr, rows, "l" + "r" * len(tfs))
    if "transfer" in T:
        t = T["transfer"]
        sws = sorted(t["swapper"].unique())
        hdr = ["Attack"] + [f"PSR {s} (%)" for s in sws] + [f"Mean id-sim {s}" for s in sws]
        rows = []
        for a in order_attacks(list(t["attack"].unique())):
            g = t[t["attack"] == a].set_index("swapper")
            rows.append([label(a)]
                        + [fmt_pct(g.loc[s, "psr"], g.loc[s, "psr_lo"], g.loc[s, "psr_hi"]) if s in g.index else "--"
                           for s in sws]
                        + [fmt_num(g.loc[s, "mean_id_sim"]) if s in g.index else "--" for s in sws])
        write_tex(out / "transfer.tex", hdr, rows, "l" + "r" * (2 * len(sws)))
    print(f"wrote tables {sorted(T)} to {out}")


# ----------------------------------------------------------------------------- figures
def fig_robustness(T: dict[str, pd.DataFrame], out: Path) -> None:
    if "robustness" not in T:
        print("notice: no multi-transform data; skipping robustness figure")
        return
    t = T["robustness"]
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
               yerr=[np.nan_to_num(y - lo), np.nan_to_num(hi - y)], error_kw={"lw": 0.4, "capsize": 0.8})
    ax.set_xticks(range(len(tfs)), tfs, rotation=30, ha="right")
    ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}, {t['swapper'].iloc[0]}]")
    ax.set_ylim(0, 105)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    save_fig(fig, out, "robustness")


def fig_ablation(T: dict[str, pd.DataFrame], out: Path) -> None:
    for name, xcol, xl in (("ablation_eps", "eps", r"$\epsilon$ (/255)"), ("ablation_steps", "steps", "PGD steps")):
        if name not in T:
            print(f"notice: no {name} data; skipping")
            continue
        t = T[name]
        fig, ax = plt.subplots(figsize=(COL_W, 2.2))
        for tf, mk in zip(order_transforms(list(t["transform"].unique())), "os^vD", strict=False):
            g = t[t["transform"] == tf].sort_values(xcol)
            ax.errorbar(g[xcol], 100 * g["psr"], yerr=[100 * (g["psr"] - g["psr_lo"]), 100 * (g["psr_hi"] - g["psr"])],
                        marker=mk, ms=3, capsize=1.5, label=tf)
        ax.set_xlabel(xl)
        ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}]")
        ax.set_ylim(0, 105)
        ax.legend(title="transform", frameon=False)
        save_fig(fig, out, name)
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
            ax.set_ylim(0, 105)
            ax.legend(title="transform", frameon=False)
            save_fig(fig, out, "ablation_psr_vs_lpips")


def fig_distributions(R: Raw, out: Path) -> None:
    main = pick_main(R)
    if main is None:
        print("notice: no data; skipping id-sim distribution figure")
        return
    S = R.swap[(R.swap["config"] == main) & (R.swap["transform"] == "none")]
    sw = pick_swapper(S)
    pj = R.primary_judge(R.judges(S))
    S = S[S["swapper"] == sw]
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
        print("notice: need >= 2 attack groups with valid swaps; skipping id-sim distribution figure")
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
    ax.grid(axis="x", visible=False)
    save_fig(fig, out, "id_sim_distributions")


def fig_main_bar(T: dict[str, pd.DataFrame], out: Path) -> None:
    if "main" not in T:
        return
    t = T["main"]
    sws = sorted(t["swapper"].unique())
    attacks = order_attacks(list(t["attack"].unique()))
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    w = 0.8 / len(sws)
    for k, s in enumerate(sws):
        g = t[t["swapper"] == s].set_index("attack")
        y = np.array([100 * g.loc[a, "psr"] if a in g.index else np.nan for a in attacks])
        lo = np.array([100 * g.loc[a, "psr_lo"] if a in g.index else np.nan for a in attacks])
        hi = np.array([100 * g.loc[a, "psr_hi"] if a in g.index else np.nan for a in attacks])
        ax.bar(np.arange(len(attacks)) + (k - (len(sws) - 1) / 2) * w, y, w, label=s,
               color=[color(a) for a in attacks], alpha=1.0 if k == 0 else 0.55,
               yerr=[np.nan_to_num(y - lo), np.nan_to_num(hi - y)], error_kw={"lw": 0.5, "capsize": 1})
    ax.set_xticks(range(len(attacks)), [label(a) for a in attacks], rotation=30, ha="right")
    ax.set_ylabel(f"PSR (%) [{t['judge'].iloc[0]}]")
    ax.set_ylim(0, 105)
    if len(sws) > 1:
        ax.legend(title="swapper (solid / faded)", frameon=False)
    save_fig(fig, out, "main_psr")


def make_all(raw_dir: Path, figs: Path, tables: Path) -> dict[str, pd.DataFrame]:
    set_style()
    R = load_raw(raw_dir)
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
