"use client";

import { useEffect, useState } from "react";
import type { ResultRow, ResultsSummary } from "./types";

export const PRIMARY_JUDGE = "glintr100";

export type SummaryState =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "ok"; data: ResultsSummary };

function isSummary(x: unknown): x is ResultsSummary {
  if (typeof x !== "object" || x === null) return false;
  const o = x as Record<string, unknown>;
  return typeof o.tables === "object" && o.tables !== null && Array.isArray(o.figures);
}

/** Loads the static /results/summary.json written by scripts/export_web_results.py (absent until the Kaggle run). */
export function useSummary(): SummaryState {
  const [s, setS] = useState<SummaryState>({ status: "loading" });
  useEffect(() => {
    const ctrl = new AbortController();
    fetch("/results/summary.json", { signal: ctrl.signal })
      .then(async (r) => {
        if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) throw new Error("missing");
        const j: unknown = await r.json();
        if (!isSummary(j)) throw new Error("bad shape");
        setS({ status: "ok", data: j });
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setS({ status: "missing" });
      });
    return () => ctrl.abort();
  }, []);
  return s;
}

export function swapperLabel(sw: string): { name: string; kind: string } {
  const l = sw.toLowerCase();
  if (l.includes("inswapper")) return { name: "inswapper", kind: "white-box encoder" };
  if (l.includes("simswap")) return { name: "SimSwap", kind: "black-box transfer" };
  return { name: sw, kind: "" };
}

export function isWhitebox(sw: string) {
  return sw.toLowerCase().includes("inswapper");
}

export const TABLE_TITLES: Record<string, string> = {
  main: "Main results (no post-processing)",
  transfer: "White-box encoder vs black-box transfer",
};

/** Mirror of facecloak/figures.py LABELS. */
export const ATTACK_LABELS: Record<string, string> = {
  clean: "Clean",
  noise: "Random noise",
  pgd_r50: "PGD (R50)",
  pgd_ens: "PGD (ens.)",
  pgd_ens_targeted: "PGD ens. + decoy",
  pgd_ens_eot: "PGD ens. + EOT",
  pgd_ens_eot_lpips: "PGD ens. + EOT + LPIPS",
  pgd_ens_det: "PGD ens. + det.",
};

export function attackLabel(a: string): string {
  return ATTACK_LABELS[a] ?? a;
}

function finiteNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

/**
 * Threshold to display: tau_used[glintr100], else the first finite tau_used value, else the first finite
 * per-config value for glintr100 / any judge. Never NaN.
 */
export function pickTau(d: ResultsSummary): number | null {
  const used = d.tau_used;
  if (used && typeof used === "object") {
    if (finiteNum(used[PRIMARY_JUDGE])) return used[PRIMARY_JUDGE];
    const first = Object.values(used).find(finiteNum);
    if (first !== undefined) return first;
  }
  const cfgs = d.tau && typeof d.tau === "object" ? Object.values(d.tau) : [];
  for (const c of cfgs) {
    if (c && typeof c === "object" && finiteNum(c[PRIMARY_JUDGE])) return c[PRIMARY_JUDGE];
  }
  for (const c of cfgs) {
    if (c && typeof c === "object") {
      const first = Object.values(c).find(finiteNum);
      if (first !== undefined) return first;
    }
  }
  return null;
}

export function prettyKey(k: string): string {
  if (TABLE_TITLES[k]) return TABLE_TITLES[k];
  const m = /^(robustness|ablation_eps|ablation_steps)_(.+)$/.exec(k);
  if (m) {
    const what = m[1] === "robustness" ? "Robustness to post-processing" : m[1] === "ablation_eps" ? "Ablation: perturbation budget ε" : "Ablation: PGD steps";
    const { name, kind } = swapperLabel(m[2] ?? "");
    return `${what}, ${name}${kind ? ` (${kind})` : ""}`;
  }
  return k.replace(/_/g, " ");
}

/** Human title for a figure file stem (without .png). */
export function figureTitle(base: string): string {
  const fixed: Record<string, string> = {
    main_psr: "Protection success rate per attack",
    id_sim_distributions: "Identity-score distributions",
    robustness: "Robustness to JPEG, resizing and blur",
  };
  if (fixed[base]) return fixed[base];
  const m = /^(robustness|id_sim_distributions|ablation_eps|ablation_steps)_(.+)$/.exec(base);
  if (m) {
    const what: Record<string, string> = {
      robustness: "Robustness to JPEG, resizing and blur",
      id_sim_distributions: "Identity-score distributions",
      ablation_eps: "Ablation: perturbation budget ε",
      ablation_steps: "Ablation: PGD steps",
    };
    const { name, kind } = swapperLabel(m[2] ?? "");
    return `${what[m[1] ?? ""]}, ${name}${kind ? ` (${kind})` : ""}`;
  }
  return base.replace(/_/g, " ");
}

/** Fixed headline attack: pgd_ens_eot_lpips if present, else pgd_ens, else the first pgd_* attack. Not best-of-N. */
export function headlineRows(rows: ResultRow[] | undefined) {
  if (!rows) return null;
  const att = rows.filter((r) => r.attack !== "clean" && typeof r.psr === "number");
  const names = [...new Set(att.map((r) => r.attack))];
  const attack = names.includes("pgd_ens_eot_lpips")
    ? "pgd_ens_eot_lpips"
    : names.includes("pgd_ens")
      ? "pgd_ens"
      : names.find((n) => n.startsWith("pgd_"));
  if (!attack) return null;
  const wb = att.find((r) => r.attack === attack && isWhitebox(r.swapper)) ?? null;
  const bb = att.find((r) => r.attack === attack && !isWhitebox(r.swapper)) ?? null;
  if (!wb && !bb) return null;
  const clean = rows.find((r) => r.attack === "clean" && isWhitebox(r.swapper) && typeof r.psr === "number") ?? null;
  return { attack, wb, bb, clean };
}
