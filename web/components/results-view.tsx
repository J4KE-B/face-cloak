"use client";

import { useMemo, type ReactNode } from "react";
import { FlaskConical, Hourglass, Info } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { GlowCard } from "@/components/ui/glow-card";
import { Reveal } from "@/components/ui/section";
import { cn } from "@/lib/cn";
import { num, pct } from "@/lib/format";
import { attackLabel, figureTitle, isWhitebox, pickTau, prettyKey, PRIMARY_JUDGE, swapperLabel, useSummary } from "@/lib/results";
import type { ResultRow, ResultsSummary } from "@/lib/types";

// Never publish qualitative grids of real people, even if one slipped into the export.
const BLOCKED = /qualitative|grid|gallery|faces?_|portrait|sample/i;

function SwapperChip({ sw }: { sw: string }) {
  const { name, kind } = swapperLabel(sw);
  return (
    <span className={cn("inline-flex flex-wrap items-center gap-1 rounded-full border px-2 py-0.5 text-xs", isWhitebox(sw) ? "border-brand/40 bg-brand/10 text-brand" : "border-brand2/40 bg-brand2/10 text-brand2")}>
      <strong>{name}</strong>
      {kind && <span className="opacity-80">({kind})</span>}
    </span>
  );
}

function PsrCell({ r }: { r: ResultRow }) {
  if (r.psr == null) return <span className="text-muted">n/a</span>;
  return (
    <div className="min-w-32">
      <div className="flex items-baseline gap-1.5 whitespace-nowrap">
        <strong className="tabular-nums">{pct(r.psr)}</strong>
        {r.psr_lo != null && r.psr_hi != null && (
          <span className="text-xs text-muted tabular-nums">
            [{pct(r.psr_lo)}, {pct(r.psr_hi)}]{r.psr_ci === "wilson" ? "*" : ""}
          </span>
        )}
      </div>
      <div className="mt-1 h-1.5 rounded-full bg-white/10">
        <div className="h-full rounded-full bg-gradient-to-r from-brand to-brand2" style={{ width: `${Math.max(0, Math.min(1, r.psr)) * 100}%` }} />
      </div>
    </div>
  );
}

function Table({ head, children, caption }: { head: ReactNode[]; children: ReactNode; caption: string }) {
  return (
    <div className="card overflow-x-auto">
      <table className="w-full min-w-[34rem] text-left text-sm">
        <caption className="sr-only">{caption} (judge: {PRIMARY_JUDGE})</caption>
        <thead>
          <tr className="border-b border-line text-xs tracking-wider text-muted uppercase">
            {head.map((h, i) => <th key={i} scope="col" className="px-4 py-3 font-semibold">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">{children}</tbody>
      </table>
    </div>
  );
}

function ResultTable({ k, rows }: { k: string; rows: ResultRow[] }) {
  const title = prettyKey(k);
  if (k === "main") {
    return (
      <Table caption={title} head={["Attack", "Swapper", "Valid / total", "PSR [95% CI]", "Mean ID-sim", "PSNR", "SSIM", "LPIPS"]}>
        {rows.map((r, i) => (
          <tr key={i} className="hover:bg-white/[0.03]">
            <th scope="row" className="px-4 py-3 font-medium">{attackLabel(r.attack)}</th>
            <td className="px-4 py-3"><SwapperChip sw={r.swapper} /></td>
            <td className="px-4 py-3 tabular-nums">{r.n_valid ?? "n/a"} / {r.n ?? "n/a"}</td>
            <td className="px-4 py-3"><PsrCell r={r} /></td>
            <td className="px-4 py-3 tabular-nums">{num(r.mean_id_sim)}</td>
            <td className="px-4 py-3 tabular-nums">{num(r.psnr, 1)}</td>
            <td className="px-4 py-3 tabular-nums">{num(r.ssim)}</td>
            <td className="px-4 py-3 tabular-nums">{num(r.lpips)}</td>
          </tr>
        ))}
      </Table>
    );
  }
  if (k.startsWith("robustness_")) {
    const tfs = [...new Set(rows.map((r) => r.transform ?? "none"))];
    const atts = [...new Set(rows.map((r) => r.attack))];
    const get = (a: string, t: string) => rows.find((r) => r.attack === a && (r.transform ?? "none") === t);
    return (
      <Table caption={title} head={["Attack", ...tfs]}>
        {atts.map((a) => (
          <tr key={a} className="hover:bg-white/[0.03]">
            <th scope="row" className="px-4 py-3 font-medium">{attackLabel(a)}</th>
            {tfs.map((t) => {
              const r = get(a, t);
              return <td key={t} className="px-4 py-3">{r ? <PsrCell r={r} /> : <span className="text-muted">n/a</span>}</td>;
            })}
          </tr>
        ))}
      </Table>
    );
  }
  if (k.startsWith("ablation_")) {
    const x = k.startsWith("ablation_eps") ? "eps" : "steps";
    return (
      <Table caption={title} head={[x === "eps" ? "ε (/255)" : "Steps", "Transform", "PSR [95% CI]", "PSNR", "LPIPS"]}>
        {rows.map((r, i) => (
          <tr key={i} className="hover:bg-white/[0.03]">
            <th scope="row" className="px-4 py-3 font-medium tabular-nums">{String(r[x] ?? "n/a")}</th>
            <td className="px-4 py-3">{r.transform ?? "none"}</td>
            <td className="px-4 py-3"><PsrCell r={r} /></td>
            <td className="px-4 py-3 tabular-nums">{num(r.psnr, 1)}</td>
            <td className="px-4 py-3 tabular-nums">{num(r.lpips)}</td>
          </tr>
        ))}
      </Table>
    );
  }
  // transfer + anything unknown
  return (
    <Table caption={title} head={["Attack", "Swapper", "Valid / total", "PSR [95% CI]"]}>
      {rows.map((r, i) => (
        <tr key={i} className="hover:bg-white/[0.03]">
          <th scope="row" className="px-4 py-3 font-medium">{attackLabel(r.attack)}</th>
          <td className="px-4 py-3"><SwapperChip sw={r.swapper} /></td>
          <td className="px-4 py-3 tabular-nums">{r.n_valid ?? "n/a"} / {r.n ?? "n/a"}</td>
          <td className="px-4 py-3"><PsrCell r={r} /></td>
        </tr>
      ))}
    </Table>
  );
}

function Placeholder({ loading }: { loading: boolean }) {
  return (
    <div className="card relative overflow-hidden p-8 text-center sm:p-12">
      <Hourglass className="mx-auto size-10 text-brand2" aria-hidden />
      <h2 className="mt-4 text-2xl font-bold">{loading ? "Loading results..." : "Results will appear after the Kaggle run"}</h2>
      <p className="mx-auto mt-2 max-w-lg text-muted">
        {loading ? "Fetching the exported summary." : (
          <>
            The full experiment (150 identities, two face swappers, post-processing stress tests) runs on a Kaggle GPU. Its
            output is exported with <code className="font-mono text-fg">scripts/export_web_results.py</code> into{" "}
            <code className="font-mono text-fg">web/public/results/</code>, and this page fills itself in.
          </>
        )}
      </p>
      <div className="mx-auto mt-8 grid max-w-2xl gap-3 sm:grid-cols-3" aria-hidden>
        <Skeleton className="h-24" /><Skeleton className="h-24" /><Skeleton className="h-24" />
      </div>
    </div>
  );
}

function Loaded({ d }: { d: ResultsSummary }) {
  const keys = useMemo(() => {
    const order = ["main", "transfer"];
    const all = Object.keys(d.tables).filter((k) => (d.tables[k]?.length ?? 0) > 0);
    return [...order.filter((k) => all.includes(k)), ...all.filter((k) => !order.includes(k)).sort()];
  }, [d.tables]);
  const figs = d.figures.filter((f) => f.toLowerCase().endsWith(".png") && !BLOCKED.test(f));
  const tau = pickTau(d);

  return (
    <div className="space-y-12">
      <Reveal>
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <GlowCard className="p-6">
            <h2 className="flex items-center gap-2 font-semibold"><Info className="size-4 text-brand2" aria-hidden /> How to read these numbers</h2>
            <ul className="mt-3 space-y-2 text-sm text-muted">
              <li><strong className="text-fg">PSR</strong> (protection success rate): share of valid swaps whose identity score stayed below the judge&apos;s threshold τ, so the swap no longer looks like you. 95% CI from a cluster bootstrap over source identities; * marks a Wilson interval used when the bootstrap was degenerate.</li>
              <li><SwapperChip sw="inswapper" /> the cloak was optimised against this swapper&apos;s own identity encoder, so it is a <strong className="text-fg">white-box</strong> result.</li>
              <li><SwapperChip sw="simswap" /> the cloak never saw this model; success here measures <strong className="text-fg">black-box transfer</strong>.</li>
              <li>The judge is an independent model ({PRIMARY_JUDGE}) so the cloak is not grading itself.</li>
            </ul>
          </GlowCard>
          <GlowCard className="p-6">
            <h2 className="flex items-center gap-2 font-semibold"><FlaskConical className="size-4 text-brand2" aria-hidden /> Run details</h2>
            <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
              <dt className="text-muted">Swap rows</dt><dd className="tabular-nums">{d.n_swap_rows}</dd>
              <dt className="text-muted">Configs</dt><dd>{d.configs.join(", ") || "n/a"}</dd>
              <dt className="text-muted">Threshold τ</dt><dd className="tabular-nums">{tau != null ? tau.toFixed(3) : "n/a"} <span className="text-muted">({PRIMARY_JUDGE}{d.tau_source ? `, ${d.tau_source}` : ""})</span></dd>
              <dt className="text-muted">White-box judges</dt><dd>{d.whitebox_judges.join(", ") || "none"}</dd>
              <dt className="text-muted">Exported</dt><dd>{d.generated ? new Date(d.generated).toLocaleString() : "n/a"}</dd>
            </dl>
          </GlowCard>
        </div>
      </Reveal>

      <p className="-mt-6 text-sm text-muted">Results from our Kaggle run on LFW; see the report for the full protocol.</p>

      {keys.map((k) => (
        <Reveal key={k}>
          <section aria-labelledby={`t-${k}`}>
            <h2 id={`t-${k}`} className="mb-3 text-xl font-bold">{prettyKey(k)} <span className="text-sm font-normal text-muted">(judge: {PRIMARY_JUDGE})</span></h2>
            <ResultTable k={k} rows={d.tables[k] ?? []} />
          </section>
        </Reveal>
      ))}
      {keys.length === 0 && <p className="text-muted">The summary file contains no tables yet.</p>}

      {figs.length > 0 && (
        <section aria-labelledby="figs">
          <h2 id="figs" className="mb-3 text-xl font-bold">Figures</h2>
          <div className="grid gap-5 md:grid-cols-2">
            {figs.map((f, i) => {
              const base = f.replace(/\.png$/i, "");
              const title = figureTitle(base);
              return (
                <Reveal key={f} delay={(i % 2) * 0.08}>
                  <figure className="card overflow-hidden">
                    <a href={`/results/${encodeURIComponent(f)}`} target="_blank" rel="noopener noreferrer" aria-label={`${title}, open full size`}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={`/results/${encodeURIComponent(f)}`} alt={title} loading="lazy" className="w-full bg-white object-contain" />
                    </a>
                    <figcaption className="p-3 text-sm font-medium">{title} <span className="font-normal text-muted">(judge: {PRIMARY_JUDGE})</span></figcaption>
                  </figure>
                </Reveal>
              );
            })}
          </div>
        </section>
      )}
      <p className="text-xs text-muted">{d.note}</p>
    </div>
  );
}

export function ResultsClient() {
  const s = useSummary();
  if (s.status === "ok") return <Loaded d={s.data} />;
  return <Placeholder loading={s.status === "loading"} />;
}
