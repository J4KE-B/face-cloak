"use client";

import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { attackLabel, headlineRows, useSummary } from "@/lib/results";
import { NumberTicker } from "@/components/ui/number-ticker";
import { GlowCard } from "@/components/ui/glow-card";
import { Reveal, SectionHeading } from "@/components/ui/section";
import { pct } from "@/lib/format";

/** Headline numbers from /results/summary.json. Renders nothing until that file exists. */
export function HeadlineStats() {
  const s = useSummary();
  if (s.status !== "ok") return null;
  const b = headlineRows(s.data.tables.main);
  if (!b) return null;
  const { attack, wb, bb, clean } = b;
  const ci = (r: { psr_lo?: number | null; psr_hi?: number | null }) =>
    r.psr_lo != null && r.psr_hi != null ? `95% CI ${pct(r.psr_lo)} to ${pct(r.psr_hi)}; ` : "";
  // One fixed headline attack, not the best of N; each swapper is shown separately and labelled.
  const items = [
    wb && {
      label: "White-box: inswapper (cloak optimised against its encoder)",
      value: (wb.psr ?? 0) * 100,
      sub: `${ci(wb)}attack: ${attackLabel(attack)}`,
    },
    bb && {
      label: "Black-box transfer: SimSwap (encoder never seen)",
      value: (bb.psr ?? 0) * 100,
      sub: `${ci(bb)}attack: ${attackLabel(attack)}`,
    },
    clean && {
      label: "Baseline: no cloak (inswapper)",
      value: (clean.psr ?? 0) * 100,
      sub: "protection rate when nothing is done",
    },
  ].filter(Boolean) as { label: string; value: number; sub: string }[];

  return (
    <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6" aria-labelledby="numbers">
      <SectionHeading eyebrow="Measured" title="Headline numbers">
        Protection success rate (PSR): the share of valid face swaps whose identity score fell below the judge&apos;s
        threshold. Results from our Kaggle run on LFW; see the report for the full protocol.
      </SectionHeading>
      <div className="grid gap-4 md:grid-cols-3" id="numbers">
        {items.map((it, i) => (
          <Reveal key={it.label} delay={i * 0.08}>
            <GlowCard className="h-full p-6">
              <p className="text-gradient text-5xl font-bold tabular-nums">
                <NumberTicker value={it.value} decimals={1} suffix="%" />
              </p>
              <p className="mt-3 text-sm font-medium">{it.label}</p>
              <p className="mt-1 text-xs text-muted">{it.sub}</p>
            </GlowCard>
          </Reveal>
        ))}
      </div>
      <Link href="/results" className="mt-6 inline-flex items-center gap-1 text-sm text-brand2 hover:underline">
        All tables and figures <ArrowRight className="size-4" />
      </Link>
    </section>
  );
}
