"use client";

import { motion } from "framer-motion";
import { Check, Loader2, ScanFace, Scale, Shuffle, Wand2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { NumberTicker } from "@/components/ui/number-ticker";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { cn } from "@/lib/cn";
import type { JobStatus, Stage } from "@/lib/types";

const STAGES: { id: Stage; label: string; icon: typeof ScanFace; desc: string }[] = [
  { id: "detecting", label: "Detecting", icon: ScanFace, desc: "Finding your face" },
  { id: "cloaking", label: "Cloaking", icon: Wand2, desc: "Optimising the invisible perturbation" },
  { id: "swapping", label: "Swapping", icon: Shuffle, desc: "Attacker tries to swap your face" },
  { id: "judging", label: "Judging", icon: Scale, desc: "Independent model compares identities" },
];

export function RunProgress({
  status,
  submitting,
  startedAt,
  onCancel,
}: {
  status: JobStatus | null;
  submitting: boolean;
  startedAt: number | null;
  onCancel: () => void;
}) {
  const [now, setNow] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);
  const elapsed = startedAt && now ? Math.max(0, Math.round((now - startedAt) / 1000)) : 0;
  const queued = status?.status === "queued";
  const cur = status?.progress?.stage;
  const idx = cur ? STAGES.findIndex((s) => s.id === cur) : -1;
  const p = status?.progress;
  const frac = p && p.total > 0 && p.stage === "cloaking" ? Math.min(1, p.step / p.total) : 0;

  return (
    <div className="card p-5 sm:p-6" aria-live="polite">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 tabIndex={-1} data-focus-target className="flex items-center gap-2 font-semibold outline-none">
          <Loader2 className="size-5 animate-spin text-brand2" aria-hidden />
          {submitting ? "Uploading your photo..." : queued ? "Waiting in the queue" : "Working on it"}
        </h3>
        <span className="font-mono text-sm text-muted" aria-label={`${elapsed} seconds elapsed`}>{elapsed}s</span>
      </div>

      {queued && (
        <p className="mt-3 rounded-lg border border-line bg-white/5 px-3 py-2 text-sm">
          {status.queue_position === 0 || status.queue_position == null
            ? "You are next. Starting in a moment."
            : `${status.queue_position} job${status.queue_position === 1 ? "" : "s"} ahead of you.`}
        </p>
      )}

      <ol className="mt-5 grid gap-3 sm:grid-cols-4">
        {STAGES.map((s, i) => {
          const done = idx > i;
          const active = idx === i && !queued;
          const Icon = s.icon;
          return (
            <li
              key={s.id}
              aria-current={active ? "step" : undefined}
              className={cn(
                "rounded-xl border p-3 transition-colors",
                active ? "border-brand2/60 bg-brand2/10" : done ? "border-ok/30 bg-ok/5" : "border-line bg-white/[0.02] opacity-60",
              )}
            >
              <div className="flex items-center gap-2">
                <span className={cn("grid size-7 place-items-center rounded-lg", active ? "bg-brand2/20 text-brand2" : done ? "bg-ok/20 text-ok" : "bg-white/10 text-muted")}>
                  {done ? <Check className="size-4" aria-hidden /> : active ? <Icon className="size-4 animate-pulse" aria-hidden /> : <Icon className="size-4" aria-hidden />}
                </span>
                <span className="text-sm font-semibold">{s.label}</span>
              </div>
              <p className="mt-1.5 text-xs text-muted">{s.desc}</p>
            </li>
          );
        })}
      </ol>

      {p?.stage === "cloaking" && (
        <div className="mt-5">
          <div className="mb-1.5 flex items-baseline justify-between text-sm">
            <span className="text-muted">PGD step</span>
            <span className="font-mono">
              <NumberTicker value={p.step} /> / {p.total}
            </span>
          </div>
          <div
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={p.total}
            aria-valuenow={p.step}
            aria-label="Cloaking progress"
            className="h-2.5 overflow-hidden rounded-full bg-white/10"
          >
            <motion.div className="h-full rounded-full bg-gradient-to-r from-brand to-brand2" animate={{ width: `${frac * 100}%` }} transition={{ ease: "linear", duration: 0.6 }} />
          </div>
          <p className="mt-1.5 text-xs text-muted">Step count is estimated from elapsed time.</p>
        </div>
      )}

      <div className="mt-5">
        <ShimmerButton type="button" variant="ghost" onClick={onCancel}>
          <X className="size-4" aria-hidden /> Cancel
        </ShimmerButton>
      </div>
    </div>
  );
}
