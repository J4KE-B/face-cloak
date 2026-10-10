"use client";

import { Cpu, Loader2, ServerCrash, Wifi, WifiOff, Zap } from "lucide-react";
import { cn } from "@/lib/cn";
import type { Health } from "@/lib/types";
import type { HealthPhase } from "./use-backend";

export function HealthBadge({ phase, health }: { phase: HealthPhase; health: Health | null }) {
  const cfg = {
    idle: { t: "Not connected", c: "border-line text-muted", i: <WifiOff className="size-4" /> },
    checking: { t: "Checking...", c: "border-line text-muted", i: <Loader2 className="size-4 animate-spin" /> },
    offline: { t: "Offline", c: "border-bad/40 bg-bad/10 text-bad", i: <ServerCrash className="size-4" /> },
    waking: { t: "Waking up (loading models)", c: "border-warn/40 bg-warn/10 text-warn", i: <Loader2 className="size-4 animate-spin" /> },
    online: { t: "Online", c: "border-ok/40 bg-ok/10 text-ok", i: <Wifi className="size-4" /> },
  }[phase];
  return (
    <div role="status" aria-live="polite" className="flex flex-wrap items-center gap-2 text-sm">
      <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-medium", cfg.c)}>
        {phase === "online" && <span className="size-2 animate-pulse rounded-full bg-ok" aria-hidden />}
        {cfg.i}
        {cfg.t}
      </span>
      {health && (phase === "online" || phase === "waking") && (
        <>
          <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-3 py-1 text-muted">
            {health.device === "cuda" ? <Zap className="size-4 text-brand2" aria-hidden /> : <Cpu className="size-4" aria-hidden />}
            {health.device === "cuda" ? "GPU (cuda)" : "CPU"}
          </span>
          <span className="rounded-full border border-line px-3 py-1 text-muted">
            queue: {health.queue_length}
          </span>
        </>
      )}
    </div>
  );
}
