"use client";

import { useRef, type KeyboardEvent } from "react";
import { Clock, Gauge } from "lucide-react";
import { cn } from "@/lib/cn";
import { estimateSeconds, fmtDuration } from "@/lib/format";
import type { Health, Mode } from "@/lib/types";

export function Controls({
  health,
  eps,
  steps,
  mode,
  onEps,
  onSteps,
  onMode,
}: {
  health: Health | null;
  eps: number;
  steps: number;
  mode: Mode;
  onEps: (n: number) => void;
  onSteps: (n: number) => void;
  onMode: (m: Mode) => void;
}) {
  const maxEps = health?.max_eps ?? 16;
  const maxSteps = health?.max_steps ?? 60;
  const est = estimateSeconds(health?.device, steps, mode);
  const MODES = ["fast", "robust"] as const;
  const radios = useRef<(HTMLButtonElement | null)[]>([]);
  const onModeKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const next = (MODES.indexOf(mode) + d + MODES.length) % MODES.length;
    onMode(MODES[next]!);
    radios.current[next]?.focus();
  };
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <div className="flex items-baseline justify-between">
          <label htmlFor="eps" className="text-sm font-medium">Strength ε (max change per pixel)</label>
          <output htmlFor="eps" className="font-mono text-sm text-brand2">{eps}/255</output>
        </div>
        <input id="eps" type="range" min={2} max={Math.max(2, maxEps)} step={1} value={eps} onChange={(e) => onEps(Number(e.target.value))} className="mt-2 w-full" aria-describedby="eps-help" />
        <p id="eps-help" className="mt-1 text-xs text-muted">Higher = stronger protection but more visible noise. 8/255 is the project default (barely visible); 4 is nearly invisible.</p>
      </div>
      <div>
        <div className="flex items-baseline justify-between">
          <label htmlFor="steps" className="text-sm font-medium">Optimisation steps</label>
          <output htmlFor="steps" className="font-mono text-sm text-brand2">{steps}</output>
        </div>
        <input id="steps" type="range" min={1} max={Math.max(1, maxSteps)} step={1} value={steps} onChange={(e) => onSteps(Number(e.target.value))} className="mt-2 w-full" aria-describedby="steps-help" />
        <p id="steps-help" className="mt-1 text-xs text-muted">More steps = better cloak, slower run (maximum {maxSteps} on this backend).</p>
      </div>
      <fieldset className="md:col-span-2">
        <legend className="mb-2 flex items-center gap-2 text-sm font-medium"><Gauge className="size-4 text-muted" aria-hidden /> Mode</legend>
        <div role="radiogroup" aria-label="Mode" onKeyDown={onModeKey} className="inline-flex rounded-xl border border-line bg-black/30 p-1">
          {MODES.map((m, i) => (
            <button
              key={m}
              ref={(el) => {
                radios.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={mode === m}
              tabIndex={mode === m ? 0 : -1}
              onClick={() => onMode(m)}
              className={cn("rounded-lg px-4 py-1.5 text-sm font-medium capitalize transition-colors", mode === m ? "bg-brand text-white" : "text-muted hover:text-fg")}
            >
              {m}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted">
          {mode === "fast"
            ? "Fast: one recognition model, plain PGD."
            : "Robust: Expectation over Transformation (EOT) simulates compression and resizing so the cloak survives re-sharing, and a FaceNet ensemble is added to the optimisation. Roughly twice as slow; on a CPU backend a robust run counts as 2 of your hourly jobs."}
        </p>
        <p className="mt-2 text-xs text-muted">
          <Clock className="mr-1.5 inline size-3.5 align-text-bottom" aria-hidden />
          Estimated time on {health ? (health.device === "cuda" ? "GPU" : "CPU") : "this backend"}:{" "}
          <strong className="text-fg">{fmtDuration(est)}</strong> (rough guess, plus waiting time if others are in the queue)
        </p>
      </fieldset>
    </div>
  );
}
