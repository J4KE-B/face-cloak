"use client";

import { motion } from "framer-motion";
import { Check, ExternalLink } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cn";
import type { Target } from "@/lib/types";

export function TargetPicker({
  targets,
  state,
  value,
  onChange,
}: {
  targets: Target[];
  state: "idle" | "loading" | "ok" | "unauthorized" | "error";
  value: string | null;
  onChange: (id: string) => void;
}) {
  const sel = targets.find((t) => t.id === value);
  return (
    <div>
      {state === "loading" && (
        <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6" aria-busy="true" aria-label="Loading targets">
          {Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="aspect-square" />)}
        </div>
      )}
      {(state === "idle" || state === "unauthorized" || state === "error") && (
        <p className="rounded-xl border border-dashed border-line p-6 text-center text-sm text-muted">
          {state === "idle" ? "Connect to a running backend to load the target portraits." : state === "unauthorized" ? "The passcode was rejected. Fix it above and press Reconnect." : "Could not load the targets. Press Reconnect to retry."}
        </p>
      )}
      {state === "ok" && (
        <>
          <div role="radiogroup" aria-label="Target portrait" className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
            {targets.map((t, i) => {
              const on = t.id === value;
              return (
                <motion.div key={t.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }} className="group relative">
                  <button
                    type="button"
                    role="radio"
                    aria-checked={on}
                    aria-label={`${t.name ?? "Target"}, license: ${t.license ?? "unknown"}`}
                    onClick={() => onChange(t.id)}
                    className={cn(
                      "relative block aspect-square w-full overflow-hidden rounded-xl border-2 transition-all",
                      on ? "border-brand2 shadow-[0_0_24px_-4px_rgb(34_211_238/0.7)]" : "border-transparent opacity-80 hover:opacity-100",
                    )}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={t.thumbnail ?? ""} alt="" className="size-full object-cover" />
                    {on && (
                      <span className="absolute top-1.5 right-1.5 grid size-6 place-items-center rounded-full bg-brand2 text-black">
                        <Check className="size-4" aria-hidden />
                      </span>
                    )}
                  </button>
                  <span
                    role="tooltip"
                    className="pointer-events-none absolute -top-2 left-1/2 z-20 w-max max-w-44 -translate-x-1/2 -translate-y-full rounded-lg border border-line bg-surface px-2.5 py-1.5 text-center text-xs opacity-0 shadow-xl transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
                  >
                    <strong className="block">{t.name ?? "Target"}</strong>
                    <span className="text-muted">{t.license ?? "licence unknown"}</span>
                  </span>
                </motion.div>
              );
            })}
          </div>
          {sel && (
            <p className="mt-3 flex flex-wrap items-center gap-x-2 text-xs text-muted">
              Selected: <strong className="text-fg">{sel.name ?? "Target"}</strong> ({sel.license ?? "licence unknown"}).
              {sel.source && (
                <a href={sel.source} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 underline hover:text-fg">
                  source <ExternalLink className="size-3" aria-hidden />
                </a>
              )}
            </p>
          )}
        </>
      )}
    </div>
  );
}
