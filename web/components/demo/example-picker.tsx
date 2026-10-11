"use client";

import { motion } from "framer-motion";
import { ExternalLink, Loader2, UserRound } from "lucide-react";
import { cn } from "@/lib/cn";
import { EXAMPLES, type Example } from "@/lib/examples";

/** One-click public-domain victims whose clean swap is unmistakable (picked by scripts/select_demo_examples.py). */
export function ExamplePicker({
  selected,
  loading,
  disabled,
  onSelect,
}: {
  selected: string | null;
  loading: string | null;
  disabled?: boolean;
  onSelect: (ex: Example) => void;
}) {
  if (EXAMPLES.length === 0) return null;
  const current = EXAMPLES.find((e) => e.id === selected);

  return (
    <div className="mb-5 rounded-xl border border-line bg-white/[0.03] p-4">
      <p id="examples-h" className="mb-3 flex items-center gap-2 text-sm font-medium">
        <UserRound className="size-4 text-brand" aria-hidden /> No photo handy? Use an example victim
      </p>
      <p className="mb-3 text-xs text-muted">
        Public-domain NASA portraits, each paired with the target where the deepfake most clearly shows the victim. Fills in
        both photos and the target.
      </p>
      <div role="radiogroup" aria-labelledby="examples-h" className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {EXAMPLES.map((ex, i) => {
          const on = ex.id === selected;
          return (
            <motion.button
              key={ex.id}
              type="button"
              role="radio"
              aria-checked={on}
              aria-busy={loading === ex.id}
              disabled={disabled || loading !== null}
              onClick={() => onSelect(ex)}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className={cn(
                "group relative overflow-hidden rounded-xl border text-left transition disabled:cursor-not-allowed disabled:opacity-60",
                on ? "border-brand ring-2 ring-brand/50" : "border-line hover:border-brand/60",
              )}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={ex.thumb} alt="" className="aspect-square w-full object-cover" loading="lazy" />
              <span className="block truncate px-2 py-1.5 text-xs">{ex.name}</span>
              {loading === ex.id && (
                <span className="absolute inset-0 grid place-items-center bg-black/50">
                  <Loader2 className="size-6 animate-spin" aria-hidden />
                </span>
              )}
            </motion.button>
          );
        })}
      </div>
      <p className="sr-only" role="status" aria-live="polite">
        {loading ? "Loading example photos…" : current ? `Example ${current.name} loaded: photo, reference and target filled in.` : ""}
      </p>
      <p className="mt-3 text-[11px] text-muted">
        Sources:{" "}
        {EXAMPLES.map((ex, i) => (
          <span key={ex.id}>
            {i > 0 && " · "}
            <a href={ex.source} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-0.5 underline hover:text-fg">
              {ex.name}
              <ExternalLink className="size-3" aria-hidden />
            </a>
          </span>
        ))}{" "}
        (Wikimedia Commons, PD-USGov-NASA; full credits in the repository).
      </p>
    </div>
  );
}
