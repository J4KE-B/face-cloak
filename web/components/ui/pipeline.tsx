"use client";

import { useEffect, useState, type ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/cn";

export interface PipelineStep {
  icon: ReactNode;
  title: string;
  body: string;
  side: "defender" | "attacker" | "judge";
}

const SIDE = {
  defender: { label: "You", cls: "text-brand border-brand/40 bg-brand/10" },
  attacker: { label: "Attacker", cls: "text-bad border-bad/40 bg-bad/10" },
  judge: { label: "Referee", cls: "text-brand2 border-brand2/40 bg-brand2/10" },
} as const;

/** Animated vertical stepper: a beam travels down the rail and the active step lights up. */
export function Pipeline({ steps }: { steps: PipelineStep[] }) {
  const reduce = useReducedMotion();
  const [active, setActive] = useState(0);
  useEffect(() => {
    if (reduce) return;
    const t = setInterval(() => setActive((a) => (a + 1) % steps.length), 2200);
    return () => clearInterval(t);
  }, [reduce, steps.length]);

  return (
    <ol className="relative mx-auto max-w-3xl">
      <div aria-hidden className="absolute top-3 bottom-3 left-[1.375rem] w-px bg-line" />
      {!reduce && (
        <motion.div
          aria-hidden
          className="absolute left-[1.3rem] w-[3px] rounded-full bg-gradient-to-b from-transparent via-brand2 to-transparent"
          style={{ height: `${100 / steps.length}%` }}
          animate={{ top: `${(active / steps.length) * 100}%` }}
          transition={{ type: "spring", stiffness: 70, damping: 18 }}
        />
      )}
      {steps.map((s, i) => {
        const on = reduce || i === active;
        const side = SIDE[s.side];
        return (
          <motion.li
            key={s.title}
            initial={{ opacity: 0, x: -16 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ delay: i * 0.05, duration: 0.45 }}
            className="relative flex gap-4 py-3"
          >
            <div
              className={cn(
                "relative z-10 grid size-11 shrink-0 place-items-center rounded-xl border bg-surface transition-all duration-500",
                on ? "border-brand2/70 text-brand2 shadow-[0_0_24px_-4px_rgb(34_211_238/0.6)]" : "border-line text-muted",
              )}
            >
              {s.icon}
              <span className="absolute -right-1.5 -bottom-1.5 grid size-5 place-items-center rounded-full bg-bg text-[10px] font-bold ring-1 ring-line">
                {i + 1}
              </span>
            </div>
            <div className={cn("card flex-1 p-4 transition-colors duration-500", on && "border-brand2/30 bg-white/[0.06]")}>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="font-semibold">{s.title}</h3>
                <span className={cn("rounded-full border px-2 py-0.5 text-[11px] font-medium", side.cls)}>{side.label}</span>
              </div>
              <p className="mt-1 text-sm text-muted">{s.body}</p>
            </div>
          </motion.li>
        );
      })}
    </ol>
  );
}
