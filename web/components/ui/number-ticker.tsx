"use client";

import { animate, useMotionValue, useReducedMotion } from "framer-motion";
import { useEffect, useRef } from "react";

/** Smoothly counting number. */
export function NumberTicker({
  value,
  decimals = 0,
  suffix = "",
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  className?: string;
}) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLSpanElement>(null);
  const mv = useMotionValue(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const write = (v: number) => {
      el.textContent = `${v.toFixed(decimals)}${suffix}`;
    };
    if (reduce) {
      write(value);
      return;
    }
    const unsub = mv.on("change", write);
    const c = animate(mv, value, { duration: 0.9, ease: "easeOut" });
    return () => {
      c.stop();
      unsub();
    };
  }, [value, decimals, suffix, reduce, mv]);

  return (
    <span ref={ref} className={className}>
      {(0).toFixed(decimals)}
      {suffix}
    </span>
  );
}
