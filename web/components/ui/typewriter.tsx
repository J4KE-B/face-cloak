"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "framer-motion";

/** Rotating typewriter line. Static (first phrase) under reduced motion. */
export function Typewriter({ phrases, className }: { phrases: string[]; className?: string }) {
  const reduce = useReducedMotion();
  const [i, setI] = useState(0);
  const [n, setN] = useState(0);
  const [del, setDel] = useState(false);

  useEffect(() => {
    if (reduce) return;
    const full = phrases[i % phrases.length] ?? "";
    const t = setTimeout(
      () => {
        if (!del && n < full.length) setN(n + 1);
        else if (!del) setDel(true);
        else if (n > 0) setN(n - 1);
        else {
          setDel(false);
          setI(i + 1);
        }
      },
      !del && n === full.length ? 1600 : del ? 22 : 45,
    );
    return () => clearTimeout(t);
  }, [n, del, i, phrases, reduce]);

  if (reduce) return <span className={className}>{phrases[0]}</span>;
  const shown = (phrases[i % phrases.length] ?? "").slice(0, n);
  return (
    <span className={className} aria-label={phrases.join(". ")}>
      <span aria-hidden>
        {shown}
        <span className="ml-0.5 inline-block h-[1em] w-[2px] translate-y-[0.15em] animate-pulse bg-brand2" />
      </span>
    </span>
  );
}
