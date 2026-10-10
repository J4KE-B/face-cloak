"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

/** Scroll-reveal wrapper. */
export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.55, delay, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}

export function SectionHeading({ eyebrow, title, children, className }: { eyebrow?: string; title: string; children?: ReactNode; className?: string }) {
  return (
    <Reveal className={cn("mb-8 max-w-2xl", className)}>
      {eyebrow && <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-brand2">{eyebrow}</p>}
      <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">{title}</h2>
      {children && <p className="mt-3 text-pretty text-muted">{children}</p>}
    </Reveal>
  );
}

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="relative mx-auto max-w-6xl px-4 pt-28 pb-8 sm:px-6">
      <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-brand2">{eyebrow}</p>
      <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl">
        <span className="text-gradient">{title}</span>
      </h1>
      {children && <p className="mt-4 max-w-2xl text-pretty text-lg text-muted">{children}</p>}
    </header>
  );
}
