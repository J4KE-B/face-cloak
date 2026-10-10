"use client";

import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

type Variant = "primary" | "ghost";

const base =
  "relative inline-flex items-center justify-center gap-2 overflow-hidden rounded-xl px-5 py-3 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50";
const variants: Record<Variant, string> = {
  primary: "bg-brand text-white shadow-[0_0_30px_-5px_rgb(124_92_255/0.7)] hover:bg-[#8d70ff]",
  ghost: "border border-line bg-white/5 text-fg hover:bg-white/10",
};

function Sheen() {
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent transition-transform duration-700 group-hover/btn:translate-x-full"
    />
  );
}

export function ShimmerButton({
  variant = "primary",
  className,
  children,
  ...p
}: { variant?: Variant; children: ReactNode } & ComponentProps<"button">) {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      className={cn("group/btn", base, variants[variant], className)}
      {...(p as object)}
    >
      <Sheen />
      <span className="relative inline-flex items-center gap-2">{children}</span>
    </motion.button>
  );
}

export function ShimmerLink({
  variant = "primary",
  className,
  children,
  href,
  external,
}: {
  variant?: Variant;
  className?: string;
  children: ReactNode;
  href: string;
  external?: boolean;
}) {
  const cls = cn("group/btn", base, variants[variant], className);
  const inner = (
    <>
      <Sheen />
      <span className="relative inline-flex items-center gap-2">{children}</span>
    </>
  );
  return external ? (
    <a href={href} target="_blank" rel="noopener noreferrer" className={cls}>
      {inner}
    </a>
  ) : (
    <Link href={href} className={cls}>
      {inner}
    </Link>
  );
}
