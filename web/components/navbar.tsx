"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Menu, ShieldCheck, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { REPO_URL } from "@/lib/config";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Live demo" },
  { href: "/results", label: "Results" },
  { href: "/about", label: "About" },
];

function GithubMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden className={className} fill="currentColor">
      <path d="M12 .5A11.5 11.5 0 0 0 .5 12c0 5.080 3.290 9.390 7.860 10.910.58.110.79-.25.79-.56v-2c-3.200.7-3.870-1.540-3.870-1.540-.52-1.330-1.280-1.690-1.280-1.690-1.040-.71.080-.7.080-.7 1.150.08 1.760 1.190 1.760 1.190 1.030 1.760 2.700 1.250 3.360.96.100-.75.400-1.250.730-1.540-2.550-.29-5.240-1.280-5.240-5.690 0-1.260.45-2.280 1.190-3.090-.12-.29-.52-1.460.11-3.050 0 0 .97-.31 3.170 1.180a11 11 0 0 1 5.780 0c2.200-1.490 3.170-1.180 3.170-1.180.63 1.590.23 2.760.11 3.050.74.810 1.190 1.830 1.190 3.090 0 4.420-2.700 5.390-5.270 5.670.41.360.78 1.060.78 2.140v3.170c0 .31.210.68.800.56A11.500 11.500 0 0 0 23.500 12 11.500 11.500 0 0 0 12 .5Z" />
    </svg>
  );
}

export function Navbar() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <motion.nav
        initial={{ y: -40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: [0.2, 0.7, 0.2, 1] }}
        aria-label="Main"
        className="mx-auto mt-3 flex max-w-5xl items-center justify-between rounded-2xl border border-line bg-bg/70 px-4 py-2.5 backdrop-blur-xl max-sm:mx-3"
      >
        <Link href="/" className="flex items-center gap-2 font-bold tracking-tight" onClick={() => setOpen(false)}>
          <ShieldCheck className="size-5 text-brand2" aria-hidden />
          face-cloak
        </Link>
        <ul className="hidden items-center gap-1 sm:flex">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <li key={l.href} className="relative">
                <Link
                  href={l.href}
                  aria-current={active ? "page" : undefined}
                  className={cn("relative z-10 block rounded-lg px-3 py-1.5 text-sm transition-colors", active ? "text-fg" : "text-muted hover:text-fg")}
                >
                  {l.label}
                </Link>
                {active && (
                  <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-lg bg-white/10" transition={{ type: "spring", stiffness: 400, damping: 32 }} />
                )}
              </li>
            );
          })}
        </ul>
        <div className="flex items-center gap-1">
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub repository"
            className="rounded-lg p-2 text-muted transition-colors hover:text-fg"
          >
            <GithubMark className="size-5" />
          </a>
          <button
            className="rounded-lg p-2 text-muted hover:text-fg sm:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            aria-controls="mobile-menu"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </motion.nav>
      <AnimatePresence>
        {open && (
          <motion.ul
            id="mobile-menu"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="mx-3 mt-2 rounded-2xl border border-line bg-bg/95 p-2 backdrop-blur-xl sm:hidden"
          >
            {LINKS.map((l) => (
              <li key={l.href}>
                <Link href={l.href} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-3 text-sm hover:bg-white/10">
                  {l.label}
                </Link>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </header>
  );
}
