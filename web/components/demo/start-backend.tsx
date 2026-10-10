"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Check, ChevronDown, Copy, ExternalLink, Laptop, Rocket } from "lucide-react";
import { ShimmerLink } from "@/components/ui/shimmer-button";
import { KAGGLE_DEMO_NOTEBOOK_URL, KAGGLE_NOTEBOOK_GITHUB_URL, KAGGLE_NOTEBOOK_RAW_URL, LOCAL_COMMAND } from "@/lib/config";

const GUIDE = [
  <>Open the notebook on Kaggle (button below) and copy it to your own account.</>,
  <>In the settings choose <strong>GPU T4</strong>, switch <strong>Internet on</strong>, and add a Kaggle secret named <code className="font-mono">DEMO_KEY</code> (12+ characters) as your passcode.</>,
  <><strong>Run all</strong>. Wait for the green &quot;ready&quot; badge, then paste the tunnel URL it printed into the field above. Only use a link or address from a notebook or script you started yourself: anyone can create a trycloudflare.com address.</>,
];

export function StartBackend() {
  const [copied, setCopied] = useState(false);
  const [open, setOpen] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(LOCAL_COMMAND);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard blocked */
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="card border-brand/30 p-5 sm:p-6"
      aria-labelledby="start-h"
    >
      <h2 id="start-h" className="flex items-center gap-2 text-lg font-semibold">
        <Rocket className="size-5 text-brand" aria-hidden /> Start the backend
      </h2>
      <p className="mt-1 text-sm text-muted">
        The face-swap models run on a Python server, not on this website. Start one, then connect above.
      </p>

      <ol className="mt-4 space-y-3">
        {GUIDE.map((g, i) => (
          <li key={i} className="flex gap-3 text-sm">
            <span className="grid size-6 shrink-0 place-items-center rounded-full bg-brand/20 text-xs font-bold text-brand">{i + 1}</span>
            <span className="pt-0.5 text-muted [&_strong]:text-fg">{g}</span>
          </li>
        ))}
      </ol>

      <div className="mt-5 flex flex-wrap gap-3">
        <ShimmerLink href={KAGGLE_DEMO_NOTEBOOK_URL} external>
          <ExternalLink className="size-4" aria-hidden /> Open the demo notebook on Kaggle
        </ShimmerLink>
        <ShimmerLink href={KAGGLE_NOTEBOOK_GITHUB_URL} variant="ghost" external>
          Notebook on GitHub
        </ShimmerLink>
        <a href={KAGGLE_NOTEBOOK_RAW_URL} target="_blank" rel="noopener noreferrer" className="self-center text-xs text-muted underline hover:text-fg">
          raw .ipynb
        </a>
      </div>

      <div className="mt-6 border-t border-line pt-4">
        <button
          type="button"
          aria-expanded={open}
          aria-controls="laptop-help"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-between rounded-lg py-1 text-left text-sm font-medium"
        >
          <span className="flex items-center gap-2"><Laptop className="size-4" aria-hidden /> Use my laptop instead</span>
          <ChevronDown className={`size-4 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
        </button>
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              id="laptop-help"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <p className="mt-3 text-sm text-muted">
                From the repository root, run this. It starts the CPU backend and a Cloudflare quick tunnel, then prints
                the URL and a generated passcode. CPU is slower; lower the steps.
              </p>
              <div className="mt-3 flex items-center gap-2 rounded-xl border border-line bg-black/40 p-2 pl-4">
                <code className="flex-1 overflow-x-auto font-mono text-sm">{LOCAL_COMMAND}</code>
                <button
                  type="button"
                  onClick={copy}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium hover:bg-white/15"
                  aria-label="Copy command"
                >
                  {copied ? <Check className="size-4 text-ok" aria-hidden /> : <Copy className="size-4" aria-hidden />}
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <p className="mt-3 text-xs text-muted">
                For a purely local test you can instead open the page on http://localhost and use{" "}
                <code className="font-mono">http://127.0.0.1:8000</code>. Safari blocks http from https pages, so prefer the tunnel there.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.section>
  );
}
