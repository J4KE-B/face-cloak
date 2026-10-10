"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Download, RotateCcw, ShieldOff, XCircle } from "lucide-react";
import { ComparisonSlider } from "@/components/ui/comparison-slider";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { NumberTicker } from "@/components/ui/number-ticker";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cn";
import { num } from "@/lib/format";
import type { JobResult } from "@/lib/types";

const item = { hidden: { opacity: 0, y: 24 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.2, 0.7, 0.2, 1] as const } } };

function Verdict({ r }: { r: JobResult }) {
  const v =
    r.protected === true
      ? { t: "Protected", Icon: CheckCircle2, c: "border-ok/50 bg-ok/10 text-ok", glow: "shadow-[0_0_60px_-10px_rgb(52_211_153/0.6)]" }
      : r.protected === false
        ? { t: "Not protected", Icon: XCircle, c: "border-bad/50 bg-bad/10 text-bad", glow: "shadow-[0_0_60px_-10px_rgb(251_113_133/0.6)]" }
        : { t: "Inconclusive", Icon: AlertTriangle, c: "border-warn/50 bg-warn/10 text-warn", glow: "shadow-[0_0_60px_-10px_rgb(251_191_36/0.5)]" };
  return (
    <motion.div
      initial={{ scale: 0.7, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 18 }}
      role="status"
      className={cn("flex flex-col items-start gap-3 rounded-2xl border p-5 sm:flex-row sm:items-center", v.c, v.glow)}
    >
      <v.Icon className="size-12 shrink-0" aria-hidden />
      <div>
        <p className="text-3xl font-extrabold tracking-tight">{v.t}</p>
        {r.note && <p className="mt-1 text-sm text-fg/80">{r.note}</p>}
      </div>
    </motion.div>
  );
}

function ScoreBar({ label, value, tau }: { label: string; value: number | null; tau: number | null }) {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  // Below tau the judge says "someone else" (green); at or above tau it says "still you" (red).
  const tone = value == null || tau == null ? "neutral" : value < tau ? "ok" : "bad";
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between text-sm">
        <span>{label}</span>
        <span className="font-mono">{value == null ? "n/a" : <NumberTicker value={value} decimals={3} />}</span>
      </div>
      <div className="relative h-4 rounded-full bg-white/10">
        <motion.div
          className={cn(
            "h-full rounded-full",
            tone === "ok" ? "bg-gradient-to-r from-emerald-500 to-emerald-300" : tone === "bad" ? "bg-gradient-to-r from-rose-600 to-rose-400" : "bg-white/30",
          )}
          initial={{ width: 0 }}
          animate={{ width: `${clamp(value ?? 0) * 100}%` }}
          transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
        />
        {tau != null && (
          <div className="absolute -top-1.5 -bottom-1.5 w-0.5 bg-white" style={{ left: `${clamp(tau) * 100}%` }}>
            <span className="absolute -top-5 left-1/2 -translate-x-1/2 font-mono text-[11px] whitespace-nowrap text-white/80">τ {tau.toFixed(2)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function SwapCard({ title, src, kind, missing, good }: { title: string; src: string | null; kind: "ok" | "bad"; missing: string; good?: boolean }) {
  return (
    <figure className={cn("card overflow-hidden", kind === "ok" ? "border-bad/30" : "border-ok/30")}>
      <div className="flex min-h-48 items-center justify-center bg-black/40">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt={`${title} (synthetic face swap)`} className="max-h-96 w-full object-contain" />
        ) : (
          <div className="p-6 text-center text-sm text-muted">
            <ShieldOff className={cn("mx-auto mb-2 size-8", good ? "text-ok" : "text-muted")} aria-hidden />
            {missing}
          </div>
        )}
      </div>
      <figcaption className="p-3 text-sm font-semibold">{title}</figcaption>
    </figure>
  );
}

export function ResultsSkeleton() {
  return (
    <div className="space-y-4" aria-hidden>
      <Skeleton className="h-28" />
      <div className="grid gap-4 md:grid-cols-2"><Skeleton className="h-72" /><Skeleton className="h-72" /></div>
    </div>
  );
}

export function ResultsView({ r, onReset }: { r: JobResult; onReset: () => void }) {
  const { scores: s, quality: q } = r;
  const cleanOk = s.tau == null || (s.id_sim_original != null && s.id_sim_original >= s.tau);
  const download = r.cloaked_download ?? r.cloaked;
  const cloakedMissing =
    r.protected === true
      ? "The attacker could not detect a face in the cloaked photo, so no swap was possible."
      : "No swap image was returned for the cloaked photo.";
  return (
    <motion.div initial="hidden" animate="show" transition={{ staggerChildren: 0.12 }} className="space-y-6" aria-label="Results">
      <h2 tabIndex={-1} data-focus-target className="sr-only">Your results</h2>
      <motion.div variants={item}><Verdict r={r} /></motion.div>

      <motion.section variants={item} className="card p-4 sm:p-6" aria-labelledby="cmp-h">
        <h3 id="cmp-h" className="text-lg font-semibold">Can you see the difference?</h3>
        <p className="mb-4 text-sm text-muted">Drag the handle (or use the arrow keys). The right-hand panel shows the perturbation amplified ×10.</p>
        <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
          {r.original && r.cloaked ? (
            <ComparisonSlider before={r.original} after={r.cloaked} alt="photo" />
          ) : (
            <p className="rounded-xl border border-dashed border-line p-6 text-sm text-muted">The backend did not return a usable preview image.</p>
          )}
          {r.perturbation && (
            <figure>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={r.perturbation} alt="The perturbation, amplified ten times" className="max-h-[70vh] w-full rounded-xl border border-line bg-black object-contain" />
              <figcaption className="mt-2 text-xs text-muted">
                Perturbation ×10.{q.linf_255 != null && <> The largest real change is {q.linf_255}/255 on any colour channel of any pixel.</>}
              </figcaption>
            </figure>
          )}
        </div>
      </motion.section>

      <motion.section variants={item} aria-labelledby="swap-h">
        <h3 id="swap-h" className="mb-1 text-lg font-semibold">What the attacker gets</h3>
        <p className="mb-2 text-sm text-muted">Both swaps use the same target portrait and the same tool. Images carry a SYNTHETIC watermark from the backend.</p>
        <p className="mb-4 text-sm text-muted">
          Attacker: inswapper. Its identity encoder (ArcFace w600k_r50) is the model the cloak is optimised against, so this
          live demo is the white-box best case. Black-box transfer (SimSwap) is reported on Results.
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          <SwapCard title="Deepfake from original photo" src={r.swap_original} kind="ok" missing="Swap unavailable for the original photo." />
          <SwapCard title="Deepfake from cloaked photo" src={r.swap_cloaked} kind="bad" missing={cloakedMissing} good={r.protected === true} />
        </div>
      </motion.section>

      <motion.section variants={item} className="card p-5 sm:p-6" aria-labelledby="id-h">
        <h3 id="id-h" className="text-lg font-semibold">Identity score (judge: {s.judge})</h3>
        <p className="mt-1 mb-8 text-sm text-muted">
          Cosine similarity between the swapped face and {s.reference === "uploaded" ? "your second reference photo" : "your original photo (no reference uploaded, so this is the self-reference, which flatters the clean swap)"}.
          The white line is the threshold τ at a 1% false-accept rate: <strong className="text-fg">at or above τ the judge says &quot;still you&quot; (red)</strong>, below it says &quot;someone else&quot; (green).
        </p>
        <div className="space-y-9">
          <ScoreBar label="Swap from original photo" value={s.id_sim_original} tau={s.tau} />
          <ScoreBar label="Swap from cloaked photo" value={s.id_sim_cloaked} tau={s.tau} />
        </div>
        {!cleanOk && !r.note && r.protected === null && <p className="mt-6 text-xs text-warn">The swap from the original photo itself scored below τ, so this run cannot prove much either way.</p>}
      </motion.section>

      <motion.section variants={item} aria-labelledby="q-h">
        <h3 id="q-h" className="mb-3 text-lg font-semibold">Image quality (cloaked vs original)</h3>
        <dl className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[
            { k: "PSNR", v: `${num(q.psnr, 1)} dB`, d: "higher is better; above 35 is hard to see" },
            { k: "SSIM", v: num(q.ssim, 3), d: "1.0 = identical" },
            { k: "LPIPS", v: num(q.lpips, 3), d: "perceptual distance; lower is better" },
            { k: "L∞", v: q.linf_255 != null ? `${q.linf_255}/255` : "n/a", d: "largest per-pixel change" },
          ].map((m) => (
            <div key={m.k} className="card p-4">
              <dt className="text-xs font-semibold tracking-wider text-muted uppercase">{m.k}</dt>
              <dd className="mt-1 text-2xl font-bold tabular-nums">{m.v}</dd>
              <p className="mt-1 text-xs text-muted">{m.d}</p>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-xs text-muted">
          Settings: ε {r.params.eps}/255, {r.params.steps} steps, {r.params.mode} mode. Cloaking took {num(r.timings.cloak_s, 1)} s, whole job {num(r.timings.total_s, 1)} s.
        </p>
      </motion.section>

      <motion.div variants={item} className="flex flex-wrap gap-3">
        {download && (
          <a
            href={download}
            download={download.startsWith("data:image/jpeg") ? "cloaked.jpg" : "cloaked.png"}
            className="relative inline-flex items-center justify-center gap-2 rounded-xl bg-brand px-5 py-3 text-sm font-semibold text-white shadow-[0_0_30px_-5px_rgb(124_92_255/0.7)] hover:bg-[#8d70ff]"
          >
            <Download className="size-4" aria-hidden /> Download cloaked image
          </a>
        )}
        <ShimmerButton type="button" variant="ghost" onClick={onReset}>
          <RotateCcw className="size-4" aria-hidden /> Try again
        </ShimmerButton>
      </motion.div>
    </motion.div>
  );
}
