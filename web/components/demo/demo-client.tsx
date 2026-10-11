"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import Link from "next/link";
import { useReducedMotion } from "framer-motion";
import { AlertTriangle, ShieldCheck, Sparkles } from "lucide-react";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { PageHeader } from "@/components/ui/section";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { useToast } from "@/components/toast";
import { prepareImage, ImageError, type PreparedImage } from "@/lib/image";
import type { Example } from "@/lib/examples";
import { hostOf } from "@/lib/backend";
import type { Mode } from "@/lib/types";
import { BackendPanel } from "./backend-panel";
import { ExamplePicker } from "./example-picker";
import { Controls } from "./controls";
import { FileDrop } from "./file-drop";
import { ResultsSkeleton, ResultsView } from "./results-view";
import { RunProgress } from "./run-progress";
import { StartBackend } from "./start-backend";
import { TargetPicker } from "./target-picker";
import { useBackend } from "./use-backend";
import { useJob } from "./use-job";

const BUILTIN_CONSENT =
  "Only upload photos of yourself or people who consented. Your photo is sent to the backend you connected to above, which is meant to process it in memory and delete it within 10 minutes.";

function Step({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <section className="card p-5 sm:p-6" aria-labelledby={`step-${n}`}>
      <h2 id={`step-${n}`} className="mb-4 flex items-center gap-3 text-lg font-semibold">
        <span className="grid size-7 place-items-center rounded-full bg-brand/20 text-sm font-bold text-brand">{n}</span>
        {title}
      </h2>
      {children}
    </section>
  );
}

export function DemoClient() {
  const { toast } = useToast();
  const b = useBackend();
  const job = useJob(b.conn);

  const [photo, setPhoto] = useState<PreparedImage | null>(null);
  const [reference, setReference] = useState<PreparedImage | null>(null);
  const [busy, setBusy] = useState<"photo" | "reference" | null>(null);
  const [targetId, setTargetId] = useState<string | null>(null);
  const [epsV, setEps] = useState<number | null>(null);
  const [stepsV, setSteps] = useState<number | null>(null);
  const [mode, setMode] = useState<Mode>("fast");
  const [consent, setConsent] = useState(false);
  const [linkAsk, setLinkAsk] = useState(false);
  const [exampleId, setExampleId] = useState<string | null>(null);
  const [exampleLoading, setExampleLoading] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const photoRef = useRef<PreparedImage | null>(null);
  const refRef = useRef<PreparedImage | null>(null);

  const h = b.health;
  const eps = Math.min(epsV ?? 8, h?.max_eps ?? 16);
  const steps = Math.min(stepsV ?? h?.default_steps ?? 40, h?.max_steps ?? 60);
  // a remembered / example-recommended id only counts if this backend actually offers it
  const target = (targetId && b.targets.some((t) => t.id === targetId) ? targetId : b.targets[0]?.id) ?? null;
  const online = b.phase === "online";
  const running = job.phase === "submitting" || job.phase === "active";

  const commit = useCallback((which: "photo" | "reference", img: PreparedImage | null) => {
    const slot = which === "photo" ? photoRef : refRef;
    if (slot.current) URL.revokeObjectURL(slot.current.previewUrl);
    slot.current = img;
    (which === "photo" ? setPhoto : setReference)(img);
  }, []);

  const pick = useCallback(
    async (f: File, which: "photo" | "reference") => {
      setBusy(which);
      try {
        const img = await prepareImage(f);
        if (exampleId) {
          // leaving an example: its reference photo belongs to the example person, not to this upload
          if (which === "photo") commit("reference", null);
          setExampleId(null);
        }
        commit(which, img);
      } catch (e) {
        toast({ kind: "error", title: "Cannot use that file", message: e instanceof ImageError ? e.message : "The image could not be read." });
      } finally {
        setBusy(null);
      }
    },
    [toast, commit, exampleId],
  );

  const clear = (which: "photo" | "reference") => {
    if (exampleId) setExampleId(null);
    commit(which, null);
  };

  const loadExample = async (ex: Example) => {
    setExampleLoading(ex.id);
    try {
      const get = async (url: string, name: string) => {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return prepareImage(new File([await r.blob()], name, { type: "image/jpeg" }));
      };
      // prepare both before touching the form, so a failure never leaves it half-switched
      const [a, ref] = await Promise.all([get(ex.photo, `${ex.id}_a.jpg`), get(ex.reference, `${ex.id}_b.jpg`)]);
      commit("photo", a);
      commit("reference", ref);
      setTargetId(ex.target);
      setExampleId(ex.id);
    } catch {
      toast({ kind: "error", title: "Example unavailable", message: "The example photos could not be loaded." });
    } finally {
      setExampleLoading(null);
    }
  };
  useEffect(
    () => () => {
      if (photoRef.current) URL.revokeObjectURL(photoRef.current.previewUrl);
      if (refRef.current) URL.revokeObjectURL(refRef.current.previewUrl);
    },
    [],
  );

  const missing: string[] = [];
  if (!online) missing.push("connect to a running backend");
  if (!photo) missing.push("add your photo");
  if (!target) missing.push("choose a target");
  if (!consent) missing.push("tick the consent box");
  const ready = missing.length === 0;

  const launch = () => {
    if (!ready || !photo || !target) return;
    setConsent(false); // consent is per run: it has to be given again for the next one
    setLinkAsk(false);
    void job.start({ photo: photo.blob, reference: reference?.blob ?? null, target_id: target, eps, steps, mode });
  };
  const run = () => {
    if (!ready) return;
    if (b.conn?.viaLink) {
      setLinkAsk(true); // first upload to a host that came from a link needs a second, explicit confirmation
      return;
    }
    launch();
  };
  const askOpen = linkAsk && !!b.conn?.viaLink;

  // Move focus + scroll to the progress / results heading when a run starts or finishes.
  useEffect(() => {
    if (job.phase !== "submitting" && job.phase !== "done" && job.phase !== "failed") return;
    const el = resultRef.current;
    if (!el) return;
    el.querySelector<HTMLElement>("[data-focus-target]")?.focus({ preventScroll: true });
    el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }, [job.phase, reduce]);

  return (
    <>
      <div className="relative isolate overflow-hidden">
        <AuroraBackground className="opacity-70" />
        <PageHeader eyebrow="Live demo" title="Cloak a photo, test it against a deepfake">
          Upload a photo of yourself. We cloak it, run a real face-swap attack on both versions, and let an independent
          model judge whether the swap still looks like you.
          <span className="mt-3 block text-sm">
            Attacker: inswapper. Its identity encoder (ArcFace w600k_r50) is the model the cloak is optimised against, so
            this live demo is the white-box best case. Black-box transfer (SimSwap) is reported on{" "}
            <Link href="/results" className="underline hover:text-fg">Results</Link>.
          </span>
        </PageHeader>
      </div>

      <div className="mx-auto grid max-w-4xl gap-5 px-4 pb-10 sm:px-6">
        <BackendPanel b={b} />
        {(b.phase === "offline" || b.phase === "idle" || b.phase === "checking") && b.hydrated && <StartBackend />}

        <Step n={1} title="Your photo">
          <ExamplePicker selected={exampleId} loading={exampleLoading} disabled={running || busy !== null} onSelect={(ex) => void loadExample(ex)} />
          <div className="grid gap-5 md:grid-cols-2">
            <FileDrop
              label="Your photo"
              hint="One clear, front-facing face. Resized to 1024 px and re-encoded in your browser; location data is removed."
              value={photo}
              busy={busy === "photo" || exampleLoading !== null}
              onFile={(f) => void pick(f, "photo")}
              onClear={() => clear("photo")}
            />
            <FileDrop
              label="Second photo of you (reference)"
              optional
              hint="A different photo of the same person. It makes the verdict honest: the judge compares the swap against this photo, not the one we cloaked."
              value={reference}
              busy={busy === "reference" || exampleLoading !== null}
              onFile={(f) => void pick(f, "reference")}
              onClear={() => clear("reference")}
            />
          </div>
          <label className="mt-5 flex cursor-pointer items-start gap-3 rounded-xl border border-line bg-white/[0.03] p-4 text-sm">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-0.5 size-4 shrink-0 accent-[var(--color-brand)]" />
            <span>
              <strong>I agree. </strong>
              {BUILTIN_CONSENT}
            </span>
          </label>
          {h?.consent_notice && b.conn && (
            <p className="mt-2 text-xs text-muted">
              <strong className="text-fg">Message from backend {hostOf(b.conn.url)}:</strong> {h.consent_notice}
            </p>
          )}
        </Step>

        <Step n={2} title="Pick the attacker's target portrait">
          <p className="mb-4 text-sm text-muted">Your face will be swapped onto this public-domain portrait. These are fixed; you cannot upload targets.</p>
          <TargetPicker targets={b.targets} state={b.targetsState} value={target} onChange={setTargetId} />
        </Step>

        <Step n={3} title="Settings">
          <Controls health={h} eps={eps} steps={steps} mode={mode} onEps={setEps} onSteps={setSteps} onMode={setMode} />
        </Step>

        <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center">
          <ShimmerButton type="button" onClick={run} disabled={!ready || running} className="px-8 py-4 text-base">
            <Sparkles className="size-5" aria-hidden /> Cloak and attack
          </ShimmerButton>
          {!ready && !running && <p className="text-sm text-muted">To start: {missing.join(", ")}.</p>}
        </div>
        {askOpen && b.conn && (
          <div role="alertdialog" aria-labelledby="link-confirm-h" className="card border-warn/50 p-5">
            <h2 id="link-confirm-h" className="flex items-center gap-2 font-semibold text-warn">
              <AlertTriangle className="size-5" aria-hidden /> Confirm the backend before your first upload
            </h2>
            <p className="mt-2 text-sm">
              This backend address came from a link: <strong className="font-mono">{hostOf(b.conn.url)}</strong>. Anyone can
              create a trycloudflare.com address, and whoever runs this one receives your photo and passcode. Continue only
              if you started the notebook or script behind it yourself.
            </p>
            <div className="mt-4 flex flex-wrap gap-3">
              <ShimmerButton
                type="button"
                onClick={() => {
                  b.confirmLink();
                  launch();
                }}
              >
                Yes, I started it. Send my photo
              </ShimmerButton>
              <ShimmerButton type="button" variant="ghost" onClick={() => setLinkAsk(false)}>Cancel</ShimmerButton>
            </div>
          </div>
        )}
      </div>

      <div ref={resultRef} className="mx-auto max-w-4xl scroll-mt-24 px-4 pb-10 sm:px-6">
        {running && (
          <div className="space-y-4">
            <RunProgress status={job.status} submitting={job.phase === "submitting"} startedAt={job.startedAt} onCancel={() => void job.cancel()} />
            <ResultsSkeleton />
          </div>
        )}
        {job.phase === "done" && job.result && <ResultsView r={job.result} onReset={job.reset} />}
        {job.phase === "failed" && (
          <div role="alert" className="card border-bad/40 p-6">
            <h2 tabIndex={-1} data-focus-target className="flex items-center gap-2 text-lg font-semibold text-bad outline-none"><ShieldCheck className="size-5" aria-hidden /> {job.failure?.title ?? "The run failed"}</h2>
            <p className="mt-2 text-sm text-muted">{job.failure?.message}</p>
            <div className="mt-4"><ShimmerButton type="button" variant="ghost" onClick={job.reset}>Dismiss</ShimmerButton></div>
          </div>
        )}
      </div>
    </>
  );
}
