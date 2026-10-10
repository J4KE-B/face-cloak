"use client";

import { useState, type FormEvent } from "react";
import { AlertTriangle, Link2, Lock, LockOpen, Plug, Trash2 } from "lucide-react";
import { ShimmerButton } from "@/components/ui/shimmer-button";
import { hostOf, MOCK_URL, normalizeBackendUrl } from "@/lib/backend";
import { cn } from "@/lib/cn";
import { HealthBadge } from "./health-badge";
import type { BackendApi } from "./use-backend";

export function BackendPanel({ b }: { b: BackendApi }) {
  const [reveal, setReveal] = useState(false);
  const norm = b.urlInput.trim() ? normalizeBackendUrl(b.urlInput) : null;
  const invalid = b.urlInput.trim() !== "" && !norm;
  const secure = norm?.startsWith("https://");
  const dirty = !!b.conn && norm !== b.conn.url;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    b.connect();
  };

  return (
    <form onSubmit={submit} className="card p-5 sm:p-6" aria-labelledby="backend-h">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="backend-h" className="flex items-center gap-2 text-lg font-semibold">
          <Plug className="size-5 text-brand2" aria-hidden /> Backend connection
        </h2>
        <HealthBadge phase={b.phase} health={b.health} />
      </div>

      {b.fromLink && (
        <div role="alert" className="mt-4 flex gap-3 rounded-xl border border-warn/40 bg-warn/10 p-3 text-sm">
          <AlertTriangle className="mt-0.5 size-5 shrink-0 text-warn" aria-hidden />
          <p>
            This page was opened with a link to <strong className="font-mono">{hostOf(b.fromLink)}</strong>. Anyone can
            create a trycloudflare.com address, so a link proves nothing. Only continue if you started the notebook or
            script behind it yourself. If you did, enter your passcode and press Connect; the passcode and your photo go
            to that host only, and you will be asked to confirm once more before the first upload.
          </p>
        </div>
      )}

      <div className="mt-5 grid gap-4 md:grid-cols-[1.6fr_1fr]">
        <div>
          <label htmlFor="backend-url" className="mb-1.5 block text-sm font-medium">Backend URL</label>
          <div className="relative">
            <Link2 className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted" aria-hidden />
            <input
              id="backend-url"
              type="text"
              inputMode="url"
              autoComplete="off"
              spellCheck={false}
              placeholder="https://xxxx.trycloudflare.com or http://127.0.0.1:8000"
              value={b.urlInput}
              onChange={(e) => b.setUrlInput(e.target.value)}
              aria-invalid={invalid}
              aria-describedby="backend-url-help"
              className={cn(
                "w-full rounded-xl border bg-black/30 py-2.5 pr-3 pl-9 font-mono text-sm outline-none placeholder:text-muted/60 focus:border-brand2",
                invalid ? "border-bad" : "border-line",
              )}
            />
          </div>
          <p id="backend-url-help" className={cn("mt-1.5 text-xs", invalid ? "text-bad" : "text-muted")}>
            {invalid ? "Not an allowed address. Use a *.trycloudflare.com tunnel, http://localhost:PORT or http://127.0.0.1:PORT." : (
              <>Paste the address printed by the notebook (or <code className="font-mono">mock</code> to try the UI without a backend).</>
            )}
          </p>
        </div>
        <div>
          <label htmlFor="backend-key" className="mb-1.5 block text-sm font-medium">Passcode</label>
          <div className="relative">
            <input
              id="backend-key"
              type={reveal ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              placeholder="DEMO_KEY"
              value={b.keyInput}
              onChange={(e) => b.setKeyInput(e.target.value)}
              className="w-full rounded-xl border border-line bg-black/30 py-2.5 pr-16 pl-3 font-mono text-sm outline-none placeholder:text-muted/60 focus:border-brand2"
            />
            <button
              type="button"
              onClick={() => setReveal((r) => !r)}
              aria-pressed={reveal}
              className="absolute top-1/2 right-2 -translate-y-1/2 rounded-md px-2 py-1 text-xs text-muted hover:text-fg"
            >
              {reveal ? "Hide" : "Show"}
            </button>
          </div>
          <p className="mt-1.5 text-xs text-muted">Sent as a header, never put in a URL. It is cleared whenever you change the backend address.</p>
        </div>
      </div>

      {norm && (
        <p className="mt-4 flex flex-wrap items-center gap-2 text-sm" aria-live="polite">
          {secure ? <Lock className="size-4 text-ok" aria-hidden /> : <LockOpen className="size-4 text-warn" aria-hidden />}
          <span className="text-muted">Photos and passcode will be sent to</span>
          <strong className="rounded-md bg-white/10 px-2 py-0.5 font-mono">{hostOf(norm)}</strong>
          {norm === MOCK_URL && <span className="text-xs text-muted">(nothing leaves your browser)</span>}
          {norm.startsWith("http://") && <span className="text-xs text-muted">(local machine, unencrypted)</span>}
        </p>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <ShimmerButton type="submit" disabled={invalid || !b.urlInput.trim()}>
          <Plug className="size-4" aria-hidden /> {b.conn ? (dirty ? "Switch backend" : "Reconnect") : "Connect"}
        </ShimmerButton>
        {b.conn && (
          <ShimmerButton type="button" variant="ghost" onClick={b.disconnect}>
            <Trash2 className="size-4" aria-hidden /> Forget
          </ShimmerButton>
        )}
        <p className="text-xs text-muted">The address and passcode are remembered in this browser only.</p>
      </div>
    </form>
  );
}
