"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react";
import { cn } from "@/lib/cn";

type Kind = "error" | "success" | "info";
interface ToastItem {
  id: number;
  kind: Kind;
  title: string;
  message?: string;
  retryAfter?: number | null;
}
interface Ctx {
  toast: (t: Omit<ToastItem, "id" | "kind"> & { kind?: Kind }) => void;
}

const ToastCtx = createContext<Ctx>({ toast: () => {} });
export const useToast = () => useContext(ToastCtx);

let nextId = 1;

function Countdown({ from }: { from: number }) {
  const [left, setLeft] = useState(from);
  useEffect(() => {
    const t = setInterval(() => setLeft((l) => Math.max(0, l - 1)), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <span className="mt-1 block text-xs font-medium text-warn" role="timer">
      {left > 0 ? `You can retry in ${left} s` : "You can retry now"}
    </span>
  );
}

function Toast({ t, onClose }: { t: ToastItem; onClose: (id: number) => void }) {
  const { id, kind, retryAfter } = t;
  // Depends only on this toast's own fields and a stable onClose, so other toasts arriving never reset the timer.
  useEffect(() => {
    const ms = retryAfter ? Math.min(60, retryAfter) * 1000 + 3000 : kind === "error" ? 9000 : 5000;
    const h = setTimeout(() => onClose(id), ms);
    return () => clearTimeout(h);
  }, [id, kind, retryAfter, onClose]);
  const Icon = t.kind === "error" ? AlertTriangle : t.kind === "success" ? CheckCircle2 : Info;
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 40 }}
      className={cn(
        "pointer-events-auto flex w-full gap-3 rounded-xl border bg-surface/95 p-4 shadow-2xl backdrop-blur",
        t.kind === "error" ? "border-bad/40" : t.kind === "success" ? "border-ok/40" : "border-line",
      )}
    >
      <Icon className={cn("mt-0.5 size-5 shrink-0", t.kind === "error" ? "text-bad" : t.kind === "success" ? "text-ok" : "text-brand2")} aria-hidden />
      <div className="min-w-0 flex-1 text-sm">
        <p className="font-semibold">{t.title}</p>
        {t.message && <p className="mt-0.5 text-muted">{t.message}</p>}
        {t.retryAfter ? <Countdown from={t.retryAfter} /> : null}
      </div>
      <button onClick={() => onClose(id)} aria-label="Dismiss notification" className="h-fit rounded p-1 text-muted hover:text-fg">
        <X className="size-4" />
      </button>
    </motion.div>
  );
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const toast = useCallback<Ctx["toast"]>((t) => {
    setItems((s) => [...s.slice(-3), { ...t, kind: t.kind ?? "info", id: nextId++ }]);
  }, []);
  const close = useCallback((id: number) => setItems((s) => s.filter((x) => x.id !== id)), []);
  const value = useMemo(() => ({ toast }), [toast]);
  return (
    <ToastCtx.Provider value={value}>
      {children}
      <div
        role="region"
        aria-label="Notifications"
        aria-live="polite"
        className="pointer-events-none fixed inset-x-0 bottom-0 z-[100] mx-auto flex max-w-sm flex-col gap-2 p-4 sm:right-0 sm:left-auto"
      >
        <AnimatePresence>
          {items.map((t) => (
            <Toast key={t.id} t={t} onClose={close} />
          ))}
        </AnimatePresence>
      </div>
    </ToastCtx.Provider>
  );
}
