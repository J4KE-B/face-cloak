"use client";

import { useId, useRef, useState, type DragEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ImagePlus, Loader2, X } from "lucide-react";
import { cn } from "@/lib/cn";
import type { PreparedImage } from "@/lib/image";

/** Animated drag-and-drop zone with preview. The hidden input stays keyboard-focusable. */
export function FileDrop({
  label,
  hint,
  value,
  busy,
  onFile,
  onClear,
  optional,
}: {
  label: string;
  hint: string;
  value: PreparedImage | null;
  busy: boolean;
  onFile: (f: File) => void;
  onClear: () => void;
  optional?: boolean;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  };

  return (
    <div>
      <p className="mb-1.5 flex items-center gap-2 text-sm font-medium">
        {label}
        {optional && <span className="rounded-full border border-line px-2 py-0.5 text-[11px] font-normal text-muted">optional</span>}
      </p>
      <label
        htmlFor={id}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={cn(
          "group relative flex min-h-48 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed p-4 text-center transition-colors focus-within:border-brand2 focus-within:ring-2 focus-within:ring-brand2/40",
          over ? "border-brand2 bg-brand2/10" : "border-line bg-white/[0.03] hover:border-brand/60 hover:bg-white/[0.06]",
        )}
      >
        <input
          ref={input}
          id={id}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />
        <AnimatePresence mode="wait">
          {busy ? (
            <motion.div key="busy" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-2 text-muted">
              <Loader2 className="size-7 animate-spin text-brand2" aria-hidden />
              <span className="text-sm">Preparing image...</span>
            </motion.div>
          ) : value ? (
            <motion.div key="img" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={value.previewUrl} alt={`${label}, preview`} className="max-h-56 rounded-lg object-contain" />
              <span className="text-xs text-muted">
                {value.width}x{value.height} px, JPEG, metadata removed. Click to replace.
              </span>
            </motion.div>
          ) : (
            <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center gap-2">
              <motion.span animate={over ? { scale: 1.2, y: -4 } : { scale: 1, y: 0 }} className="grid size-12 place-items-center rounded-2xl bg-brand/15 text-brand">
                <ImagePlus className="size-6" aria-hidden />
              </motion.span>
              <span className="text-sm font-medium">{over ? "Drop it here" : "Drag a photo here, or click to choose"}</span>
              <span className="max-w-xs text-xs text-muted">{hint}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </label>
      {value && (
        <button type="button" onClick={onClear} className="mt-2 inline-flex items-center gap-1 text-xs text-muted hover:text-fg">
          <X className="size-3.5" aria-hidden /> Remove
        </button>
      )}
    </div>
  );
}
