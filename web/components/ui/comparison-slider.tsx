"use client";

import { useCallback, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { ChevronsLeftRight } from "lucide-react";

/**
 * Before/after image comparison slider (pattern from vengenceui.com's comparison component, re-implemented).
 * Draggable divider with pointer events; Left/Right/Home/End keys for keyboard users.
 */
export function ComparisonSlider({
  before,
  after,
  beforeLabel = "Original",
  afterLabel = "Cloaked",
  alt = "photo",
}: {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
  alt?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  const setFromX = useCallback((x: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || r.width === 0) return;
    setPos(Math.max(0, Math.min(100, ((x - r.left) / r.width) * 100)));
  }, []);

  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
    setFromX(e.clientX);
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragging.current) setFromX(e.clientX);
  };
  const onUp = () => {
    dragging.current = false;
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 10 : 2;
    if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - step));
    else if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + step));
    else if (e.key === "Home") setPos(0);
    else if (e.key === "End") setPos(100);
    else return;
    e.preventDefault();
  };

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={0}
      aria-label={`Compare ${beforeLabel.toLowerCase()} and ${afterLabel.toLowerCase()} ${alt}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pos)}
      aria-valuetext={`${Math.round(pos)}% ${beforeLabel} visible`}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onKeyDown={onKey}
      className="relative mx-auto w-full max-w-full cursor-ew-resize touch-pan-y select-none overflow-hidden rounded-xl border border-line bg-black"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={after} alt={`${afterLabel} ${alt}`} draggable={false} className="block max-h-[70vh] w-full object-contain" />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={before} alt={`${beforeLabel} ${alt}`} draggable={false} className="block max-h-[70vh] size-full object-contain" />
      </div>
      <span className="pointer-events-none absolute top-3 left-3 rounded-md bg-black/60 px-2 py-1 text-xs font-medium backdrop-blur">{beforeLabel}</span>
      <span className="pointer-events-none absolute top-3 right-3 rounded-md bg-black/60 px-2 py-1 text-xs font-medium backdrop-blur">{afterLabel}</span>
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90 shadow-[0_0_12px_rgb(255_255_255/0.6)]" style={{ left: `${pos}%` }}>
        <span className="absolute top-1/2 left-1/2 grid size-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-black shadow-lg">
          <ChevronsLeftRight className="size-5" aria-hidden />
        </span>
      </div>
    </div>
  );
}
