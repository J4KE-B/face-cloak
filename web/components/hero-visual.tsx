"use client";

import { motion, useReducedMotion } from "framer-motion";

/** Decorative SVG: a face inside a shield, scanned by a beam, surrounded by rotating rings. */
export function HeroVisual() {
  const reduce = useReducedMotion();
  return (
    <div aria-hidden className="relative mx-auto aspect-square w-full max-w-sm">
      <svg viewBox="0 0 400 400" className="size-full">
        <defs>
          <linearGradient id="hv-g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#7c5cff" />
            <stop offset="1" stopColor="#22d3ee" />
          </linearGradient>
          <clipPath id="hv-clip">
            <path d="M200 50 90 90v90c0 80 50 130 110 160 60-30 110-80 110-160V90z" />
          </clipPath>
        </defs>
        <g className={reduce ? "" : "origin-center animate-spin-slow"} style={{ transformOrigin: "200px 200px" }}>
          <circle cx="200" cy="200" r="185" fill="none" stroke="url(#hv-g)" strokeOpacity=".35" strokeWidth="1.500" strokeDasharray="4 10" />
        </g>
        <g className={reduce ? "" : "origin-center animate-spin-slow"} style={{ transformOrigin: "200px 200px", animationDirection: "reverse", animationDuration: "14s" }}>
          <circle cx="200" cy="200" r="160" fill="none" stroke="#22d3ee" strokeOpacity=".25" strokeWidth="1" strokeDasharray="30 14" />
        </g>
        <motion.path
          d="M200 50 90 90v90c0 80 50 130 110 160 60-30 110-80 110-160V90z"
          fill="rgb(124 92 255 / 0.08)"
          stroke="url(#hv-g)"
          strokeWidth="4"
          strokeLinejoin="round"
          initial={reduce ? false : { pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.6, ease: "easeInOut" }}
        />
        <g clipPath="url(#hv-clip)">
          <ellipse cx="200" cy="180" rx="52" ry="64" fill="none" stroke="#e8ecf5" strokeOpacity=".7" strokeWidth="2.500" />
          <path d="M150 300c8-45 28-62 50-62s42 17 50 62" fill="none" stroke="#e8ecf5" strokeOpacity=".5" strokeWidth="2.500" />
          {[[180, 165], [220, 165], [200, 190], [183, 215], [217, 215]].map(([x, y], i) => (
            <motion.circle
              key={i}
              cx={x}
              cy={y}
              r="3.500"
              fill="#22d3ee"
              animate={reduce ? undefined : { opacity: [0.2, 1, 0.2], r: [2.500, 4.500, 2.500] }}
              transition={{ duration: 2.400, repeat: Infinity, delay: i * 0.25 }}
            />
          ))}
          {!reduce && (
            <motion.rect
              x="80"
              width="240"
              height="3"
              fill="#22d3ee"
              style={{ filter: "drop-shadow(0 0 8px #22d3ee)" }}
              animate={{ y: [80, 330, 80] }}
              transition={{ duration: 4.500, repeat: Infinity, ease: "easeInOut" }}
            />
          )}
        </g>
      </svg>
    </div>
  );
}
