"use client";

import { motion, useReducedMotion } from "framer-motion";

/** Word-by-word blur/slide-in heading (text-reveal pattern). Renders plain text under reduced motion. */
export function TextReveal({
  text,
  className,
  delay = 0,
  wordClassName,
  as: Tag = "span",
}: {
  text: string;
  className?: string;
  delay?: number;
  wordClassName?: string;
  as?: "span" | "h1" | "h2" | "h3" | "p";
}) {
  const reduce = useReducedMotion();
  const words = text.split(" ");
  if (reduce) return <Tag className={`${className ?? ""} ${wordClassName ?? ""}`}>{text}</Tag>;
  return (
    <Tag className={className} aria-label={text}>
      {words.map((w, i) => (
        <motion.span
          key={`${w}-${i}`}
          aria-hidden
          className={`inline-block whitespace-pre ${wordClassName ?? ""}`}
          initial={{ opacity: 0, y: 18, filter: "blur(10px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.6, delay: delay + i * 0.07, ease: [0.2, 0.7, 0.2, 1] }}
        >
          {w}
          {i < words.length - 1 ? " " : ""}
        </motion.span>
      ))}
    </Tag>
  );
}
