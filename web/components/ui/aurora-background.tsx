import { cn } from "@/lib/cn";

/**
 * Animated gradient / aurora backdrop. Pattern inspired by the animated-background components on vengenceui.com;
 * re-implemented here with Tailwind keyframes (pure CSS, so it costs no JS and respects prefers-reduced-motion).
 */
export function AuroraBackground({ className }: { className?: string }) {
  return (
    <div aria-hidden className={cn("pointer-events-none absolute inset-0 -z-10 overflow-hidden", className)}>
      <div className="absolute -top-40 left-1/2 h-[34rem] w-[34rem] -translate-x-[90%] animate-aurora rounded-full bg-brand/30 blur-[120px]" />
      <div className="absolute -top-24 left-1/2 h-[30rem] w-[30rem] translate-x-[10%] animate-aurora rounded-full bg-brand2/20 blur-[120px] [animation-delay:-6s] [animation-direction:alternate-reverse]" />
      <div className="absolute top-60 left-1/2 h-[26rem] w-[26rem] -translate-x-1/2 animate-aurora rounded-full bg-fuchsia-500/10 blur-[120px] [animation-delay:-11s]" />
      <div className="grid-bg absolute inset-0" />
      <div className="absolute inset-x-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-bg" />
    </div>
  );
}
