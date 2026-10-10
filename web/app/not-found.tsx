import type { Metadata } from "next";
import { ShieldQuestion } from "lucide-react";
import { ShimmerLink } from "@/components/ui/shimmer-button";
import { AuroraBackground } from "@/components/ui/aurora-background";

export const metadata: Metadata = { title: "Page not found" };

export default function NotFound() {
  return (
    <div className="relative mx-auto flex min-h-[70dvh] max-w-xl flex-col items-center justify-center px-4 pt-24 text-center">
      <AuroraBackground />
      <ShieldQuestion className="size-14 animate-float text-brand2" aria-hidden />
      <h1 className="mt-6 text-4xl font-bold">404: face not found</h1>
      <p className="mt-3 text-muted">That page does not exist. Perhaps it was cloaked a little too well.</p>
      <div className="mt-8 flex gap-3">
        <ShimmerLink href="/">Back home</ShimmerLink>
        <ShimmerLink href="/demo" variant="ghost">Live demo</ShimmerLink>
      </div>
    </div>
  );
}
