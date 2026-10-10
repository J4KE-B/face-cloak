"use client";

import { ArrowRight, Play } from "lucide-react";
import { motion } from "framer-motion";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { TextReveal } from "@/components/ui/text-reveal";
import { Typewriter } from "@/components/ui/typewriter";
import { ShimmerLink } from "@/components/ui/shimmer-button";
import { HeroVisual } from "@/components/hero-visual";
import { REPO_URL } from "@/lib/config";

export function Hero() {
  return (
    <section className="relative isolate overflow-hidden">
      <AuroraBackground />
      <div className="mx-auto grid max-w-6xl items-center gap-10 px-4 pt-32 pb-20 sm:px-6 lg:grid-cols-[1.25fr_1fr] lg:pt-40">
        <div>
          <motion.p
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-white/5 px-3 py-1 text-xs text-muted"
          >
            <span className="size-1.5 rounded-full bg-ok" /> Multimedia Forensics project, live demo
          </motion.p>
          <h1 className="text-balance text-4xl font-extrabold tracking-tight sm:text-6xl">
            <TextReveal text="Your photo." />
            <br />
            <TextReveal text="Invisible armour against deepfakes." delay={0.25} wordClassName="text-gradient pb-[0.12em]" />
          </h1>
          <p className="mt-6 max-w-xl text-pretty text-lg text-muted">
            Face-swap tools read an identity fingerprint from your face. We nudge every pixel by at most{" "}
            <span className="font-semibold text-fg">8 out of 255</span>, a change that is barely visible, designed to break
            the identity fingerprint face-swap tools rely on.
          </p>
          <p className="mt-4 h-7 font-mono text-sm text-brand2">
            <Typewriter
              phrases={[
                "> original photo  ->  swap works",
                "> cloaked photo   ->  identity no longer matches (tested on inswapper)",
                "> change per pixel <= 8/255 (barely visible)",
                "> judged by an independent model",
              ]}
            />
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <ShimmerLink href="/demo">
              <Play className="size-4" aria-hidden /> Try the live demo
            </ShimmerLink>
            <ShimmerLink href="/results" variant="ghost">
              See the results <ArrowRight className="size-4" aria-hidden />
            </ShimmerLink>
            <ShimmerLink href={REPO_URL} variant="ghost" external>
              GitHub
            </ShimmerLink>
          </div>
        </div>
        <HeroVisual />
      </div>
    </section>
  );
}
