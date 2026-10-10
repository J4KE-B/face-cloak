import Link from "next/link";
import {
  ScanFace, Fingerprint, Wand2, Save, Shuffle, Scale, Waves, Eye, ShieldAlert, Lock, ArrowRight, FileText,
} from "lucide-react";
import { Hero } from "@/components/hero";
import { HeadlineStats } from "@/components/headline-stats";
import { Pipeline, type PipelineStep } from "@/components/ui/pipeline";
import { GlowCard } from "@/components/ui/glow-card";
import { Reveal, SectionHeading } from "@/components/ui/section";
import { ShimmerLink } from "@/components/ui/shimmer-button";
import { REPO_URL, REPORT_URL } from "@/lib/config";

const STEPS: PipelineStep[] = [
  { icon: <ScanFace className="size-5" />, side: "defender", title: "Find the face", body: "A face detector locates your face and aligns it, exactly as the face-swap tool will." },
  { icon: <Fingerprint className="size-5" />, side: "defender", title: "Read the fingerprint", body: "An ArcFace-style encoder turns the face into a 512-number identity vector. This is what swap tools copy." },
  { icon: <Wand2 className="size-5" />, side: "defender", title: "Optimise the cloak (PGD)", body: "Projected gradient descent searches for a tiny perturbation that pushes the fingerprint away from yours, never changing a pixel by more than ε." },
  { icon: <Save className="size-5" />, side: "defender", title: "Save as lossless PNG", body: "The cloaked image is stored as PNG so the perturbation survives. You post this version instead of the original." },
  { icon: <Shuffle className="size-5" />, side: "attacker", title: "Attacker swaps your face", body: "A face-swap tool (inswapper, SimSwap) tries to paste your identity onto a target portrait." },
  { icon: <Scale className="size-5" />, side: "judge", title: "Independent judge", body: "A different model, glintr100, compares the swapped face with your real face. Below the threshold τ means: not you." },
  { icon: <Waves className="size-5" />, side: "judge", title: "Stress test", body: "We repeat it after JPEG compression, resizing and blur to see whether the cloak survives real-world re-sharing." },
];

export default function Home() {
  return (
    <>
      <Hero />

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6" aria-labelledby="threat">
        <SectionHeading eyebrow="The threat" title="One selfie is enough to put you in someone else's video">
          Tools such as roop and FaceFusion do not need training on your face. They compute one identity embedding and
          graft it onto any target video.
        </SectionHeading>
        <div className="grid gap-4 md:grid-cols-3" id="threat">
          {[
            { icon: <Fingerprint />, t: "Identity fingerprint", d: "The swapper extracts a 512-dimensional ArcFace embedding from your photo. That vector, not your pixels, drives the fake." },
            { icon: <Eye />, t: "Invisible armour", d: "Projected gradient descent (PGD) adds a perturbation bounded by 8/255 per pixel, barely visible to the eye, aimed at moving that embedding." },
            { icon: <ShieldAlert />, t: "Measured, not assumed", d: "A separate recognition model (glintr100) judges every swap, so the result is not just the cloak grading its own homework." },
          ].map((c, i) => (
            <Reveal key={c.t} delay={i * 0.08}>
              <GlowCard className="h-full p-6">
                <div className="mb-4 grid size-10 place-items-center rounded-xl bg-brand/15 text-brand [&_svg]:size-5">{c.icon}</div>
                <h3 className="font-semibold">{c.t}</h3>
                <p className="mt-2 text-sm text-muted">{c.d}</p>
              </GlowCard>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6" aria-labelledby="how">
        <SectionHeading eyebrow="How it works" title="Seven steps from photo to verdict">
          The first four happen to your photo. The last three are our own attack and independent check, so you can see
          whether the armour holds.
        </SectionHeading>
        <div id="how">
          <Pipeline steps={STEPS} />
        </div>
      </section>

      <HeadlineStats />

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <Reveal>
          <GlowCard className="p-8 text-center sm:p-12">
            <Lock className="mx-auto size-8 text-brand2" aria-hidden />
            <h2 className="mt-4 text-balance text-3xl font-bold sm:text-4xl">Test it on your own photo</h2>
            <p className="mx-auto mt-3 max-w-xl text-muted">
              Upload a photo of yourself, pick a public-domain target, and compare the same face-swap on the
              original and on the cloaked copy. Photos are processed in memory and deleted within 10 minutes.
            </p>
            <div className="mt-7 flex flex-wrap justify-center gap-3">
              <ShimmerLink href="/demo">
                Open the live demo <ArrowRight className="size-4" aria-hidden />
              </ShimmerLink>
              <ShimmerLink href={REPORT_URL} variant="ghost" external>
                <FileText className="size-4" aria-hidden /> Read the report
              </ShimmerLink>
              <ShimmerLink href={REPO_URL} variant="ghost" external>
                Source on GitHub
              </ShimmerLink>
            </div>
            <p className="mt-6 text-xs text-muted">
              Limits and ethics are on the <Link href="/about" className="underline hover:text-fg">About page</Link>.
            </p>
          </GlowCard>
        </Reveal>
      </section>
    </>
  );
}
