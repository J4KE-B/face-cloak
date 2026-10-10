import type { Metadata } from "next";
import type { ReactNode } from "react";
import { AlertTriangle, BookOpen, Heart, Scale, ShieldCheck, Users } from "lucide-react";
import { GlowCard } from "@/components/ui/glow-card";
import { PageHeader, Reveal } from "@/components/ui/section";
import { REPO_URL, REPORT_URL, TEAM } from "@/lib/config";

export const metadata: Metadata = {
  title: "About, limits and ethics",
  description: "Method, threat model, limitations, ethics and credits of the face-cloak multimedia-forensics project.",
};

function Block({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <Reveal>
      <section className="mb-10" aria-labelledby={title.replace(/\W+/g, "-")}>
        <h2 id={title.replace(/\W+/g, "-")} className="mb-4 flex items-center gap-3 text-2xl font-bold">
          <span className="grid size-9 place-items-center rounded-xl bg-brand/15 text-brand [&_svg]:size-5">{icon}</span>
          {title}
        </h2>
        <div className="space-y-3 text-muted [&_strong]:text-fg">{children}</div>
      </section>
    </Reveal>
  );
}

const Li = ({ children }: { children: ReactNode }) => <li className="card p-4 text-sm">{children}</li>;

export default function AboutPage() {
  return (
    <>
      <PageHeader eyebrow="About" title="Method, limits and ethics">
        What the cloak does, what it cannot do, and how the demo treats your photo.
      </PageHeader>
      <div className="mx-auto max-w-3xl px-4 pb-10 sm:px-6">
        <Block icon={<ShieldCheck />} title="Method">
          <p>
            Face-swap tools (roop, FaceFusion and the models behind them, such as <strong>inswapper</strong> and <strong>SimSwap</strong>) extract an
            identity embedding from the source photo with an ArcFace-style encoder and inject it into a target image.
            face-cloak attacks that embedding.
          </p>
          <p>
            We run <strong>projected gradient descent (PGD)</strong> on the photo: each step moves the pixels in the direction that pushes the
            encoder&apos;s embedding away from the true identity, then projects back so no pixel differs from the original by more than ε
            (default 8/255, the L∞ budget). The result is saved as a lossless PNG. An optional <strong>robust</strong> mode averages over random
            JPEG, resize and blur transforms (Expectation over Transformation) so the cloak survives re-sharing.
          </p>
          <p>
            Success is measured by a <strong>separate</strong> recognition model, glintr100, at a threshold τ chosen for a 1% false-accept rate.
            If the swapped face scores below τ against a real photo of the person, the attack failed and the person is{" "}
            <strong>protected</strong>. Details and numbers are in the{" "}
            <a className="text-brand2 underline" href={REPORT_URL} target="_blank" rel="noopener noreferrer">report</a>.
          </p>
        </Block>

        <Block icon={<Scale />} title="Threat model">
          <ul className="grid gap-3">
            <Li><strong>Attacker:</strong> has your published photo and a public face-swap tool. Does not know your cloak, runs the tool as shipped.</Li>
            <Li><strong>Defender (you):</strong> cloaks the photo once, before posting. Needs access to an identity encoder to compute the cloak.</Li>
            <Li><strong>Goal:</strong> the swapped face should no longer be recognised as you by an independent recogniser.</Li>
          </ul>
        </Block>

        <Block icon={<AlertTriangle />} title="Limitations">
          <ul className="grid gap-3">
            <Li><strong>White-box encoder vs black-box transfer.</strong> The cloak is built against the same encoder family that inswapper uses, so inswapper is a <em>white-box</em> result. SimSwap uses a different encoder: protection there is <em>black-box transfer</em> and is expected to be weaker. The Results page reports both separately and never merges them.</Li>
            <Li><strong>Purification.</strong> Diffusion-based purification (for example DiffPure) or heavy denoising can remove small perturbations. We do not claim robustness against it.</Li>
            <Li><strong>Adaptive attackers.</strong> Someone who knows the method can train against it. This is an arms race, not a guarantee; protection today may not hold against tomorrow&apos;s tools.</Li>
            <Li><strong>Re-encoding.</strong> Screenshots, strong compression and resizing weaken the cloak; robust mode helps but costs time.</Li>
            <Li><strong>Speed.</strong> On a CPU the optimisation takes minutes; the Kaggle T4 GPU takes seconds to a minute. That is why the demo normally runs on a GPU backend.</Li>
            <Li><strong>One face, one judge.</strong> The live demo runs one photo against one target with one judge. It illustrates the effect; the Results page holds the statistics.</Li>
          </ul>
        </Block>

        <Block icon={<Heart />} title="Ethics and privacy">
          <ul className="grid gap-3">
            <Li><strong>Consent.</strong> Upload only photos of yourself, or of people who agreed. The demo asks you to tick a consent box before every run, and unticks it again as soon as a run starts, so the next run needs a fresh confirmation.</Li>
            <Li><strong>In memory only.</strong> The backend never writes your photos to disk. Jobs and their images are deleted within 10 minutes, and this page asks the backend to delete a job as soon as you have your result.</Li>
            <Li><strong>No tracking.</strong> This website has no analytics and no cookies. The backend address and passcode you enter are stored in your browser&apos;s local storage only; use &quot;Forget&quot; to remove them.</Li>
            <Li><strong>Watermarked fakes.</strong> Every face-swap image the backend returns carries a visible SYNTHETIC band and is capped at 512 px.</Li>
            <Li><strong>Fixed public-domain targets.</strong> You cannot choose other people as targets. The portraits are public-domain NASA photographs, so the demo cannot be used to put your face on a private individual.</Li>
            <Li><strong>Dual use.</strong> A cloak helps defenders; the same research helps attackers build better swappers. We publish the method and its limits openly.</Li>
          </ul>
        </Block>

        <Block icon={<Users />} title="Team">
          <div className="grid gap-4 sm:grid-cols-3">
            {TEAM.map((m) => (
              <GlowCard key={m.name} className="p-5">
                <div className="mb-3 grid size-12 place-items-center rounded-full bg-gradient-to-br from-brand to-brand2 text-lg font-bold text-white" aria-hidden>
                  {m.name.split(" ").map((p) => p[0]).slice(0, 2).join("")}
                </div>
                <p className="font-semibold text-fg">{m.name}</p>
                <p className="text-sm">{m.role}</p>
              </GlowCard>
            ))}
          </div>
          <p className="text-sm">Multimedia Forensics course project, IIIT Kottayam.</p>
        </Block>

        <Block icon={<BookOpen />} title="Credits and licences">
          <ul className="grid gap-3">
            <Li><strong>InsightFace</strong> models (detector, ArcFace recognisers, inswapper) are released for <strong>non-commercial research use only</strong>. This project is a non-commercial university demo.</Li>
            <Li><strong>SimSwap</strong> (Chen et al., ACM MM 2020) is used as the black-box swapper under its own research licence.</Li>
            <Li><strong>LFW (Labeled Faces in the Wild)</strong> supplies the evaluation identities; no LFW faces are shown on this site.</Li>
            <Li><strong>NASA portraits</strong> used as swap targets are U.S. government works in the public domain; sources are linked in the demo.</Li>
            <Li><strong>Web UI</strong> built with Next.js, Tailwind CSS, Framer Motion and Lucide icons; component patterns inspired by vengenceui.com and re-implemented here.</Li>
          </ul>
          <p>
            Source code, notebooks and the report:{" "}
            <a className="text-brand2 underline" href={REPO_URL} target="_blank" rel="noopener noreferrer">{REPO_URL}</a>
          </p>
        </Block>
      </div>
    </>
  );
}
