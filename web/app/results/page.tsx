import type { Metadata } from "next";
import { PageHeader } from "@/components/ui/section";
import { ResultsClient } from "@/components/results-view";

export const metadata: Metadata = {
  title: "Experiment results",
  description: "Protection success rates, image-quality metrics and figures from the face-cloak experiments, with white-box and black-box swappers reported separately.",
};

export default function ResultsPage() {
  return (
    <>
      <PageHeader eyebrow="Experiments" title="Does the cloak actually work?">
        Measured protection against two face-swap tools, with confidence intervals. One swapper shares the encoder the cloak
        was built against (white-box); the other does not (black-box transfer).
      </PageHeader>
      <div className="mx-auto max-w-6xl px-4 pb-10 sm:px-6">
        <ResultsClient />
      </div>
    </>
  );
}
