import type { Metadata } from "next";
import { DemoClient } from "@/components/demo/demo-client";

export const metadata: Metadata = {
  title: "Live demo",
  robots: { index: false, follow: true },
  description: "Upload your own photo, cloak it, and compare a face-swap deepfake made from the original with one made from the cloaked copy.",
};

export default function DemoPage() {
  return <DemoClient />;
}
