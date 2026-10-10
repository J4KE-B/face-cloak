import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import "./globals.css";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { ToastProvider } from "@/components/toast";
import { MotionProvider } from "@/components/motion-provider";
import { SITE_URL } from "@/lib/config";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: "face-cloak: invisible armour against face-swap deepfakes", template: "%s | face-cloak" },
  description:
    "Multimedia Forensics project: an adversarial perturbation designed to break the identity fingerprint that face-swap tools rely on, with barely visible changes to your photo. Try the live demo and see the measured results for white-box (inswapper) and black-box (SimSwap) attacks.",
  openGraph: {
    title: "face-cloak: invisible armour against face-swap deepfakes",
    description: "Cloak a photo with a barely visible perturbation and test it against a face-swap attack.",
    url: SITE_URL,
    siteName: "face-cloak",
    type: "website",
  },
  twitter: { card: "summary_large_image" },
  robots: { index: true, follow: true },
};

export const viewport: Viewport = { themeColor: "#05060a", colorScheme: "dark" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a
          href="#main"
          className="fixed top-2 left-2 z-[200] -translate-y-20 rounded-lg bg-brand px-4 py-2 text-sm font-semibold text-white focus:translate-y-0"
        >
          Skip to content
        </a>
        <MotionProvider>
          <ToastProvider>
            <Navbar />
            {children}
            <Footer />
          </ToastProvider>
        </MotionProvider>
      </body>
    </html>
  );
}
