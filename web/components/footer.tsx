import Link from "next/link";
import { REPO_URL } from "@/lib/config";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-10 text-sm text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div>
          <p className="font-semibold text-fg">face-cloak</p>
          <p className="mt-1 max-w-md">
            University project for Multimedia Forensics: adversarial image cloaking for protection against face-swap
            attacks. Research demo, not a security product.
          </p>
        </div>
        <nav aria-label="Footer" className="flex flex-wrap gap-x-5 gap-y-2">
          <Link href="/demo" className="hover:text-fg">Live demo</Link>
          <Link href="/results" className="hover:text-fg">Results</Link>
          <Link href="/about" className="hover:text-fg">About &amp; ethics</Link>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="hover:text-fg">GitHub</a>
        </nav>
      </div>
    </footer>
  );
}
