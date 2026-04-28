import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "NYC Commercial Intelligence",
  description:
    "Decision-support dashboard for ranking NYC neighborhoods for commercial site selection.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-slate-200 bg-white sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-6">
            <Link href="/" className="font-semibold text-ink">
              NYC Commercial Intelligence
            </Link>
            <nav className="flex gap-4 text-sm text-muted">
              <Link href="/" className="hover:text-ink">
                K-Selection
              </Link>
              <Link href="/ranking" className="hover:text-ink">
                Ranking
              </Link>
            </nav>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 py-6">{children}</main>
        <footer className="max-w-7xl mx-auto px-4 py-6 text-xs text-muted">
          NYC Commercial Intelligence · Streamlit-derived UI ported to Next.js
          on Vercel · API on Railway
        </footer>
      </body>
    </html>
  );
}
