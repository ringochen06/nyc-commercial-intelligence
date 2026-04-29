import type { Metadata } from "next";
import "./globals.css";
import { NavBar } from "@/components/NavBar";

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
      <body className="font-sans">
        <NavBar />
        <main className="max-w-7xl mx-auto px-5 py-8">{children}</main>
        <footer className="max-w-7xl mx-auto px-5 py-10 text-xs text-muted">
          NYC Commercial Intelligence · FastAPI on Railway · Next.js on Vercel
        </footer>
      </body>
    </html>
  );
}
