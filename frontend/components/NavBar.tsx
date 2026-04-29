"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/k-selection", label: "K-Selection" },
  { href: "/ranking", label: "Ranking" },
];

export function NavBar() {
  const pathname = usePathname();
  return (
    <header className="glass-header sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-5 py-3 flex items-center gap-8">
        <Link href="/" className="font-semibold text-ink tracking-tightish">
          <span className="inline-flex items-center gap-2">
            <span
              aria-hidden
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{
                background:
                  "linear-gradient(135deg, #fb923c 0%, #f59e0b 55%, #b45309 100%)",
                boxShadow: "0 0 12px rgba(251, 146, 60, 0.55)",
              }}
            />
            NYC Commercial Intelligence
          </span>
        </Link>
        <nav className="flex gap-1.5 text-sm">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-1.5 rounded-full nav-anim ${
                  active ? "nav-active" : "text-muted hover:text-ink hover:bg-white/45"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
