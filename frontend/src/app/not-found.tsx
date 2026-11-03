import Link from "next/link";
import { Logo } from "@/components/Logo";
import { PublicMain } from "@/components/PublicMain";
import { manrope } from "@/lib/brandFont";

// Rendered by the root layout, outside the app shell. The root layout is shared by every route, and the CSS of this file
// and error.tsx reaches every route's HTML, so this page uses the brand face and the system UI font, not Geist (which
// would be preloaded on the public site). PublicMain supplies the skip link and <main>, as the root LayoutWrapper did.
export default function NotFound() {
  return (
    <PublicMain>
      <div className={`flex min-h-screen flex-col items-center justify-center bg-bg-base px-4 py-16 text-center ${manrope.variable}`}>
        <Logo size={36} />
        <p className="mt-10 text-6xl font-semibold text-text-primary">404</p>
        <h1 className="mt-3 text-lg font-medium text-text-primary">That page has gone quiet</h1>
        <p className="mt-2 max-w-sm text-sm text-text-secondary">
          The link may be old or mistyped. Head back and pick up where you left off.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/dashboard"
            className="inline-flex h-10 items-center rounded-lg bg-brand px-4 text-sm font-medium text-[var(--color-on-accent)] transition-[filter] duration-200 hover:brightness-110"
          >
            Go to dashboard
          </Link>
          <Link
            href="/"
            className="inline-flex h-10 items-center rounded-lg border border-border px-4 text-sm font-medium text-text-primary transition-colors duration-200 hover:border-accent-primary hover:bg-bg-elevated"
          >
            Home
          </Link>
        </div>
      </div>
    </PublicMain>
  );
}
