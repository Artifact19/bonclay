import { ErrorBoundary } from "@/components/ui";
import { SkipLink } from "@/components/SkipLink";

/**
 * The skip link and <main id="main-content"> that the public pages (the marketing site and /verify) render. The app
 * shell has its own copy in LayoutWrapper, which also draws the sidebar. Nothing here loads auth, sidebar or toaster code.
 */
export function PublicMain({ children }: { children: React.ReactNode }) {
  return (
    <>
      <SkipLink />
      <main id="main-content" tabIndex={-1} className="relative z-10 min-h-screen outline-none">
        <ErrorBoundary>{children}</ErrorBoundary>
      </main>
    </>
  );
}
