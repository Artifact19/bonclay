/** The first focusable element on a page. It jumps past the navigation to <main id="main-content">. */
export function SkipLink() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[100] focus:rounded-lg focus:bg-bg-surface focus:px-4 focus:py-2.5 focus:text-sm focus:font-medium focus:text-text-primary focus:shadow-lg focus:outline-2 focus:outline-offset-2 focus:outline-brand"
    >
      Skip to content
    </a>
  );
}
