import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { nextPage } from "./routes";

/**
 * The top of every marketing page except Home: the page's one <h1> and a one-sentence intro, in the same type scale as
 * the sections below it. There is no label above the heading: the nav already says where you are. A server component,
 * so it adds no JavaScript.
 */
export function PageIntro({ title, children, id = "page-title" }: { title: string; children?: React.ReactNode; id?: string }) {
  return (
    <section className="s-container pt-28 md:pt-36" aria-labelledby={id}>
      <h1 id={id} className="s-title max-w-[18ch] text-[var(--s-ink)]">
        {title}
      </h1>
      {children && (
        <p className="lede mt-6">
          {children}
        </p>
      )}
    </section>
  );
}

/**
 * "Next" link at the foot of a page: the page that follows in reading order (see routes.ts). `to` overrides the
 * destination, for the last page, which points back to Home.
 */
export function NextPage({ from, to }: { from: string; to?: { href: string; title: string; blurb: string } }) {
  const target = to ?? nextPage(from);
  if (!target) return null;
  return (
    <nav aria-label="Next page" className="s-container mt-24 md:mt-32">
      <Link
        href={target.href}
        data-reveal
        className="group flex min-h-[5.5rem] flex-col gap-3 rounded-[var(--s-r-card)] border border-[var(--s-line)] bg-[var(--s-paper)] p-6 transition-colors hover:border-[var(--s-ink)] sm:flex-row sm:items-center sm:justify-between sm:gap-8 sm:p-8"
      >
        <span className="min-w-0">
          <span className="s-label block text-[var(--s-mute)]">Next</span>
          <span className="s-h3 mt-2 block text-[var(--s-ink)]">{target.title}</span>
          <span className="mt-2 block max-w-[60ch] s-body leading-snug text-[var(--s-mute)]">{target.blurb}</span>
        </span>
        <ArrowRight className="h-6 w-6 shrink-0 text-[var(--s-ink)] transition-transform duration-200 group-hover:translate-x-1 motion-reduce:transition-none" aria-hidden />
      </Link>
    </nav>
  );
}
