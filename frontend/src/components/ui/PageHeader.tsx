import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Content widths shared by every dashboard page: narrow = forms, default = most pages, wide = data tables, full = studios. */
export type PageWidth = "narrow" | "default" | "wide" | "full";

const WIDTHS: Record<PageWidth, string> = {
  narrow: "max-w-3xl",
  default: "max-w-5xl",
  wide: "max-w-6xl",
  full: "max-w-7xl",
};

/** One width and padding scheme for every page, so content does not jump left and right between pages. */
export function PageShell({
  width = "default",
  className,
  children,
}: {
  width?: PageWidth;
  className?: string;
  children: ReactNode;
}) {
  return <div className={cn("mx-auto w-full px-4 py-6 md:px-8 md:py-10", WIDTHS[width], className)}>{children}</div>;
}

export interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  className?: string;
  /**
   * Opt in to the shared page shell: pass a width and the header (plus its `children`) is wrapped in the standard
   * max-width and padding. Without it the header renders alone, as before.
   */
  width?: PageWidth;
  children?: ReactNode;
}

export function PageHeader({ title, description, actions, className, width, children }: PageHeaderProps) {
  const header = (
    <header className={cn("mb-8 flex flex-wrap items-end justify-between gap-4", className)}>
      <div className="min-w-0">
        <h1 className="font-display text-[clamp(1.75rem,2.6vw,2.5rem)] font-medium leading-[1.08] tracking-[-0.03em] text-text-primary">
          {title}
        </h1>
        {description && <p className="mt-2 max-w-prose text-[15px] leading-relaxed text-text-secondary">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  );
  if (!width) return header;
  return (
    <PageShell width={width}>
      {header}
      {children}
    </PageShell>
  );
}
