import type { InputHTMLAttributes, ReactNode } from "react";
import { useSearchParams } from "next/navigation";
import { Logo } from "@/components/Logo";

/** Only same-origin relative paths are honoured, to avoid open redirects. */
export function useNextPath(): string {
  const params = useSearchParams();
  const next = params.get("next");
  if (next && next.startsWith("/") && !next.startsWith("//") && !next.includes("\\")) return next;
  return "/dashboard";
}

export function AuthShell({ subtitle, children }: { subtitle: string; children: ReactNode }) {
  return (
    <div className="relative flex min-h-screen flex-col items-center justify-center bg-bg-base px-4 py-10 sm:px-6">
      <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-80" style={{ background: "var(--gradient-glow)" }} />
      <div className="relative w-full max-w-sm">
        <div className="mb-8 flex flex-col items-center text-center">
          <Logo size={36} />
          <p className="mt-3 text-sm text-text-secondary">{subtitle}</p>
        </div>
        <div className="space-y-5 rounded-2xl border border-border bg-bg-surface p-6 sm:p-8">{children}</div>
      </div>
    </div>
  );
}

export function Field({ label, id, ...rest }: { label: string; id: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-xs font-medium text-text-secondary">
        {label}
      </label>
      <input
        id={id}
        {...rest}
        className="h-10 w-full rounded-lg border border-border bg-bg-base px-3 text-sm text-text-primary placeholder:text-text-muted transition-colors duration-200 focus:border-accent-primary focus:outline-none"
      />
    </div>
  );
}

export function Divider() {
  return (
    <div className="flex items-center gap-3" role="separator">
      <div className="h-px flex-1 bg-border" />
      <span className="text-xs text-text-muted">or</span>
      <div className="h-px flex-1 bg-border" />
    </div>
  );
}

export function GoogleButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex h-10 w-full items-center justify-center gap-3 rounded-lg border border-border text-sm font-medium text-text-primary transition-colors duration-200 hover:border-accent-primary hover:bg-bg-elevated"
    >
      <svg className="size-4" viewBox="0 0 24 24" aria-hidden>
        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
      </svg>
      Continue with Google
    </button>
  );
}

export function FormError({ message }: { message: string }) {
  if (!message) return null;
  return (
    <p role="alert" className="text-xs text-accent-error">
      {message}
    </p>
  );
}
