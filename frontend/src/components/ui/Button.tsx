"use client";

import { forwardRef, useCallback, useId, useState, type ButtonHTMLAttributes, type MouseEvent, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type ButtonVariant = "solid" | "ghost" | "outline";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  /**
   * Why the button is disabled ("Type something to speak"). Shown as a tooltip and announced to screen readers
   * (aria-describedby) while the button is disabled; ignored while it is enabled.
   */
  disabledReason?: string;
}

const variants: Record<ButtonVariant, string> = {
  solid:
    "bg-brand text-[var(--color-on-accent)] border border-transparent hover:opacity-90 active:scale-[0.98]",
  ghost:
    "bg-transparent text-text-secondary border border-transparent hover:bg-bg-elevated hover:text-text-primary",
  outline:
    "bg-bg-surface text-text-primary border border-border hover:border-text-primary",
};

const sizes: Record<ButtonSize, string> = {
  // 40 px minimum on touch screens (coarse pointer); sm stays compact for mouse users.
  sm: "h-8 px-3 text-xs gap-1.5 [@media(pointer:coarse)]:h-10",
  md: "h-10 min-h-10 px-4 text-sm gap-2",
  lg: "h-12 px-6 text-base gap-2.5",
};

interface Ripple {
  id: number;
  x: number;
  y: number;
  size: number;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "solid",
    size = "md",
    loading = false,
    leftIcon,
    rightIcon,
    className,
    children,
    disabled,
    onClick,
    type = "button",
    disabledReason,
    title,
    "aria-describedby": describedBy,
    ...rest
  },
  ref
) {
  const reasonId = useId();
  const explain = !!disabledReason && !!disabled;
  const [ripples, setRipples] = useState<Ripple[]>([]);

  const handleClick = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      if (disabled || loading) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const s = Math.max(rect.width, rect.height) * 2;
      const id = Date.now() + Math.random();
      setRipples((r) => [
        ...r,
        { id, x: e.clientX - rect.left - s / 2, y: e.clientY - rect.top - s / 2, size: s },
      ]);
      window.setTimeout(() => setRipples((r) => r.filter((x) => x.id !== id)), 650);
      onClick?.(e);
    },
    [disabled, loading, onClick]
  );

  return (
    <>
      <button
        ref={ref}
        type={type}
        disabled={disabled || loading}
        aria-busy={loading || undefined}
        title={explain ? disabledReason : title}
        aria-describedby={[describedBy, explain ? reasonId : null].filter(Boolean).join(" ") || undefined}
        onClick={handleClick}
        className={cn(
          "relative inline-flex items-center justify-center overflow-hidden rounded-full font-medium select-none",
          "transition-[filter,background-color,border-color,color,transform] duration-200 ease-[var(--ease-out-expo)] active:scale-[0.97]",
          "disabled:cursor-not-allowed disabled:opacity-50 disabled:active:scale-100",
          variants[variant],
          sizes[size],
          className
        )}
        {...rest}
      >
        {ripples.map((r) => (
          <span
            key={r.id}
            aria-hidden
            className="pointer-events-none absolute rounded-full bg-white animate-ripple"
            style={{ left: r.x, top: r.y, width: r.size, height: r.size }}
          />
        ))}
        {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : leftIcon}
        <span className="relative inline-flex items-center gap-[inherit]">{children}</span>
        {!loading && rightIcon}
      </button>
      {explain && (
        <span id={reasonId} className="sr-only">
          {disabledReason}
        </span>
      )}
    </>
  );
});
