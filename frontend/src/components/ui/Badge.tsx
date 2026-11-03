import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type BadgeTone = "neutral" | "primary" | "secondary" | "success" | "warning" | "error";

const tones: Record<BadgeTone, string> = {
  neutral: "bg-bg-elevated text-text-secondary border-border",
  primary: "bg-accent-primary/15 text-[#C4B5FD] border-accent-primary/30 [[data-theme=light]_&]:text-accent-primary",
  secondary: "bg-accent-secondary/15 text-accent-secondary border-accent-secondary/30",
  success: "bg-accent-success/15 text-accent-success border-accent-success/30",
  warning: "bg-accent-warning/15 text-accent-warning border-accent-warning/30",
  error: "bg-accent-error/15 text-accent-error border-accent-error/30",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  dot?: boolean;
}

export function Badge({ tone = "neutral", dot = false, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold tracking-wide",
        tones[tone],
        className
      )}
      {...rest}
    >
      {dot && <span aria-hidden className="size-1.5 rounded-full bg-current" />}
      {children}
    </span>
  );
}
