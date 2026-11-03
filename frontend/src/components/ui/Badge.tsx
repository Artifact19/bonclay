import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export type BadgeTone = "neutral" | "primary" | "secondary" | "success" | "warning" | "error";

const tones: Record<BadgeTone, string> = {
  neutral: "bg-[#E2DDD3] text-[#141414] border-[#141414]",
  primary: "bg-[#F3F59E] text-[#141414] border-[#141414]",
  secondary: "bg-[#E2DDD3] text-[#141414] border-[#141414]",
  success: "bg-[#D1E7DD] text-[#0F5132] border-[#141414]",
  warning: "bg-[#F3F59E] text-[#141414] border-[#141414]",
  error: "bg-[#F8D7DA] text-[#842029] border-[#141414]",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  dot?: boolean;
}

export function Badge({ tone = "neutral", dot = false, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border-[1.5px] px-2.5 py-0.5 text-[11px] font-semibold tracking-wide",
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
