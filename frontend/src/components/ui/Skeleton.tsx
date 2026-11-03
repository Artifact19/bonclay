import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Skeleton({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden
      className={cn(
        "rounded-md bg-[linear-gradient(90deg,var(--color-bg-elevated)_25%,var(--color-border)_50%,var(--color-bg-elevated)_75%)] bg-[length:200%_100%] animate-shimmer",
        className
      )}
      {...rest}
    />
  );
}
