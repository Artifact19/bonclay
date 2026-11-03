import { forwardRef, type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** glass: translucent blurred surface. solid: opaque surface. */
  variant?: "glass" | "solid";
  interactive?: boolean;
  padded?: boolean;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(function Card(
  { variant = "solid", interactive = false, padded = true, className, ...rest },
  ref
) {
  return (
    <div
      ref={ref}
      className={cn(
        "rounded-2xl border-[1.5px] border-[#141414] bg-[#FFFFFF] shadow-[0_2px_0_#141414]",
        variant === "glass" ? "bg-[#FFFFFF]" : "bg-[#FFFFFF]",
        padded && "p-5",
        interactive &&
          "transition-[transform,box-shadow] duration-200 ease-[var(--ease-out-expo)] hover:-translate-y-0.5 hover:shadow-[0_4px_0_#141414]",
        className
      )}
      {...rest}
    />
  );
});
