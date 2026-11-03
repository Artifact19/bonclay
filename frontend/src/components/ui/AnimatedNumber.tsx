"use client";

import { useEffect, useState } from "react";
import { useMotionValue, useReducedMotion, useSpring } from "motion/react";
import { cn } from "@/lib/utils";

export interface AnimatedNumberProps {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  locale?: string;
}

/**
 * Spring count-up. Re-animates whenever `value` changes.
 * State-driven (no direct DOM writes) and not gated on visibility, so the final value
 * always lands even in background tabs or when IntersectionObserver never fires.
 */
export function AnimatedNumber({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  className,
  locale,
}: AnimatedNumberProps) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(0);
  const spring = useSpring(mv, { stiffness: 90, damping: 20, mass: 0.8 });
  const [display, setDisplay] = useState(0);

  useEffect(() => spring.on("change", setDisplay), [spring]);

  useEffect(() => {
    // Hidden documents pause rAF, so a spring would stall mid-count: land on the value instead.
    if (reduce || document.visibilityState !== "visible") {
      spring.jump(value);
      mv.jump(value);
    } else {
      mv.set(value);
    }
  }, [value, reduce, mv, spring]);

  const hidden = typeof document !== "undefined" && document.visibilityState !== "visible";
  const shown = reduce || hidden ? value : display;
  const text =
    prefix +
    shown.toLocaleString(locale, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) +
    suffix;

  return (
    <span data-numeric suppressHydrationWarning className={cn("tabular-nums", className)} aria-label={`${prefix}${value}${suffix}`}>
      {text}
    </span>
  );
}
