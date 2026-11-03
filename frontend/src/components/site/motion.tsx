"use client";

/**
 * The marketing site's Motion (motion.dev) vocabulary. Optimized for 60-120fps local performance:
 * animations use compositor-only properties (transform & opacity) without expensive per-frame CSS blur filters.
 */

import {
  motion,
  useInView,
  useMotionTemplate,
  useMotionValue,
  useReducedMotion,
  useScroll,
  useSpring,
  useTransform,
  type MotionValue,
  type Variants,
} from "motion/react";
import { useEffect, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from "react";

/** The site's deceleration curve (matches --s-ease in globals.css). */
export const EASE = [0.22, 1, 0.36, 1] as const;
/** A light spring for pointer-following motion. */
export const SOFT_SPRING = { stiffness: 150, damping: 20, mass: 0.4 };
/** The spring for cards and buttons that lift. */
export const LIFT_SPRING = { type: "spring", stiffness: 300, damping: 24 } as const;

/** True when the visitor prefers reduced motion (or before hydration, so the server and first client render agree). */
export function useStill(): boolean {
  const reduced = useReducedMotion();
  return reduced === true;
}

/* ------------------------------------------------------------------ split heading */

const lineVariants: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04, delayChildren: 0.03 } },
};
const wordVariants: Variants = {
  hidden: { y: "105%", opacity: 0.001 },
  show: { y: "0%", opacity: 1, transition: { duration: 0.65, ease: EASE } },
};

type SplitHeadingProps = {
  /** The upright part of the heading. */
  text: string;
  /** The italic accent phrase that follows it, on its own line when `break` is set. */
  accent?: string;
  /** Put the accent on a new line. */
  break?: boolean;
  /** Animate as soon as it mounts (the hero) instead of when it scrolls into view. */
  immediate?: boolean;
  delay?: number;
  as?: "h1" | "h2" | "h3" | "p";
  id?: string;
  className?: string;
};

/** A heading whose words rise one by one out of a clipped line. Server-rendered text is complete in the DOM. */
export function SplitHeading({ text, accent, break: br, immediate, delay = 0, as = "h2", id, className = "" }: SplitHeadingProps) {
  const still = useStill();
  const Tag = motion[as];
  const words = (s: string, accentWord: boolean) =>
    s.split(" ").filter(Boolean).map((w, i, arr) => (
      <span key={`${accentWord ? "a" : "t"}${i}`} className="inline-block overflow-hidden pb-[0.08em] -mb-[0.08em] align-top">
        <motion.span variants={still ? undefined : wordVariants} className={`inline-block ${accentWord ? "accent" : ""}`}>
          {w}
          {i < arr.length - 1 ? "\u00A0" : ""}
        </motion.span>
      </span>
    ));
  return (
    <Tag
      id={id}
      className={className}
      variants={still ? undefined : lineVariants}
      initial={still ? false : "hidden"}
      {...(immediate ? { animate: "show" } : { whileInView: "show", viewport: { once: true, amount: 0.35 } })}
      transition={{ delayChildren: delay }}
    >
      {words(text, false)}
      {accent && (
        <>
          {br ? <br /> : " "}
          {words(accent, true)}
        </>
      )}
    </Tag>
  );
}

/* ------------------------------------------------------------------ reveal */

type RevealProps = {
  children: ReactNode;
  delay?: number;
  /** Lift distance in px. */
  y?: number;
  /** Start slightly smaller as well (for images and panels). */
  scale?: boolean;
  /** Animate on mount instead of on view. */
  immediate?: boolean;
  as?: "div" | "p" | "li" | "section" | "span" | "figure";
  className?: string;
  style?: CSSProperties;
  id?: string;
  "aria-label"?: string;
  role?: string;
};

/** Fades and lifts its content the first time it scrolls into view (GPU compositor only, no blur filter). */
export function Reveal({ children, delay = 0, y = 20, scale, immediate, as = "div", className, style, ...rest }: RevealProps) {
  const still = useStill();
  const Tag = motion[as];
  if (still) {
    const Plain = as;
    return (
      <Plain className={className} style={style} {...rest}>
        {children}
      </Plain>
    );
  }
  return (
    <Tag
      className={className}
      style={style}
      initial={{ opacity: 0, y, scale: scale ? 0.98 : 1 }}
      {...(immediate
        ? { animate: { opacity: 1, y: 0, scale: 1 } }
        : { whileInView: { opacity: 1, y: 0, scale: 1 }, viewport: { once: true, margin: "0px 0px -10% 0px" } })}
      transition={{ duration: 0.55, ease: EASE, delay }}
      {...rest}
    >
      {children}
    </Tag>
  );
}

const staggerParent: Variants = { hidden: {}, show: { transition: { staggerChildren: 0.06 } } };
const staggerChild: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE } },
};

/** A list (or grid) whose children reveal one after another. Wrap each child in <Item>. */
export function Stagger({ children, as = "div", className, ...rest }: { children: ReactNode; as?: "div" | "ul" | "ol"; className?: string; role?: string; "aria-label"?: string; "aria-labelledby"?: string; id?: string }) {
  const still = useStill();
  const Tag = motion[as];
  return (
    <Tag className={className} variants={still ? undefined : staggerParent} initial={still ? false : "hidden"} whileInView="show" viewport={{ once: true, margin: "0px 0px -8% 0px" }} {...rest}>
      {children}
    </Tag>
  );
}

export function Item({ children, as = "div", className, style, ...rest }: { children: ReactNode; as?: "div" | "li" | "figure"; className?: string; style?: CSSProperties; lang?: string }) {
  const still = useStill();
  const Tag = motion[as];
  return (
    <Tag className={className} style={style} variants={still ? undefined : staggerChild} {...rest}>
      {children}
    </Tag>
  );
}

/* ------------------------------------------------------------------ magnetic */

/** Lightweight wrapper for buttons/links. */
export function Magnetic({ children, className = "" }: { children: ReactNode; strength?: number; className?: string }) {
  return <div className={`inline-flex ${className}`}>{children}</div>;
}

/* ------------------------------------------------------------------ tilt + spotlight */

type TiltProps = {
  children: ReactNode;
  max?: number;
  light?: string;
  radius?: number;
  className?: string;
  style?: CSSProperties;
  as?: "div" | "article" | "li";
  "aria-label"?: string;
  lang?: string;
};

/** Clean bento card container without expensive per-frame 3D perspective/radial-gradient repaints. */
export function TiltCard({ children, className = "", style, as = "div", ...rest }: TiltProps) {
  const Tag = as;
  return (
    <Tag className={className} style={style} {...rest}>
      {children}
    </Tag>
  );
}

/* ------------------------------------------------------------------ count up */

/** The first number in `value` counts up from 0 the first time it is seen; the rest of the text stays put. */
export function CountUp({ value, className, duration = 1.2 }: { value: string; className?: string; duration?: number }) {
  const still = useStill();
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -10% 0px" });
  const m = value.match(/\d[\d,]*(\.\d+)?/);
  const target = m ? parseFloat(m[0].replace(/,/g, "")) : 0;
  const decimals = m?.[1] ? m[1].length - 1 : 0;
  const before = m && m.index !== undefined ? value.slice(0, m.index) : "";
  const after = m && m.index !== undefined ? value.slice(m.index + m[0].length) : value;
  const [n, setN] = useState(target);

  useEffect(() => {
    if (still || !m || !inView) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / (duration * 1000));
      const eased = 1 - Math.pow(1 - p, 4);
      setN(target * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    setN(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, still]);

  const shown = m ? (m[0].includes(",") ? n.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : n.toFixed(decimals)) : "";
  return (
    <span ref={ref} className={`inline-block tabular-nums ${className ?? ""}`}>
      {before}
      {shown}
      {after}
    </span>
  );
}

/* ------------------------------------------------------------------ parallax */

export function useParallax(_ref: RefObject<HTMLElement | null>, _distance = 60): MotionValue<number> {
  return useMotionValue(0);
}

export function usePointerLean(_ref: RefObject<HTMLElement | null>, _spring = SOFT_SPRING) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  return { x, y };
}

export { motion, useScroll, useTransform, useSpring, useMotionValue, useMotionTemplate, useInView };
