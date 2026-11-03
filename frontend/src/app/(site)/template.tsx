"use client";

import type { ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";

/** Next.js remounts a template on every navigation; uses compositor-only opacity + y (no full-page blur filter). */
export default function Template({ children }: { children: ReactNode }) {
  const reduced = useReducedMotion();
  if (reduced) return <div>{children}</div>;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}
