"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { FileText } from "lucide-react";
import { Skeleton } from "@/components/ui";
import { PageShell } from "@/components/ui/PageHeader";
import { SpeechPanel } from "@/components/studio/SpeechPanel";
import { DialogueBuilder } from "@/components/studio/DialogueBuilder";

type Tab = "speech" | "dialogue";

const pillCls =
  "relative inline-flex h-9 items-center rounded-full px-4 text-sm font-medium transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)]";

export default function StudioPage() {
  const [tab, setTab] = useState<Tab>("speech");
  const reduce = useReducedMotion();

  return (
    <PageShell width="full">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <h1 className="font-display text-[clamp(1.75rem,2.6vw,2.5rem)] font-medium leading-[1.08] tracking-[-0.03em] text-text-primary">
            Studio
          </h1>
          <p className="mt-2 max-w-prose text-[15px] leading-relaxed text-text-secondary">
            Turn a script into speech, or stage a conversation between several
            voices.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <div
            role="tablist"
            aria-label="Studio modes"
            className="inline-flex rounded-full bg-bg-elevated p-1"
          >
            {(
              [
                ["speech", "Speech"],
                ["dialogue", "Dialogue"],
              ] as [Tab, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                id={`tab-${id}`}
                aria-selected={tab === id}
                aria-controls={`panel-${id}`}
                type="button"
                onClick={() => setTab(id)}
                className={`${pillCls} ${tab === id ? "text-text-primary" : "text-text-secondary hover:text-text-primary"}`}
              >
                {tab === id && (
                  <motion.span
                    layoutId="studio-mode"
                    aria-hidden="true"
                    className="absolute inset-0 rounded-full bg-bg-surface shadow-sm"
                    transition={
                      reduce
                        ? { duration: 0 }
                        : { type: "spring", stiffness: 400, damping: 32 }
                    }
                  />
                )}
                <span className="relative">{label}</span>
              </button>
            ))}
          </div>
          <Link
            href="/dashboard/knowledge/try"
            className="inline-flex h-10 items-center gap-1.5 whitespace-nowrap rounded-full border border-border px-4 text-sm font-medium text-text-secondary transition-colors hover:text-text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)]"
          >
            <FileText className="h-4 w-4" aria-hidden="true" />
            Document Q&amp;A
          </Link>
        </div>
      </header>

      {/* Both panels stay mounted and the inactive one is hidden, so the script, the voice and a running render survive a
          trip to the other tab. */}
      <div
        role="tabpanel"
        id="panel-speech"
        aria-labelledby="tab-speech"
        hidden={tab !== "speech"}
      >
        <Suspense fallback={<Skeleton className="h-64 rounded-2xl" />}>
          <SpeechPanel />
        </Suspense>
      </div>
      <div
        role="tabpanel"
        id="panel-dialogue"
        aria-labelledby="tab-dialogue"
        hidden={tab !== "dialogue"}
      >
        <DialogueBuilder />
      </div>
    </PageShell>
  );
}
