"use client";

import { memo, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { Bot, ChevronDown, FileText, MoreHorizontal, Play, ShieldCheck, Trash2 } from "lucide-react";
import { Button } from "@/components/ui";
import type { VoiceView } from "./voiceApi";
import { coverageText, type IndianCoverage } from "./coverage";
import { engineLabel, formatDuration, languageLabel } from "./validate";

interface Props {
  voice: VoiceView;
  /** "Preview": opens a short sample for this voice. */
  onGenerate: (v: VoiceView) => void;
  /** "Withdraw consent and delete": the parent asks for confirmation, then deletes. */
  onDelete: (v: VoiceView) => void;
  /** Open the voice's consent record. */
  onShowConsent?: (v: VoiceView) => void;
  /** Indian languages this voice can speak (when DhVaani is set up on this server). */
  coverage?: IndianCoverage | null;
}

function formatDate(iso: string | null): string {
  if (!iso) return "Unknown date";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "Unknown date";
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

type Status = { label: string; tone: "good" | "warn" | "muted" };

/** What the card says about the voice: the engine name is not a status. */
export function voiceStatus(v: VoiceView): Status {
  const raw = v.raw as Record<string, unknown>;
  if (raw.withdrawn_at || raw.status === "withdrawn") return { label: "Withdrawn", tone: "muted" };
  if (v.consent) {
    return v.consentVerified === false
      ? { label: "Consent recorded, not checked", tone: "warn" }
      : { label: "Consent recorded", tone: "good" };
  }
  return { label: "Ready", tone: "good" };
}

const toneClass: Record<Status["tone"], string> = {
  good: "border-[var(--color-accent-success)]/30 bg-[var(--color-accent-success)]/10 text-[var(--color-accent-success)]",
  warn: "border-amber-500/40 bg-amber-500/10 text-amber-600",
  muted: "border-[var(--color-border)] bg-[var(--color-bg-elevated)] text-[var(--color-text-secondary)]",
};

const menuItem =
  "flex min-h-11 w-full items-center gap-2 rounded-lg px-3 text-left text-sm text-[var(--color-text-primary)] transition-colors hover:bg-[hsl(var(--au-text-primary)/0.06)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent)]";

/** The "..." menu: less common actions, with the destructive one last. */
function OverflowMenu({ voice, onDelete, onShowConsent, onOpenChange }: Pick<Props, "voice" | "onDelete" | "onShowConsent"> & { onOpenChange: (o: boolean) => void }) {
  const [open, setOpenState] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const setOpen = (o: boolean) => {
    setOpenState(o);
    onOpenChange(o);
  };

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        trigger.current?.focus();
      }
    };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  return (
    <div ref={root} className="relative shrink-0">
      <button
        ref={trigger}
        type="button"
        aria-label={`More actions for ${voice.name}`}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="inline-flex h-11 w-11 items-center justify-center rounded-full text-[var(--color-text-secondary)] transition-colors hover:bg-[hsl(var(--au-text-primary)/0.06)] hover:text-[var(--color-text-primary)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent)]"
      >
        <MoreHorizontal className="h-5 w-5" aria-hidden="true" />
      </button>
      {open && (
        <div
          role="menu"
          aria-label={`Actions for ${voice.name}`}
          className="absolute right-0 top-full z-30 mt-1 w-64 rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-surface))] p-1 shadow-xl"
        >
          {onShowConsent && voice.consent && (
            <button
              type="button"
              role="menuitem"
              className={menuItem}
              onClick={() => {
                setOpen(false);
                onShowConsent(voice);
              }}
            >
              <FileText className="h-4 w-4" aria-hidden="true" />
              Consent record
            </button>
          )}
          <button
            type="button"
            role="menuitem"
            aria-label={`Withdraw consent and delete ${voice.name}`}
            className={`${menuItem} !text-[var(--color-accent-error)] hover:!bg-red-500/10`}
            onClick={() => {
              setOpen(false);
              onDelete(voice);
            }}
          >
            <Trash2 className="h-4 w-4" aria-hidden="true" />
            Withdraw consent and delete
          </button>
        </div>
      )}
    </div>
  );
}

export const VoiceCard = memo(function VoiceCard({ voice, onGenerate, onDelete, onShowConsent, coverage }: Props) {
  const duration = formatDuration(voice.durationSec);
  const status = voiceStatus(voice);
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <article
      className={`relative flex h-full flex-col rounded-2xl border border-[var(--color-border)] bg-[hsl(var(--au-surface)/0.6)] p-4 transition-[box-shadow,border-color] duration-300 hover:border-[hsl(var(--au-accent)/0.5)] focus-within:border-[hsl(var(--au-accent)/0.5)] sm:p-5 ${
        menuOpen ? "z-20" : ""
      }`}
    >
      <div className="flex items-start gap-3">
        <span
          aria-hidden="true"
          className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-[hsl(var(--au-accent)/0.18)] text-base font-semibold text-[var(--color-text-primary)]"
        >
          {voice.name.trim().charAt(0).toUpperCase() || "?"}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-lg font-semibold tracking-tight text-[var(--color-text-primary)]" title={voice.name}>
            {voice.name}
          </h3>
          <span
            data-testid="voice-status"
            className={`mt-1 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${toneClass[status.tone]}`}
          >
            <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
            {status.label}
          </span>
        </div>
        <OverflowMenu voice={voice} onDelete={onDelete} onShowConsent={onShowConsent} onOpenChange={setMenuOpen} />
      </div>

      <p className="mt-3 text-sm text-[var(--color-text-secondary)]" data-testid="voice-languages">
        <span className="font-medium text-[var(--color-text-primary)]">{languageLabel(voice.language)}</span>
        {coverage ? <span className="mt-0.5 block">Also speaks {coverageText(coverage)}</span> : null}
      </p>

      <div className="mt-4 flex flex-col gap-2 sm:flex-row">
        <Button variant="outline" className="h-11 sm:flex-none sm:px-5" onClick={() => onGenerate(voice)} aria-label={`Preview ${voice.name}`}>
          <Play className="h-4 w-4" aria-hidden="true" />
          Preview
        </Button>
        <Link
          href={`/dashboard/studio/agents?voice=${encodeURIComponent(voice.id)}`}
          onClick={() => toast.info(`In your agent's Voice box, pick "${voice.name}".`)}
          className="inline-flex h-11 w-full shrink-0 items-center justify-center gap-2 whitespace-nowrap sm:w-auto sm:flex-1 rounded-full bg-brand px-4 text-sm font-medium text-[var(--color-on-accent)] transition-opacity hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent)]"
        >
          <Bot className="h-4 w-4" aria-hidden="true" />
          Use for an agent
        </Link>
      </div>

      <details className="group/details mt-3">
        <summary className="flex min-h-11 cursor-pointer list-none items-center gap-1.5 rounded-lg px-1 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent)] [&::-webkit-details-marker]:hidden">
          <ChevronDown className="h-4 w-4 transition-transform group-open/details:rotate-180 motion-reduce:transition-none" aria-hidden="true" />
          Details
        </summary>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 px-1 pb-1 text-sm text-[var(--color-text-secondary)]">
          <dt>Engine</dt>
          <dd className="text-[var(--color-text-primary)]">{engineLabel(voice.engine)}</dd>
          <dt>Added</dt>
          <dd className="text-[var(--color-text-primary)]">{formatDate(voice.createdAt)}</dd>
          {duration && (
            <>
              <dt>Sample length</dt>
              <dd className="tabular-nums text-[var(--color-text-primary)]">{duration}</dd>
            </>
          )}
          {coverage && (
            <>
              <dt>Indian languages</dt>
              <dd className="text-[var(--color-text-primary)]">{coverage.codes.length} through DhVaani (ARTPARK-IISc)</dd>
            </>
          )}
        </dl>
      </details>
    </article>
  );
});
