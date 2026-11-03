"use client";

import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Copy } from "lucide-react";
import { useSetupHints } from "@/hooks/useApi";
import { cn } from "@/lib/utils";
import type { SetupHint, SetupHintSeverity } from "@/types/api";

const DOT: Record<SetupHintSeverity, string> = {
  error: "bg-accent-error",
  warning: "bg-accent-warning",
  info: "bg-accent-secondary",
};

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
    } catch {
      return; // clipboard blocked: the command is still selectable
    }
    setCopied(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mt-2 flex items-center gap-2 rounded-lg border border-border bg-bg-elevated px-3 py-1.5">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap font-mono text-xs text-text-primary">{command}</code>
      <button
        type="button"
        onClick={copy}
        aria-label={`Copy command: ${command}`}
        className="inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary"
      >
        {copied ? <Check className="size-3.5 text-accent-success" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
        <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
      </button>
    </div>
  );
}

/** What each backend hint means for the owner, in plain words. Unknown hints fall back to the backend's own text. */
const PLAIN: Record<string, { title: string; text: string }> = {
  no_llm: { title: "Your agents cannot answer yet", text: "Agents need a language model to reply to people. Ask the person who installed BonClay to set one up." },
  no_whisper: { title: "BonClay cannot listen to speech yet", text: "Listening to recordings and checking consent needs a speech tool that is not installed." },
  no_ffmpeg: { title: "An audio tool is missing", text: "Uploading and playing audio will fail until it is installed." },
  no_espeak: { title: "Some voices may mispronounce words", text: "A small pronunciation tool is missing. Most voices still work." },
  no_mlx: { title: "Voice cloning will be slow on this computer", text: "It works, but takes much longer than it should." },
  supabase_unreachable: { title: "Online saving is not working", text: "Check the internet connection, or switch to saving on this computer." },
  no_indic_speak: { title: "Indian-language voices are limited", text: "Only Hindi is available in preset voices until an extra voice pack is added." },
  no_chatterbox: { title: "Cloned voices cannot speak Hindi yet", text: "An extra voice pack is needed. It is for audio clips, not live calls." },
  no_dhvaani: { title: "Cloned voices only speak Hindi for now", text: "An extra voice pack is needed for other Indian languages." },
  no_groq: { title: "Answers use the built-in model", text: "A faster, stronger online model can be added." },
};

function HintRow({ hint }: { hint: SetupHint }) {
  const plain = PLAIN[hint.id];
  return (
    <li className="flex items-start gap-3 py-3">
      <span aria-hidden className={cn("mt-1.5 size-2 shrink-0 rounded-full", DOT[hint.severity])} />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-text-primary">
          <span className="sr-only">{hint.severity}: </span>
          {plain?.title ?? hint.title}
        </p>
        <p className="text-xs text-text-secondary">{plain?.text ?? hint.detail}</p>
        <details className="group mt-1">
          <summary className="inline-flex min-h-8 cursor-pointer list-none items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary">
            <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" aria-hidden /> Details
          </summary>
          {plain && <p className="mt-1 text-xs text-text-secondary">{hint.title}. {hint.detail}</p>}
          {hint.command && (
            <>
              <p className="mt-1 text-xs text-text-secondary">Fix:</p>
              <CopyCommand command={hint.command} />
            </>
          )}
        </details>
      </div>
    </li>
  );
}

const hintText = (h: SetupHint) => `${h.title}. ${h.detail}${h.command ? ` Fix: ${h.command}` : ""}`;

/** "Setup needed" card: what is not working in plain words, with the technical fix tucked behind Details. */
export function SetupHints() {
  const { data } = useSetupHints();
  const [notesOpen, setNotesOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const hints = data?.hints ?? [];
  const problems = hints.filter((h) => h.severity === "error" || h.severity === "warning");
  const notes = hints.filter((h) => h.severity === "info");
  if (problems.length === 0) return null;

  const copyAll = async () => {
    try {
      await navigator.clipboard.writeText(problems.map(hintText).join("\n"));
    } catch {
      return;
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section aria-labelledby="setup-hints-title" className="rounded-xl border border-border bg-bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id="setup-hints-title" className="text-sm font-semibold text-text-primary">
            Setup needed
          </h2>
          <p className="mt-0.5 max-w-prose text-xs text-text-secondary">
            {problems.length === 1 ? "One thing" : `${problems.length} things`} must be set up before everything works. If you did not install
            BonClay yourself, send the details to the person who did.
          </p>
        </div>
        <button
          type="button"
          onClick={copyAll}
          className="inline-flex min-h-10 items-center gap-1.5 rounded-full border border-border px-3.5 text-xs font-medium text-text-primary transition-colors hover:border-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
        >
          {copied ? <Check className="size-3.5 text-accent-success" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
          <span aria-live="polite">{copied ? "Copied" : "Copy details for your IT person"}</span>
        </button>
      </div>
      <ul className="mt-1 divide-y divide-border">
        {problems.map((h) => (
          <HintRow key={h.id} hint={h} />
        ))}
      </ul>
      {notes.length > 0 && (
        <div className="mt-2 border-t border-border pt-3">
          <button
            type="button"
            onClick={() => setNotesOpen((o) => !o)}
            aria-expanded={notesOpen}
            className="inline-flex min-h-10 items-center gap-1.5 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary"
          >
            <ChevronDown className={cn("size-3.5 transition-transform", notesOpen && "rotate-180")} aria-hidden />
            Optional improvements ({notes.length})
          </button>
          {notesOpen && (
            <ul className="mt-1 divide-y divide-border">
              {notes.map((h) => (
                <HintRow key={h.id} hint={h} />
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
