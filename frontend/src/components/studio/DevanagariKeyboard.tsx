"use client";

import { Delete } from "lucide-react";

const ROWS: { label: string; keys: string[] }[] = [
  { label: "Vowels", keys: ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ए", "ऐ", "ओ", "औ", "अं", "अः"] },
  { label: "Matras", keys: ["ा", "ि", "ी", "ु", "ू", "ृ", "े", "ै", "ो", "ौ", "ं", "ः", "ँ", "्"] },
  { label: "Consonants", keys: ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"] },
  { label: "Digits and marks", keys: ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९", "।", "॥"] },
];

interface Props {
  onInsert: (s: string) => void;
  onBackspace: () => void;
}

export function DevanagariKeyboard({ onInsert, onBackspace }: Props) {
  return (
    <div
      role="group"
      aria-label="Devanagari keyboard"
      className="space-y-2 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-3"
    >
      {ROWS.map((row) => (
        <div key={row.label}>
          <p className="mb-1 text-[11px] text-[var(--color-text-secondary)]">{row.label}</p>
          <div lang="hi" className="flex flex-wrap gap-1">
            {row.keys.map((k) => (
              <button
                key={k}
                type="button"
                // Keep focus (and caret) in the textarea.
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => onInsert(k)}
                className="h-10 min-w-10 rounded-lg border border-[var(--color-border)] px-2 text-base text-[var(--color-text-primary)] transition-colors hover:bg-[hsl(var(--au-text-primary)/0.08)] active:scale-95 focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
              >
                {k}
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="flex gap-1">
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => onInsert(" ")}
          className="h-10 flex-1 rounded-lg border border-[var(--color-border)] text-xs text-[var(--color-text-secondary)] hover:bg-[hsl(var(--au-text-primary)/0.08)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
        >
          Space
        </button>
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={onBackspace}
          aria-label="Delete previous character"
          className="grid h-10 w-14 place-items-center rounded-lg border border-[var(--color-border)] text-[var(--color-text-primary)] hover:bg-[hsl(var(--au-text-primary)/0.08)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
        >
          <Delete className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
