"use client";

import { LANGUAGE_OPTIONS, languageName } from "./languages";

/** Pick languages by name. The value is the same list of ISO codes the agent already stores. */
export function LanguagePicker({ value, onChange, id }: { value: string[]; onChange: (codes: string[]) => void; id?: string }) {
  const known = new Set(LANGUAGE_OPTIONS.map((l) => l.code));
  // A code saved earlier that is not in the list stays visible and removable instead of vanishing.
  const extra = value.filter((c) => !known.has(c));
  const options = [...LANGUAGE_OPTIONS, ...extra.map((c) => ({ code: c, name: languageName(c) }))];
  const toggle = (code: string) => onChange(value.includes(code) ? value.filter((c) => c !== code) : [...value, code]);

  return (
    <div id={id} role="group" aria-label="Languages" className="flex flex-wrap gap-2">
      {options.map((o) => {
        const on = value.includes(o.code);
        return (
          <button
            key={o.code}
            type="button"
            aria-pressed={on}
            onClick={() => toggle(o.code)}
            className={`min-h-10 rounded-full border px-3.5 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-primary ${
              on
                ? "border-accent-primary bg-accent-primary/15 font-medium text-text-primary"
                : "border-border bg-bg-surface text-text-secondary hover:border-text-primary hover:text-text-primary"
            }`}
          >
            {o.name}
          </button>
        );
      })}
    </div>
  );
}
