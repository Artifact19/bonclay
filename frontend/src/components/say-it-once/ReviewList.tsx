"use client";

import { AlertTriangle } from "lucide-react";
import { Badge, Card } from "@/components/ui";
import { MAX_TRANSLATION_CHARS, VoiceKindBadge } from "./LanguageCard";
import type { ReviewItem } from "./types";

/** One editable translation per language, with the voice it will be spoken in and any warnings about it. */
export function ReviewList({
  items,
  edits,
  onEdit,
  disabled,
}: {
  items: ReviewItem[];
  edits: Record<string, string>;
  onEdit: (language: string, value: string) => void;
  disabled?: boolean;
}) {
  return (
    <section aria-label="Review translations" className="grid gap-4 md:grid-cols-2">
      {items.map((it) => {
        const value = edits[it.language] ?? it.translation;
        const id = `review-${it.language}`;
        return (
          <Card key={it.language} variant="solid" data-testid={`review-${it.language}`} className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <label htmlFor={id} className="text-base font-semibold text-text-primary">
                {it.language_name}
              </label>
              {it.voice && <VoiceKindBadge kind={it.voice === "yours" ? "your_voice" : "preset_voice"} />}
              {it.engine_name && <Badge>{it.engine_name}</Badge>}
            </div>

            {it.available ? (
              <textarea
                id={id}
                value={value}
                rows={3}
                maxLength={MAX_TRANSLATION_CHARS}
                disabled={disabled}
                onChange={(e) => onEdit(it.language, e.target.value)}
                className="w-full resize-y rounded-xl border border-border bg-bg-base px-3 py-2.5 text-[15px] text-text-primary"
              />
            ) : (
              <p id={id} className="text-sm text-text-secondary">
                {it.reason ?? "This language cannot be spoken here."}
              </p>
            )}

            {it.available &&
              value === it.translation && // a warning is about the model's text: once the user edits it, it no longer applies
              it.warnings.map((w) => (
                <p key={w} role="alert" data-testid={`warning-${it.language}`} className="flex items-start gap-2 text-sm text-accent-warning">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
                  <span>{w}</span>
                </p>
              ))}
            {it.available && it.attribution && <p className="text-xs text-text-secondary">{it.attribution}</p>}
          </Card>
        );
      })}
    </section>
  );
}
