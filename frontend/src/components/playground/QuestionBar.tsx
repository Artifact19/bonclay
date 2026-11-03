"use client";

import { useState } from "react";
import { Loader2, Send } from "lucide-react";

/** Clickable example questions. Clicking one sends it straight away. */
export function SuggestedQuestions({
  questions,
  onPick,
  disabled,
  className = "",
}: {
  questions: string[];
  onPick: (q: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  if (!questions.length) return null;
  return (
    <ul
      className={`flex flex-wrap justify-center gap-2 ${className}`}
      aria-label="Suggested questions"
    >
      {questions.map((q) => (
        <li key={q} className="max-w-full">
          <button
            type="button"
            disabled={disabled}
            onClick={() => onPick(q)}
            className="min-h-10 max-w-full rounded-full border border-[var(--color-border)] bg-bg-base px-4 py-2 text-left text-sm text-[var(--color-text-primary)] transition-colors hover:bg-[var(--color-bg-elevated)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {q}
          </button>
        </li>
      ))}
    </ul>
  );
}

/** "Type a question": a silent way to test an agent (noisy office, no microphone). */
export function TypeQuestion({
  onSend,
  disabled,
  busy,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
  busy?: boolean;
}) {
  const [text, setText] = useState("");
  const send = () => {
    const t = text.trim();
    if (!t || disabled || busy) return;
    onSend(t);
    setText("");
  };
  return (
    <form
      className="flex min-w-0 flex-1 items-center gap-1 rounded-full border border-border bg-bg-elevated py-1 pl-4 pr-1 transition-colors focus-within:border-[#F0A23A]/70"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <label htmlFor="typed-question" className="sr-only">
        Type a question
      </label>
      <input
        id="typed-question"
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
        maxLength={1000}
        autoComplete="off"
        enterKeyHint="send"
        placeholder="Type a question"
        className="h-10 min-w-0 flex-1 border-0 bg-transparent text-base text-[var(--color-text-primary)] outline-none placeholder:text-[var(--color-text-muted)] disabled:cursor-not-allowed disabled:opacity-40 sm:text-sm"
      />
      <button
        type="submit"
        disabled={disabled || busy || !text.trim()}
        className="inline-flex h-10 shrink-0 items-center justify-center gap-1.5 rounded-full bg-[#F0A23A] px-3 text-sm font-medium text-[#141008] transition-opacity focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)] disabled:cursor-not-allowed disabled:opacity-40 sm:px-4"
      >
        {busy ? (
          <Loader2
            className="h-4 w-4 motion-safe:animate-spin"
            aria-hidden="true"
          />
        ) : (
          <Send className="h-4 w-4" aria-hidden="true" />
        )}
        <span className="sr-only sm:not-sr-only">Send</span>
      </button>
    </form>
  );
}
