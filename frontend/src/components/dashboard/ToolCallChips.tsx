import { Wrench } from "lucide-react";
import type { ToolCall } from "@/types/api";

/** What the agent did with its tools during a reply, one chip per call: "Used tool: book_appointment → booking A12 ...".
 *  The wording carries the outcome (used / failed), so it never relies on colour alone. */
export function ToolCallChips({ calls }: { calls?: ToolCall[] }) {
  if (!calls?.length) return null;
  return (
    <ul aria-label="Tools used" className="flex flex-col items-start gap-1.5">
      {calls.map((c, i) => (
        <li
          key={`${c.name}-${i}`}
          data-testid="tool-call-chip"
          className={`inline-flex max-w-full items-start gap-1.5 rounded-md border px-2 py-1 text-xs ${
            c.ok
              ? "border-[var(--color-border)] bg-[var(--color-bg-elevated)] text-[var(--color-text-secondary)]"
              : "border-[var(--color-accent-error)] text-[var(--color-text-primary)]"
          }`}
        >
          <Wrench className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
          <span className="min-w-0 break-words">
            {c.ok ? "Used tool" : "Tool failed"}: <span className="font-mono">{c.name}</span>
            {c.result ? ` → ${c.result}` : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}
