import { Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

export const AI_GENERATED_TOOLTIP = "Synthetic speech made with BonClay";

/** Small label for synthetic audio. Show it wherever generated or agent speech can be played. */
export function AiGeneratedTag({ className }: { className?: string }) {
  return (
    <span
      title={AI_GENERATED_TOOLTIP}
      aria-label={`AI-generated. ${AI_GENERATED_TOOLTIP}`}
      data-testid="ai-generated-tag"
      className={cn(
        "inline-flex shrink-0 cursor-help items-center gap-1 rounded-md border border-[var(--color-border)] px-1.5 py-0.5 text-[10px] font-medium leading-none tracking-wide text-[var(--color-text-secondary)]",
        className
      )}
    >
      <Sparkles className="size-2.5" aria-hidden />
      AI-generated
    </span>
  );
}

export default AiGeneratedTag;
