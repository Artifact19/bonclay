"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AlertCircle, ShieldCheck, FileText, RotateCcw, Volume2 } from "lucide-react";
import { Badge, Button, Card } from "@/components/ui";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { API_URL, api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { NATIVE_NAMES, ProgressBar, formatEta, ordinal } from "./progress";
import type { ItemStage, SayItem } from "./types";

export const MAX_TRANSLATION_CHARS = 800;

/** The stage of a language, from the server's `stage` or (older servers) from its status. */
function stageOf(item: SayItem): ItemStage {
  if (item.stage) return item.stage;
  if (item.status === "queued") return "waiting";
  return item.status;
}

function stageLabel(item: SayItem, stage: ItemStage): string {
  switch (stage) {
    case "waiting":
      return item.queue_position == null
        ? "Waiting"
        : item.queue_position === 1
          ? "Waiting, up next"
          : `Waiting, ${ordinal(item.queue_position)} in line`;
    case "translating":
      return "Translating";
    case "speaking":
      return "Generating voice";
    case "finishing":
      return "Watermarking and saving";
    case "done":
      return item.seconds != null ? `Done in ${Math.round(item.seconds)} s` : "Done";
    case "failed":
      return "Failed";
    case "unavailable":
      return "Not available here";
    case "cancelled":
      return "Cancelled";
  }
}

export function VoiceKindBadge({ kind }: { kind: SayItem["voice_kind"] }) {
  if (kind === "your_voice") {
    return (
      <Badge
        data-testid="badge-your-voice"
        className="border-accent-primary/40 bg-accent-primary/15 text-accent-secondary"
      >
        Your voice
      </Badge>
    );
  }
  return <Badge data-testid="badge-preset-voice">Preset voice</Badge>;
}

/** Plays a clip: a cloud clip has a public URL; a local one needs the auth and storage headers, so it is fetched. */
function ClipPlayer({ url, filename }: { url: string; filename: string }) {
  const absolute = /^https?:\/\//i.test(url);
  const [blob, setBlob] = useState<Blob | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (absolute) return;
    let cancelled = false;
    api
      .blob(url, undefined, { retries: 1 })
      .then((b) => !cancelled && setBlob(b))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [url, absolute]);

  if (failed) return <p className="text-sm text-text-secondary">The audio could not be loaded.</p>;
  const src = absolute ? url : blob;
  if (!src) return <div className="h-[110px] animate-pulse rounded-2xl bg-bg-elevated" aria-label="Loading audio" />;
  return <WaveformPlayer src={src} filename={filename} height={56} aiGenerated />;
}

export function LanguageCard({
  item,
  text,
  onTextChange,
  onRespeak,
  canAct,
}: {
  item: SayItem;
  /** The translation as shown (the user's edit when there is one). */
  text: string;
  onTextChange: (value: string) => void;
  /** Speak this language again from the text as shown, without translating. */
  onRespeak: () => void;
  /** False while any language is being worked on: one heavy engine at a time. */
  canAct: boolean;
}) {
  const stage = stageOf(item);
  const finished = stage === "done" || stage === "failed" || stage === "cancelled" || stage === "unavailable";
  const working = !finished;
  const progress = item.progress ?? (stage === "done" ? 100 : 0);
  const slow = working && !!item.slow;
  const eta = working && !slow ? formatEta(item.eta_s) : null;
  const native = NATIVE_NAMES[item.language];
  const editable = (stage === "done" || stage === "failed") && item.text != null;
  const edited = editable && text.trim() !== (item.text ?? "").trim();
  const canRespeak = canAct && text.trim().length > 0;
  const tone = stage === "failed" ? "error" : stage === "unavailable" || stage === "cancelled" ? "muted" : "accent";

  return (
    <Card variant="solid" data-testid={`result-${item.language}`} data-stage={stage} className="flex flex-col gap-3">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="font-display text-lg font-medium text-text-primary">
            {item.language_name}
            {native && native !== item.language_name && (
              <span lang={item.language} className="ml-2 text-base font-normal text-text-secondary">
                {native}
              </span>
            )}
          </h3>
          {item.voice_kind && <VoiceKindBadge kind={item.voice_kind} />}
        </div>
        {(item.engine_name || item.attribution) && (
          <p className="mt-1 text-xs text-text-secondary">
            {item.engine_name}
            {item.engine_name && item.attribution ? ". " : ""}
            {item.attribution}
          </p>
        )}
      </div>

      {editable ? (
        <div>
          <label htmlFor={`edit-${item.language}`} className="sr-only">
            {item.language_name} translation
          </label>
          <textarea
            id={`edit-${item.language}`}
            data-testid={`edit-${item.language}`}
            lang={item.language}
            value={text}
            rows={2}
            maxLength={MAX_TRANSLATION_CHARS}
            disabled={!canAct}
            onChange={(e) => onTextChange(e.target.value)}
            className="w-full resize-y rounded-xl border border-border bg-bg-base px-3 py-2 text-[17px] leading-relaxed text-text-primary"
          />
          {stage === "done" && (
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <Button size="sm" variant={edited ? "solid" : "outline"} onClick={onRespeak} disabled={!canRespeak}
                leftIcon={<Volume2 className="size-3.5" aria-hidden />}>
                Re-speak
              </Button>
              <span className="text-xs text-text-secondary">
                {edited ? "Speaks your edited text in this language only." : "Edit the text to speak it again."}
              </span>
            </div>
          )}
        </div>
      ) : item.text ? (
        <p lang={item.language} className="text-[17px] leading-relaxed text-text-primary" data-testid={`text-${item.language}`}>
          {item.text}
        </p>
      ) : stage === "waiting" || stage === "translating" ? (
        <p className="text-sm italic text-text-secondary" data-testid={`text-${item.language}`}>
          {stage === "translating" ? "Translating…" : "The translation appears here."}
        </p>
      ) : null}

      {stage !== "unavailable" && (
        <div data-testid={`status-${item.language}`}>
          <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
            <span role="status" className={cn("font-medium", stage === "failed" ? "text-accent-error" : "text-text-primary")}>
              {stageLabel(item, stage)}
            </span>
            <span className="tabular-nums text-text-secondary">
              <span data-testid={`pct-${item.language}`}>{Math.round(progress)}%</span>
              {eta && <span className="ml-2 text-xs">{eta}</span>}
            </span>
          </div>
          {slow && (
            <p data-testid={`slow-${item.language}`} className="mb-1.5 text-xs text-accent-warning">
              Taking longer than expected. The laptop is probably busy; it is still working.
            </p>
          )}
          <ProgressBar value={progress} label={`${item.language_name} progress`} tone={tone} testId={`progress-${item.language}`} />
        </div>
      )}

      {(stage === "failed" || stage === "unavailable" || stage === "cancelled") && (
        <div className="flex flex-col gap-2">
          {stage !== "cancelled" && (
            <p role="alert" className={cn("flex items-start gap-2 text-sm", stage === "failed" ? "text-accent-error" : "text-text-secondary")}>
              <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
              <span>
                {stage === "unavailable" ? "Not available on this server. " : ""}
                {item.error || "This language could not be generated."}
              </span>
            </p>
          )}
          {stage !== "unavailable" && (
            <div>
              <Button size="sm" variant="outline" onClick={onRespeak} disabled={!canAct}
                leftIcon={<RotateCcw className="size-3.5" aria-hidden />}>
                {stage === "cancelled" ? "Speak this language" : "Retry"}
              </Button>
            </div>
          )}
        </div>
      )}

      {stage === "done" && item.audio_url && (
        <ClipPlayer url={item.audio_url} filename={`say-it-once-${item.language}.wav`} />
      )}

      {stage === "done" && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          <Link href="/verify" className="inline-flex items-center gap-1.5 font-medium text-accent-secondary hover:underline">
            <ShieldCheck className="size-4" aria-hidden />
            Verify
          </Link>
          {item.receipt_id && (
            <a
              href={`${API_URL}/api/receipts/${encodeURIComponent(item.receipt_id)}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 font-medium text-accent-secondary hover:underline"
            >
              <FileText className="size-4" aria-hidden />
              Receipt
            </a>
          )}
          <span className="text-xs text-text-secondary">Watermarked, with a signed receipt.</span>
        </div>
      )}
    </Card>
  );
}
