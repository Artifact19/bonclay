"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { CheckCircle2, FileUp, Loader2, X } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui";
import { friendlyUploadError, PDF_TEXT_HINT } from "@/lib/uploadErrors";

export interface UploadResult {
  doc_id: string;
  chunks: number;
}

const MAX_BYTES = 20 * 1024 * 1024;
const ACCEPT = ".pdf,.txt,application/pdf,text/plain";

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; name: string; pct: number }
  | { kind: "indexing"; name: string }
  | { kind: "done"; name: string; chunks: number }
  | { kind: "error"; name: string; message: string };

function validate(file: File): string | null {
  const ext = file.name.toLowerCase().split(".").pop();
  if (ext !== "pdf" && ext !== "txt") return "Only PDF and TXT files can be indexed.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > MAX_BYTES) return "File is over the 20 MB limit.";
  return null;
}

export default function UploadZone({ onUploaded }: { onUploaded: (r: UploadResult, name: string) => void }) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const busy = phase.kind === "uploading" || phase.kind === "indexing";

  const start = useCallback(
    async (file: File) => {
      const invalid = validate(file);
      if (invalid) {
        setPhase({ kind: "error", name: file.name, message: invalid });
        toast.error(invalid);
        return;
      }
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setPhase({ kind: "uploading", name: file.name, pct: 0 });
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await api.upload<UploadResult>("/api/rag/upload", fd, {
          signal: ctrl.signal,
          onProgress: (pct) =>
            setPhase((p) =>
              p.kind === "uploading" || p.kind === "indexing"
                ? pct >= 100
                  ? { kind: "indexing", name: file.name }
                  : { kind: "uploading", name: file.name, pct }
                : p
            ),
        });
        setPhase({ kind: "done", name: file.name, chunks: res.chunks });
        toast.success(`${file.name} indexed into ${res.chunks} chunks`);
        onUploaded(res, file.name);
      } catch (e) {
        if (e instanceof ApiError && e.code === "aborted") {
          setPhase({ kind: "idle" });
          return;
        }
        const message = friendlyUploadError(e, file.name, "Upload failed");
        setPhase({ kind: "error", name: file.name, message });
        toast.error(message);
      } finally {
        abortRef.current = null;
        if (inputRef.current) inputRef.current.value = "";
      }
    },
    [onUploaded]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (busy) return;
    const f = e.dataTransfer.files?.[0];
    if (f) void start(f);
  };

  const pct = phase.kind === "uploading" ? phase.pct : phase.kind === "indexing" ? 100 : 0;

  return (
    <div>
      <label htmlFor={inputId} className="sr-only">
        Choose a PDF or TXT document to add (up to 20 MB)
      </label>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={ACCEPT}
        tabIndex={-1}
        disabled={busy}
        className="sr-only"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void start(f);
        }}
      />
      <div
        role="button"
        tabIndex={busy ? -1 : 0}
        aria-disabled={busy}
        aria-label="Upload a PDF or TXT document"
        onClick={() => !busy && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!busy && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
        }}
        onDrop={onDrop}
        className={`flex min-h-[168px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-4 py-6 text-center transition-colors outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)] ${
          busy ? "cursor-default" : "cursor-pointer"
        } ${
          dragging
            ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
            : "border-[var(--color-border)] hover:border-[var(--color-accent)]"
        }`}
      >
        {phase.kind === "uploading" || phase.kind === "indexing" ? (
          <div className="w-full max-w-xs rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-surface)] p-3.5 text-left shadow-xs">
            <div className="flex items-center justify-between gap-2">
              <p className="truncate text-sm font-medium text-[var(--color-text-primary)]">{phase.name}</p>
              <span className="inline-flex shrink-0 items-center rounded-full border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-2 py-0.5 font-mono text-xs font-semibold tabular-nums text-[var(--color-text-primary)]">
                {pct}%
              </span>
            </div>
            <div
              className="mt-2.5 h-2.5 w-full overflow-hidden rounded-full border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-[1px]"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={phase.kind === "uploading" ? pct : undefined}
              aria-label="Upload progress"
            >
              <div
                className={`h-full rounded-full bg-[var(--color-accent)] transition-[width] duration-200 ${
                  phase.kind === "indexing" ? "animate-pulse motion-reduce:animate-none" : ""
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="mt-2.5 flex items-center justify-between gap-2">
              <p className="flex items-center gap-1.5 text-xs text-[var(--color-text-secondary)]">
                <Loader2 className="size-3.5 animate-spin motion-reduce:animate-none text-[var(--color-accent)]" aria-hidden />
                {phase.kind === "uploading" ? `Uploading ${pct}%` : "Indexing: reading and embedding passages"}
              </p>
              <Button
                variant="ghost"
                size="sm"
                onClick={(ev) => {
                  ev.stopPropagation();
                  abortRef.current?.abort();
                }}
                leftIcon={<X className="size-3.5" aria-hidden />}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <>
            <FileUp className="size-7 text-[var(--color-text-secondary)]" aria-hidden />
            <p className="text-sm font-medium text-[var(--color-text-primary)]">
              {dragging ? "Drop to upload" : "Drag a file here, or click to choose"}
            </p>
            <p className="text-xs text-[var(--color-text-secondary)]">PDF or TXT, up to 20 MB</p>
          </>
        )}
      </div>

      <p data-testid="pdf-text-hint" className="mt-2 text-xs text-[var(--color-text-secondary)]">
        {PDF_TEXT_HINT}
      </p>

      <div role="status" aria-live="polite" className="mt-2 min-h-5 text-xs">
        {(phase.kind === "uploading" || phase.kind === "indexing") && (
          <span className="sr-only">
            {phase.kind === "uploading" ? `Uploading ${phase.name}` : `Reading ${phase.name}`}
          </span>
        )}
        {phase.kind === "done" && (
          <p className="flex items-center gap-1.5 text-[var(--color-success)]">
            <CheckCircle2 className="size-3.5 shrink-0" aria-hidden />
            <span className="truncate">
              {phase.name}: {phase.chunks} chunks indexed
            </span>
          </p>
        )}
        {phase.kind === "error" && <p className="text-accent-error">{phase.message}</p>}
      </div>
    </div>
  );
}
