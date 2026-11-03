"use client";

import { useId, useRef, useState } from "react";
import Link from "next/link";
import { AlertCircle, CheckCircle2, FileText, Loader2, RotateCcw, Trash2, UploadCloud, X } from "lucide-react";
import { toast } from "sonner";
import { Badge, Card, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useAgentDocuments, useAgents, useExampleQuestions } from "@/hooks/useApi";
import { errMessage, relativeTime } from "@/components/dashboard/format";
import { friendlyUploadError, PDF_TEXT_HINT } from "@/lib/uploadErrors";
import type { AgentDocument } from "@/types/api";
import { ConfirmDialog } from "./ConfirmDialog";
import { talkHref } from "./agentStatus";

const ACCEPT = ".pdf,.txt,.md,application/pdf,text/plain,text/markdown";
const OK_EXT = /\.(pdf|txt|md)$/i;

/** POST /api/knowledge/upload: `chunks_created` and `document.chunk_count` are both the number of sections. */
interface UploadResponse {
  chunks_created?: number;
  document?: { chunk_count?: number };
}

/** The file being sent now. At 100 the bytes are up and the server is reading and indexing: no percentage exists then. */
interface Batch {
  index: number;
  total: number;
  name: string;
  pct: number;
}

interface FailedFile {
  id: number;
  file: File;
  message: string;
  /** Sending the same file again can only help when the cause was not the file itself. */
  retryable: boolean;
}

let failedSeq = 0;

export function KnowledgePanel({ agentId }: { agentId: string }) {
  const docs = useAgentDocuments(agentId);
  const agents = useAgents();
  const questions = useExampleQuestions(agentId, docs.data?.length ?? 0);
  const inputRef = useRef<HTMLInputElement>(null);
  const hintId = useId();
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState<Batch | null>(null);
  const [failed, setFailed] = useState<FailedFile[]>([]);
  // How many sections the last upload added; null until a file has gone through.
  const [added, setAdded] = useState<number | null>(null);
  const [toDelete, setToDelete] = useState<AgentDocument | null>(null);

  const refresh = async () => {
    await Promise.all([docs.mutate(), agents.mutate()]);
  };

  const uploadFiles = async (files: FileList | File[]) => {
    const list = Array.from(files);
    if (!list.length) return;
    setAdded(null);
    const fail = (file: File, message: string, retryable: boolean) =>
      setFailed((prev) => [...prev.filter((f) => f.file.name !== file.name), { id: ++failedSeq, file, message, retryable }]);
    const valid = list.filter((f) => OK_EXT.test(f.name));
    for (const file of list) {
      if (OK_EXT.test(file.name)) continue;
      const message = "Only PDF, TXT and MD files are supported.";
      fail(file, message, false);
      toast.error(`${file.name}: ${message}`);
    }
    let sections = 0;
    let uploaded = 0;
    for (const [i, file] of valid.entries()) {
      setUploading({ index: i + 1, total: valid.length, name: file.name, pct: 0 });
      try {
        const fd = new FormData();
        fd.append("agent_id", agentId);
        fd.append("file", file);
        const res = await api.upload<UploadResponse>(`/api/knowledge/upload`, fd, {
          onProgress: (pct) => setUploading({ index: i + 1, total: valid.length, name: file.name, pct }),
        });
        sections += res?.chunks_created ?? res?.document?.chunk_count ?? 0;
        uploaded += 1;
        // A file that failed before and now went through is no longer a problem.
        setFailed((prev) => prev.filter((f) => f.file.name !== file.name));
        toast.success(`${file.name} added to your agent's documents.`);
        await refresh();
      } catch (e) {
        const message = friendlyUploadError(e, file.name, `Could not upload ${file.name}.`);
        fail(file, message, true);
        toast.error(`${file.name}: ${message}`);
      }
    }
    setUploading(null);
    if (uploaded > 0) setAdded(sections);
  };

  const retry = (f: FailedFile) => {
    setFailed((prev) => prev.filter((x) => x.id !== f.id));
    void uploadFiles([f.file]);
  };

  const removeDoc = async () => {
    if (!toDelete) return;
    try {
      await api.del(`/api/agents/${agentId}/documents/${toDelete.id}`);
      toast.success("Document removed.");
      setToDelete(null);
      await refresh();
    } catch (e) {
      toast.error(errMessage(e, "Could not remove the document."));
    }
  };

  const sections = (docs.data ?? []).reduce((n, d) => n + d.chunk_count, 0);

  return (
    <Card className="space-y-5">
      <div className="border-b border-border pb-4">
        <h2 className="font-display text-lg font-semibold text-text-primary">
          Documents{docs.data?.length ? <span className="ml-2 text-sm font-normal text-text-secondary">{sections} section{sections === 1 ? "" : "s"}</span> : null}
        </h2>
        <p className="mt-1 text-sm text-text-secondary">
          Add the files your agent should answer from, such as timings, fees or rules (PDF, TXT or Markdown). It says so when it cannot find an answer.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!uploading) void uploadFiles(e.dataTransfer.files);
        }}
        className={`flex flex-col items-center gap-2 rounded-xl border border-dashed px-4 py-8 text-center transition-colors ${
          dragging ? "border-accent-primary bg-accent-primary/5" : "border-border"
        }`}
      >
        {uploading ? (
          <div className="w-full max-w-xs rounded-xl border border-border bg-bg-surface p-3.5 text-left shadow-xs">
            {/* Only the file name (and "File 2 of 3") is announced; the percentage below is not a live region. */}
            <div className="flex items-center justify-between gap-2">
              <p role="status" className="flex min-w-0 items-center gap-2 text-sm font-medium text-text-primary">
                <Loader2 className="size-4 shrink-0 animate-spin motion-reduce:animate-none text-accent-primary" aria-hidden />
                <span className="truncate">
                  {uploading.total > 1 ? `File ${uploading.index} of ${uploading.total}: ` : "Uploading "}
                  {uploading.name}
                </span>
              </p>
              <span className="inline-flex shrink-0 items-center rounded-full border border-border bg-bg-elevated px-2 py-0.5 font-mono text-xs font-semibold tabular-nums text-text-primary">
                {uploading.pct}%
              </span>
            </div>
            <div
              role="progressbar"
              aria-label={`Uploading ${uploading.name}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={uploading.pct < 100 ? uploading.pct : undefined}
              aria-valuetext={uploading.pct < 100 ? undefined : "Processing and indexing"}
              className="mt-2.5 h-2.5 overflow-hidden rounded-full border border-border bg-bg-elevated p-[1px]"
            >
              <div
                className={`h-full rounded-full bg-accent-primary transition-[width] ${
                  uploading.pct >= 100 ? "animate-pulse motion-reduce:animate-none" : ""
                }`}
                style={{ width: `${uploading.pct}%` }}
              />
            </div>
            <p className="mt-1.5 text-xs tabular-nums text-text-secondary">
              {uploading.pct >= 100 ? "Processing and indexing" : `${uploading.pct}% uploaded`}
            </p>
          </div>
        ) : (
          <>
            <UploadCloud className="size-7 text-text-secondary" aria-hidden />
            <p className="text-sm text-text-primary">Drag files here, or</p>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="rounded-lg border border-border bg-bg-surface min-h-10 px-4 py-1.5 text-sm text-text-primary hover:border-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
            >
              Choose files
            </button>
            <p className="text-xs text-text-secondary">PDF, TXT or MD</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="sr-only"
          aria-label="Upload documents"
          aria-describedby={hintId}
          onChange={(e) => {
            if (e.target.files) void uploadFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      <p id={hintId} data-testid="pdf-text-hint" className="-mt-2 text-xs text-text-secondary">
        {PDF_TEXT_HINT}
      </p>

      {failed.length > 0 && (
        <ul aria-label="Files that did not upload" className="space-y-2">
          {failed.map((f) => (
            <li key={f.id} className="flex flex-wrap items-start gap-3 rounded-xl border border-accent-error/30 bg-accent-error/10 px-4 py-3">
              <AlertCircle className="mt-0.5 size-4 shrink-0 text-accent-error" aria-hidden />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-text-primary">{f.file.name}</p>
                <p className="mt-0.5 text-sm text-text-secondary">{f.message}</p>
              </div>
              {f.retryable && (
                <button
                  type="button"
                  onClick={() => retry(f)}
                  disabled={!!uploading}
                  aria-label={`Retry ${f.file.name}`}
                  className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-border bg-bg-surface px-3 text-sm text-text-primary hover:border-text-primary disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent-primary"
                >
                  <RotateCcw className="size-3.5" aria-hidden /> Retry
                </button>
              )}
              <button
                type="button"
                onClick={() => setFailed((prev) => prev.filter((x) => x.id !== f.id))}
                aria-label={`Dismiss ${f.file.name}`}
                className="grid size-10 place-items-center rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
              >
                <X className="size-4" aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      )}

      {added !== null && !uploading && (
        <div
          role="status"
          data-testid="upload-ready"
          className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-accent-success/30 bg-accent-success/10 px-4 py-3 text-sm text-text-primary"
        >
          <p className="flex items-center gap-2">
            <CheckCircle2 className="size-4 shrink-0 text-accent-success" aria-hidden />
            {added > 0 ? `Ready: ${added} section${added === 1 ? "" : "s"}. ` : "Ready. "}Try asking a question.
          </p>
          <Link
            href={talkHref(agentId)}
            className="inline-flex min-h-10 items-center rounded-lg border border-border bg-bg-surface px-3 text-sm font-medium text-text-primary hover:border-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
          >
            Ask it in the Playground
          </Link>
        </div>
      )}

      {docs.isLoading ? (
        <Skeleton className="h-12 w-full rounded-lg" />
      ) : docs.error ? (
        <p role="alert" className="text-sm text-accent-error">
          {docs.error.message || "Could not load documents."}{" "}
          <button type="button" className="underline underline-offset-4" onClick={() => docs.mutate()}>
            Retry
          </button>
        </p>
      ) : !docs.data?.length ? (
        <p className="text-center text-sm text-text-secondary">No documents yet. Until you add one, this agent can only use its instructions.</p>
      ) : (
        <ul className="divide-y divide-border rounded-xl border border-border">
          {docs.data.map((d) => (
            <li key={d.id} className="flex items-center gap-3 px-4 py-3">
              <FileText className="size-4 shrink-0 text-text-secondary" aria-hidden />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-text-primary">{d.filename}</p>
                <p className="text-xs text-text-secondary">{relativeTime(d.created_at)}</p>
              </div>
              <Badge tone="neutral">
                {d.chunk_count} section{d.chunk_count === 1 ? "" : "s"}
              </Badge>
              <button
                type="button"
                onClick={() => setToDelete(d)}
                aria-label={`Delete ${d.filename}`}
                className="grid size-11 place-items-center rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-accent-error focus-visible:outline-2 focus-visible:outline-accent-primary"
              >
                <Trash2 className="size-4" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {questions.data?.length ? (
        <div className="rounded-xl border border-border bg-bg-elevated/40 px-4 py-3">
          <h3 className="text-sm font-medium text-text-primary">Questions you can ask</h3>
          <p className="mt-0.5 text-xs text-text-secondary">Written from your documents. Try them in the Playground.</p>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-text-primary">
            {questions.data.map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <ConfirmDialog
        open={!!toDelete}
        title={toDelete ? `Delete ${toDelete.filename}?` : "Delete document"}
        description="The agent will stop answering from this document. This cannot be undone."
        confirmLabel="Delete document"
        keepLabel="Keep document"
        onCancel={() => setToDelete(null)}
        onConfirm={removeDoc}
      />
    </Card>
  );
}
