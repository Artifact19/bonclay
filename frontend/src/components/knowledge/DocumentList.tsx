"use client";

import { useState } from "react";
import { FileText, Trash2 } from "lucide-react";
import type { RagDocument } from "@/types/api";
import { Badge, Button, Card, EmptyState, ErrorState, Skeleton } from "@/components/ui";

interface Props {
  docs: RagDocument[] | undefined;
  loading: boolean;
  error: Error | undefined;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
  onRetry: () => void;
}

export default function DocumentList({ docs, loading, error, selectedId, onSelect, onDelete, onRetry }: Props) {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  if (loading && !docs) {
    return (
      <ul className="space-y-2" aria-busy="true" aria-label="Loading documents">
        {[0, 1, 2].map((i) => (
          <li key={i}>
            <Skeleton className="h-14 rounded-xl" />
          </li>
        ))}
      </ul>
    );
  }

  if (error && !docs) {
    return <ErrorState title="Could not load your documents" message={error.message} onRetry={onRetry} className="py-8" />;
  }

  if (!docs || docs.length === 0) {
    return (
      <EmptyState
        icon={<FileText aria-hidden />}
        title="No documents yet"
        description="Upload a PDF or TXT above and it will appear here, ready for questions."
        className="py-8"
      />
    );
  }

  return (
    <ul className="space-y-1.5" role="radiogroup" aria-label="Documents">
      {docs.map((d) => {
        const selected = d.id === selectedId;
        const confirming = confirmId === d.id;
        const deleting = deletingId === d.id;
        return (
          <li key={d.id}>
            <Card
              variant="solid"
              padded={false}
              className={`flex items-center gap-1 pr-1 transition-colors ${
                selected ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]" : ""
              }`}
            >
              <button
                type="button"
                role="radio"
                aria-checked={selected}
                onClick={() => onSelect(d.id)}
                className="flex min-h-14 min-w-0 flex-1 items-center gap-3 rounded-2xl px-3 text-left outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]"
              >
                <FileText className="size-4 shrink-0 text-[var(--color-text-secondary)]" aria-hidden />
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium text-[var(--color-text-primary)]">
                    {d.filename}
                  </span>
                  <span className="mt-0.5 flex items-center gap-2 text-xs tabular-nums text-[var(--color-text-secondary)]">
                    {d.chunks} chunks
                    <Badge tone={d.embedder === "sentence-transformers" ? "primary" : "neutral"}>
                      {d.embedder === "sentence-transformers" ? "Semantic" : "Keyword"}
                    </Badge>
                  </span>
                </span>
              </button>

              {confirming ? (
                <span className="flex shrink-0 items-center gap-1">
                  <Button
                    size="sm"
                    variant="outline"
                    loading={deleting}
                    onClick={async () => {
                      setDeletingId(d.id);
                      try {
                        await onDelete(d.id);
                      } finally {
                        setDeletingId(null);
                        setConfirmId(null);
                      }
                    }}
                    className="text-accent-error"
                  >
                    Delete
                  </Button>
                  <Button size="sm" variant="ghost" disabled={deleting} onClick={() => setConfirmId(null)}>
                    Keep
                  </Button>
                </span>
              ) : (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setConfirmId(d.id)}
                  aria-label={`Delete ${d.filename}`}
                  className="size-10 shrink-0 px-0"
                >
                  <Trash2 className="size-4" aria-hidden />
                </Button>
              )}
            </Card>
          </li>
        );
      })}
    </ul>
  );
}
