"use client";

import Link from "next/link";
import { useCallback, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { useRagDocuments } from "@/hooks/useApi";
import { PageHeader } from "@/components/ui";
import UploadZone, { type UploadResult } from "@/components/knowledge/UploadZone";
import DocumentList from "@/components/knowledge/DocumentList";
import AskPanel from "@/components/knowledge/AskPanel";

export default function TryDocumentPage() {
  const { data: docs, error, isLoading, mutate } = useRagDocuments();
  const [pickedId, setSelectedId] = useState<string | null>(null);

  // Default to the newest document; fall back if the picked one vanished.
  const selectedId = docs?.some((d) => d.id === pickedId) ? pickedId : (docs?.[0]?.id ?? null);

  const onUploaded = useCallback(
    async (r: UploadResult) => {
      await mutate();
      setSelectedId(r.doc_id);
    },
    [mutate]
  );

  const onDelete = useCallback(
    async (id: string) => {
      try {
        await api.del(`/api/rag/documents/${id}`);
        await mutate((cur) => cur?.filter((d) => d.id !== id), { revalidate: true });
        toast.success("Document removed");
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Could not delete the document");
      }
    },
    [mutate]
  );

  const selected = docs?.find((d) => d.id === selectedId);

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-8 sm:py-10">
      <Link
        href="/dashboard/knowledge"
        className="mb-3 inline-flex min-h-10 items-center gap-1.5 rounded text-sm text-text-secondary hover:text-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
      >
        <ArrowLeft className="size-4" aria-hidden /> Back to Documents
      </Link>
      <PageHeader
        title="Try a document"
        description="A quick test: upload any PDF, ask it a question and hear the answer. This is not saved to an agent."
      />

      <div className="grid gap-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-14">
        <section aria-labelledby="step-docs" className="min-w-0 space-y-6">
          <div>
            <h2 id="step-docs" className="mb-3 text-base font-semibold text-[var(--color-text-primary)]">
              1. Add a document
            </h2>
            <UploadZone onUploaded={onUploaded} />
          </div>
          <div>
            <h2 className="mb-3 text-base font-semibold text-[var(--color-text-primary)]">
              2. Choose what to ask about
              {docs && docs.length > 0 && (
                <span className="ml-2 text-sm font-normal tabular-nums text-[var(--color-text-secondary)]">
                  {docs.length}
                </span>
              )}
            </h2>
            <DocumentList
              docs={docs}
              loading={isLoading}
              error={error}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onDelete={onDelete}
              onRetry={() => void mutate()}
            />
          </div>
        </section>

        <section aria-labelledby="step-ask" className="min-w-0">
          <h2 id="step-ask" className="mb-3 text-base font-semibold text-[var(--color-text-primary)]">
            3. Ask and listen
          </h2>
          <AskPanel docId={selectedId} docName={selected?.filename} />
        </section>
      </div>
    </div>
  );
}
