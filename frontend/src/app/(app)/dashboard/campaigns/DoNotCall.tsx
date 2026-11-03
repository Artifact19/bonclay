"use client";

import { useState } from "react";
import useSWR from "swr";
import { X } from "lucide-react";
import { toast } from "sonner";
import { Button, Card } from "@/components/ui";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { maskNumber } from "@/app/(app)/dashboard/calls/format";

/** Numbers that are never dialled, in any campaign. Shown masked; removing one needs the full number again. */
export function DoNotCall({ onChanged }: { onChanged?: () => void }) {
  const list = useSWR<{ numbers: string[] }>("/api/do-not-call", (p: string) => api.get<{ numbers: string[] }>(p), { shouldRetryOnError: false });
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const add = async () => {
    const numbers = text.split(/[\s,;]+/).filter(Boolean);
    if (!numbers.length) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.post<{ added: number; total: number }>("/api/do-not-call", { numbers });
      toast.success(r.added ? `${r.added} added to the do-not-call list` : "Those numbers were already on the list");
      setText("");
      await list.mutate();
      onChanged?.();
    } catch (e) {
      setError(errMessage(e, "Could not add those numbers."));
    } finally {
      setBusy(false);
    }
  };
  const remove = async (n: string) => {
    try {
      await api.del(`/api/do-not-call/${encodeURIComponent(n)}`);
      await list.mutate();
      onChanged?.();
    } catch (e) {
      setError(errMessage(e, "Could not remove that number."));
    }
  };

  const numbers = list.data?.numbers ?? [];
  return (
    <Card className="space-y-3 p-5">
      <div>
        <h2 className="text-base font-semibold text-text-primary">Do-not-call list</h2>
        <p className="mt-0.5 text-sm text-text-secondary">Anyone here is skipped in every campaign, even if they are on a list you upload later.</p>
      </div>
      <div className="flex flex-wrap gap-2">
        <input
          value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()}
          aria-label="Numbers to add" placeholder="+91 98765 43210, +91 98765 43211"
          className="h-10 min-w-0 flex-1 rounded-lg border border-border bg-bg-surface px-3 text-sm text-text-primary"
        />
        <Button onClick={add} loading={busy} disabled={!text.trim()}>Add</Button>
      </div>
      {error && <p role="alert" className="text-sm text-accent-error">{error}</p>}
      {numbers.length === 0 ? (
        <p className="text-sm text-text-secondary">No numbers yet.</p>
      ) : (
        <ul className="flex flex-wrap gap-2" aria-label="Numbers on the do-not-call list">
          {numbers.map((n) => (
            <li key={n} className="inline-flex items-center gap-1 rounded-full border border-border bg-bg-surface py-1 pl-3 pr-1 text-sm text-text-primary">
              {maskNumber(n)}
              <button type="button" onClick={() => remove(n)} aria-label={`Remove ${maskNumber(n)} from the list`} className="rounded-full p-1 text-text-secondary hover:bg-bg-elevated hover:text-text-primary">
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
