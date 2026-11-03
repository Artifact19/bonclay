"use client";

import { useState } from "react";
import { Check, Copy, KeyRound, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card, EmptyState, ErrorBoundary, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { useApiKeys } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { errMessage, relativeTime } from "@/components/dashboard/format";

function KeysContent() {
  const { data, error, isLoading, mutate } = useApiKeys();
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [revealed, setRevealed] = useState<{ id?: string; name: string; key: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  const unavailable = error?.status === 404 || error?.status === 501;

  const create = async (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    const label = name.trim();
    if (!label) return;
    setCreating(true);
    try {
      const res = await api.post<{ id?: string; key?: string; api_key?: string; token?: string }>("/api/keys", { name: label });
      const key = res?.key ?? res?.api_key ?? res?.token;
      if (!key) throw new Error("The server did not return a key.");
      setRevealed({ id: res?.id, name: label, key });
      setCopied(false);
      setName("");
      await mutate();
      toast.success("API key created");
    } catch (err) {
      toast.error(errMessage(err, "Could not create the key."));
    } finally {
      setCreating(false);
    }
  };

  const copy = async () => {
    if (!revealed) return;
    try {
      await navigator.clipboard.writeText(revealed.key);
      setCopied(true);
      toast.success("Key copied");
    } catch {
      toast.error("Copy failed. Select the key and copy it manually.");
    }
  };

  const revoke = async (id: string) => {
    setRevoking(id);
    try {
      await api.del(`/api/keys/${encodeURIComponent(id)}`);
      const gone = data?.find((k) => k.id === id);
      await mutate((keys) => keys?.filter((k) => k.id !== id), { revalidate: true });
      // Don't keep showing the one-time secret of a key that no longer works.
      setRevealed((r) => (r && (r.id === id || (!!gone?.prefix && r.key.startsWith(gone.prefix))) ? null : r));
      toast.success("Key revoked");
    } catch (err) {
      toast.error(errMessage(err, "Could not revoke the key."));
    } finally {
      setRevoking(null);
      setConfirmId(null);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 sm:p-8 md:p-12">
      <PageHeader
        title="API keys"
        description="Personal keys let scripts and apps call the BonClay API as you. Treat them like passwords."
      />

      {unavailable ? (
        <EmptyState
          icon={<KeyRound />}
          title="API keys are not available on this server"
          description="The backend does not expose the keys endpoint yet. Update the backend, then reload this page."
        />
      ) : (
        <>
          <Card variant="solid">
            <form onSubmit={create} className="flex flex-col gap-3 sm:flex-row sm:items-end">
              <div className="flex-1">
                <label htmlFor="key-name" className="mb-1.5 block text-xs font-medium text-text-secondary">
                  Key name
                </label>
                <input
                  id="key-name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  maxLength={60}
                  placeholder="e.g. Podcast build script"
                  className="h-10 w-full rounded-lg border border-border bg-bg-base px-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent-primary focus:outline-none"
                />
              </div>
              <Button type="submit" loading={creating} disabled={!name.trim()} leftIcon={<Plus className="size-4" />}>
                Generate key
              </Button>
            </form>

            {revealed && (
              <div role="status" className="mt-4 rounded-xl border border-accent-warning/40 bg-accent-warning/10 p-4">
                <p className="text-sm font-medium text-text-primary">
                  Copy &ldquo;{revealed.name}&rdquo; now. You will not be able to see it again.
                </p>
                <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center">
                  <code className="min-w-0 flex-1 break-all rounded-lg bg-bg-base px-3 py-2 font-mono text-xs text-text-primary">
                    {revealed.key}
                  </code>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={copy}
                    leftIcon={copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                  >
                    {copied ? "Copied" : "Copy"}
                  </Button>
                </div>
                <button
                  type="button"
                  onClick={() => setRevealed(null)}
                  className="mt-3 text-xs text-text-secondary underline underline-offset-4 hover:text-text-primary"
                >
                  I have saved it, hide
                </button>
              </div>
            )}
          </Card>

          <section aria-labelledby="keys-heading">
            <h2 id="keys-heading" className="mb-3 font-display text-lg font-semibold text-text-primary">
              Your keys
            </h2>
            {error && !data ? (
              <ErrorState title="Could not load your keys" message={error.message} onRetry={() => mutate()} />
            ) : isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-16 rounded-xl" />
                <Skeleton className="h-16 rounded-xl" />
              </div>
            ) : !data || data.length === 0 ? (
              <EmptyState
                icon={<KeyRound />}
                title="No keys yet"
                description="Name a key above and generate it to start calling the API."
              />
            ) : (
              <Card variant="solid" padded={false}>
                <ul className="divide-y divide-border">
                  {data.map((k) => (
                    <li key={k.id} className="flex flex-wrap items-center gap-3 p-4">
                      <KeyRound className="size-4 shrink-0 text-text-muted" aria-hidden />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-text-primary">{k.name}</p>
                        <p className="truncate text-xs text-text-secondary">
                          {k.prefix && <span className="font-mono">{k.prefix}…  · </span>}
                          Created {relativeTime(k.created_at) || "recently"}
                          {" · "}
                          {k.last_used_at ? `Used ${relativeTime(k.last_used_at)}` : "Never used"}
                        </p>
                      </div>
                      {confirmId === k.id ? (
                        <div className="flex items-center gap-2">
                          <Badge tone="error">Revoke?</Badge>
                          <Button size="sm" variant="outline" loading={revoking === k.id} onClick={() => revoke(k.id)}>
                            Yes, revoke
                          </Button>
                          <Button size="sm" variant="ghost" onClick={() => setConfirmId(null)}>
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => setConfirmId(k.id)}
                          aria-label={`Revoke ${k.name}`}
                          leftIcon={<Trash2 className="size-3.5" />}
                        >
                          Revoke
                        </Button>
                      )}
                    </li>
                  ))}
                </ul>
              </Card>
            )}
          </section>
        </>
      )}
    </div>
  );
}

export default function KeysPage() {
  return (
    <ErrorBoundary>
      <KeysContent />
    </ErrorBoundary>
  );
}
