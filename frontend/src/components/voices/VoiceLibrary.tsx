"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import useSWR, { useSWRConfig } from "swr";
import { useHealth, useModelStatus, useVoices } from "@/hooks/useApi";
import { toast } from "sonner";
import { AudioLines, Plus, Search } from "lucide-react";
import { Button, EmptyState, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { StorageToggle } from "@/components/StorageToggle";
import { useVoiceStore } from "@/stores";
import { api } from "@/lib/api";
import { VoiceCard } from "./VoiceCard";
import { CloneWizard } from "./CloneWizard";
import { GenerateModal } from "./GenerateModal";
import { DeleteDialog } from "./DeleteDialog";
import { ConsentRecordDialog } from "./ConsentRecordPanel";
import { VOICES_KEY, deleteVoice, toView, type VoiceView } from "./voiceApi";
import { languageLabel } from "./validate";
import { indianCoverage, previewLine } from "./coverage";

interface RetentionReport {
  retention_days: number;
  enforce: boolean;
  expired: { id: string; name: string | null; retain_until: string | null }[];
}

const selectClass =
  "h-11 rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-surface))] px-3 text-sm text-[var(--color-text-primary)] focus:border-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40";

export function VoiceLibrary() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();
  const wizardOpen = params.get("new") === "1";

  const { mutate } = useSWRConfig();
  const { data, error, isLoading, mutate: revalidate } = useVoices();
  const health = useHealth();
  const modelStatus = useModelStatus();
  // Indian languages a saved voice can speak (needs DhVaani on this server); null when it is not set up.
  const coverage = useMemo(() => indianCoverage(modelStatus.data?.engines), [modelStatus.data]);
  const cloud = health.data?.supabase?.configured === true;
  const setStoreVoices = useVoiceStore((s) => s.setVoices);

  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("all");
  const [generating, setGenerating] = useState<VoiceView | null>(null);
  const [deleting, setDeleting] = useState<VoiceView | null>(null);
  const [viewing, setViewing] = useState<VoiceView | null>(null);
  // How long voices are kept, and which are past that (report only: nothing is deleted from here).
  const retention = useSWR<RetentionReport>("/api/voices/retention", (p: string) => api.get<RetentionReport>(p), {
    revalidateOnFocus: false,
    shouldRetryOnError: false,
  });

  useEffect(() => {
    if (data) setStoreVoices(data);
  }, [data, setStoreVoices]);

  // Storage toggle swaps the data source.
  useEffect(() => {
    const onChange = () => {
      revalidate();
      retention.mutate();
    };
    window.addEventListener("storageModeChanged", onChange);
    return () => window.removeEventListener("storageModeChanged", onChange);
  }, [revalidate, retention]);

  const voices = useMemo(() => (data ?? []).map(toView), [data]);
  const languages = useMemo(() => Array.from(new Set(voices.map((v) => v.language))).sort(), [voices]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return voices.filter(
      (v) =>
        (!q || v.name.toLowerCase().includes(q)) &&
        (language === "all" || v.language === language)
    );
  }, [voices, query, language]);

  const setWizard = useCallback(
    (open: boolean) => {
      router.replace(open ? `${pathname}?new=1` : pathname, { scroll: false });
    },
    [router, pathname]
  );

  const confirmDelete = async (voice: VoiceView) => {
    const previous = data;
    const next = (data ?? []).filter((v) => v.id !== voice.id);
    setDeleting(null);
    // Optimistic: remove immediately, restore on failure.
    await mutate(VOICES_KEY, next, { revalidate: false });
    try {
      await deleteVoice(voice.id);
      toast.success(`Consent withdrawn. ${voice.name} was deleted`);
      retention.mutate();
      revalidate();
    } catch (e) {
      await mutate(VOICES_KEY, previous, { revalidate: false });
      toast.error(`Could not withdraw consent and delete ${voice.name}`, {
        description: e instanceof Error ? e.message : "Try again in a moment.",
      });
    }
  };

  const filtersActive = query.trim() !== "" || language !== "all";
  const newButton = (
    <Button onClick={() => setWizard(true)}>
      <Plus className="h-4 w-4" />
      Clone a voice
    </Button>
  );

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 sm:py-8 lg:px-10">
      <PageHeader
        title="My voices"
        description="Voices cloned from a real person who gave their consent. Each one can speak its own language and, where set up, many Indian languages."
        actions={
          <>
            <StorageToggle />
            {newButton}
          </>
        }
      />

      <p className="-mt-2 mb-6 text-sm text-[var(--color-text-secondary)]" data-testid="voice-retention-note">
        {cloud
          ? "Your voice samples and consent recordings stay in your BonClay storage"
          : "Your voice samples and consent recordings stay on this computer (local mode)"}
        {retention.data ? ` for ${retention.data.retention_days} days, or until you withdraw consent and delete the voice.` : " until you delete them."}
      </p>

      {retention.data && retention.data.expired.length > 0 && (
        <p
          role="status"
          className="-mt-3 mb-6 rounded-xl border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-[var(--color-text-primary)]"
          data-testid="retention-expired"
        >
          {retention.data.expired.length === 1 ? "1 voice is" : `${retention.data.expired.length} voices are`} past the{" "}
          {retention.data.retention_days}-day retention period:{" "}
          {retention.data.expired.map((e) => e.name ?? e.id).join(", ")}.{" "}
          {retention.data.enforce
            ? "They are deleted automatically."
            : "Nothing is deleted automatically. Review them and withdraw consent and delete the ones you no longer need."}
        </p>
      )}

      {voices.length > 0 && (
        <div className="mb-6 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--color-text-secondary)]" aria-hidden="true" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search voices by name"
              placeholder="Search voices"
              className="h-11 w-full rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-surface))] pl-10 pr-3 text-base text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)]/70 focus:border-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40"
            />
          </div>
          <select aria-label="Filter by language" value={language} onChange={(e) => setLanguage(e.target.value)} className={selectClass}>
            <option value="all">All languages</option>
            {languages.map((l) => (
              <option key={l} value={l}>
                {languageLabel(l)}
              </option>
            ))}
          </select>
        </div>
      )}

      {isLoading ? (
        <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3" aria-busy="true" aria-label="Loading voices">
          {[0, 1, 2].map((i) => (
            <div key={i} className="overflow-hidden rounded-2xl border border-[var(--color-border)]">
              <div className="space-y-3 p-5">
                <Skeleton className="h-5 w-2/3" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-11 w-full" />
              </div>
            </div>
          ))}
        </div>
      ) : error ? (
        <ErrorState
          title="Could not load your voices"
          message={error instanceof Error ? error.message : undefined}
          onRetry={() => revalidate()}
        />
      ) : voices.length === 0 ? (
        <EmptyState
          icon={<AudioLines />}
          title="No voices yet"
          description="Record or upload a short sample of someone speaking, with their consent, and BonClay will turn it into a voice you can use in agents and audio."
          action={newButton}
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Search />}
          title="No voices match"
          description={filtersActive ? "Try a different name or clear the filters." : undefined}
          action={
            <Button
              onClick={() => {
                setQuery("");
                setLanguage("all");
              }}
            >
              Clear filters
            </Button>
          }
        />
      ) : (
        <ul className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((v) => (
            <li key={v.id}>
              <VoiceCard voice={v} coverage={coverage} onGenerate={setGenerating} onDelete={setDeleting} onShowConsent={setViewing} />
            </li>
          ))}
        </ul>
      )}

      <CloneWizard
        open={wizardOpen}
        onClose={() => setWizard(false)}
        onCreated={() => {
          setWizard(false);
          revalidate();
        }}
      />
      <GenerateModal
        voice={generating}
        onClose={() => setGenerating(null)}
        preview
        initialText={generating ? previewLine(generating.language) : ""}
      />
      <ConsentRecordDialog
        voice={viewing}
        onClose={() => setViewing(null)}
        onWithdraw={(v) => {
          setViewing(null);
          setDeleting(v);
        }}
      />
      <DeleteDialog voice={deleting} onCancel={() => setDeleting(null)} onConfirm={confirmDelete} />
    </div>
  );
}

