"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { Megaphone, Plus } from "lucide-react";
import { Button, EmptyState, ErrorBoundary, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { useAgents } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import { CampaignCard } from "./CampaignCard";
import { DoNotCall } from "./DoNotCall";
import { NewCampaign } from "./NewCampaign";
import type { CampaignsResponse } from "./types";

function CampaignsContent() {
  const [creating, setCreating] = useState(false);
  const agents = useAgents();
  const { data, error, isLoading, mutate } = useSWR<CampaignsResponse>("/api/campaigns", (p: string) => api.get<CampaignsResponse>(p), {
    // progress without a reload while anything is calling
    refreshInterval: (d) => (d?.campaigns.some((c) => c.status === "running") ? 5000 : 30000),
    revalidateOnFocus: true,
    shouldRetryOnError: false,
  });
  const names = useMemo(() => new Map((agents.data ?? []).map((a) => [a.id, a.name])), [agents.data]);
  const campaigns = data?.campaigns ?? [];

  return (
    <PageHeader
      width="default"
      title="Campaigns"
      description="Call a list of people who agreed to hear from you. BonClay calls inside the hours you choose, tries again when nobody answers, and never calls anyone on your do-not-call list."
      actions={!creating ? <Button onClick={() => setCreating(true)}><Plus className="size-4" />New campaign</Button> : undefined}
    >
      <div className="space-y-6">
        <p className="rounded-lg border border-border bg-bg-surface p-3 text-sm text-text-secondary">
          Only add people who agreed to receive this call. BonClay does not check national do-not-disturb registers, so
          check your list against them yourself. Phone calls need Exotel to be set up (see the phone guide).
        </p>

        {creating && (
          <NewCampaign
            onCancel={() => setCreating(false)}
            onCreated={() => { setCreating(false); mutate(); }}
          />
        )}

        {error && <ErrorState title="Could not load the campaigns" message={errMessage(error, "Check that the API is running.")} onRetry={() => mutate()} />}
        {isLoading && (
          <div className="space-y-3" aria-busy="true" aria-label="Loading campaigns">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </div>
        )}
        {data && campaigns.length === 0 && !creating && (
          <EmptyState
            icon={<Megaphone />}
            title="No campaigns yet"
            description="Upload a list of numbers, choose the hours, and an agent calls them one by one. Try it with your own number first."
            action={
              <div className="flex flex-wrap justify-center gap-2">
                <Button onClick={() => setCreating(true)}>New campaign</Button>
                <Link href="/dashboard/studio/agents" className="inline-flex h-10 items-center justify-center rounded-full border border-border bg-bg-surface px-4 text-sm font-medium text-text-primary hover:border-text-primary">
                  Go to agents
                </Link>
              </div>
            }
          />
        )}
        {campaigns.length > 0 && (
          <ul className="space-y-4">
            {campaigns.map((c) => (
              <li key={c.id}><CampaignCard campaign={c} agentName={names.get(c.agent_id) ?? "Agent"} onChanged={() => mutate()} /></li>
            ))}
          </ul>
        )}

        <DoNotCall onChanged={() => mutate()} />
      </div>
    </PageHeader>
  );
}

export default function CampaignsPage() {
  return (
    <ErrorBoundary>
      <CampaignsContent />
    </ErrorBoundary>
  );
}
