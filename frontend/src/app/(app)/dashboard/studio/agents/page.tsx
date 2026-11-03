"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Bot, ChevronDown, Plus, Sparkles } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { useAgents, useVoices } from "@/hooks/useApi";
import { PageShell } from "@/components/ui/PageHeader";
import { toView } from "@/components/voices/voiceApi";
import {
  Button,
  Card,
  EmptyState,
  ErrorState,
  PageHeader,
  Skeleton,
} from "@/components/ui";
import {
  AgentEditor,
  parseTabParam,
  fieldClass as input,
  labelClass as label,
  type AgentTab,
} from "@/components/agents/AgentEditor";
import { AgentCard } from "@/components/agents/AgentCard";
import { usePhoneConnected } from "@/components/agents/agentStatus";
import { ConfirmDialog } from "@/components/agents/ConfirmDialog";
import { TemplateGallery } from "@/components/agents/TemplateGallery";
import { TestCallDialog } from "@/components/agents/TestCallDialog";
import { CallDialog } from "@/components/agents/CallDialog";
import { errMessage } from "@/components/dashboard/format";
import { LLM_FIX_TEXT } from "@/lib/llm";
import type { Agent, AgentTemplate } from "@/types/api";

/** Keep `?agent=<id>&tab=<tab>` in the address bar without adding history entries (Back still leaves the page). */
function writeUrl(agentId: string | null, tab: AgentTab = "basics") {
  if (typeof window === "undefined") return;
  const qs = agentId ? `?agent=${encodeURIComponent(agentId)}&tab=${tab}` : "";
  window.history.replaceState(
    window.history.state,
    "",
    `${window.location.pathname}${qs}`,
  );
}

export default function AgentStudioPage() {
  // useSearchParams needs a Suspense boundary for static rendering
  return (
    <Suspense fallback={null}>
      <AgentStudio />
    </Suspense>
  );
}

function AgentStudio() {
  const agents = useAgents();
  const reduceMotion = useReducedMotion();
  const params = useSearchParams();
  const deepLinkAgent = params?.get("agent") ?? null;
  const deepLinkTab = params?.get("tab") ?? null;
  const voiceParam = params?.get("voice") ?? null;
  const voices = useVoices();
  const phoneConnected = usePhoneConnected();
  const voiceViews = useMemo(
    () => (voices.data ?? []).map(toView),
    [voices.data],
  );
  const voiceName = (id: string | null) =>
    id
      ? (voiceViews.find((v) => v.id === id)?.name ?? "Unknown voice")
      : "Default voice";

  // A voice sent from the Voices page ("Use for an agent"): waits here until an agent is open.
  const [pendingVoice, setPendingVoice] = useState<string | null>(null);
  useEffect(() => {
    if (voiceParam) setPendingVoice(voiceParam);
  }, [voiceParam]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [tab, setTab] = useState<AgentTab>("basics");
  const [focusKnowledge, setFocusKnowledge] = useState(false);
  const handledLink = useRef<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [roleDesc, setRoleDesc] = useState("");
  const [agentName, setAgentName] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<Agent | null>(null);
  const [testing, setTesting] = useState<Agent | null>(null);
  const [testOpen, setTestOpen] = useState(false);
  const openTest = (a: Agent) => {
    setTesting(a);
    setTestOpen(true);
  };
  const [calling, setCalling] = useState<Agent | null>(null);
  const [callOpen, setCallOpen] = useState(false);
  const openCall = (a: Agent) => {
    setCalling(a);
    setCallOpen(true);
  };
  const [tryAsking, setTryAsking] = useState<{
    agentId: string;
    questions: string[];
  } | null>(null);

  /** Open one agent's page (optionally on a tab) and put it in the address bar so the link can be shared or reloaded. */
  const openAgent = (
    id: string,
    nextTab: AgentTab = "basics",
    focus = false,
  ) => {
    handledLink.current = id;
    setCreating(false);
    setEditingId(id);
    setTab(nextTab);
    setFocusKnowledge(focus);
    writeUrl(id, nextTab);
  };
  const closeAgent = () => {
    setPendingVoice(null);
    setEditingId(null);
    setFocusKnowledge(false);
    writeUrl(null);
  };

  const handleTemplateCreated = (agent: Agent, template: AgentTemplate) => {
    setTryAsking({
      agentId: agent.id,
      questions: template.suggested_questions,
    });
    openAgent(agent.id);
  };

  // /dashboard/studio/agents?agent=<id>&tab=basics|phone|channels|advanced opens that agent on that tab. The old
  // `tab=knowledge`, and a link with no tab (Analytics > "Add to knowledge"), open Basics at its Documents.
  // Handled once per link, so Back to the list works.
  useEffect(() => {
    if (!deepLinkAgent) {
      handledLink.current = null;
      return;
    }
    if (!agents.data || handledLink.current === deepLinkAgent) return;
    handledLink.current = deepLinkAgent;
    if (agents.data.some((a) => a.id === deepLinkAgent)) {
      const parsed = parseTabParam(deepLinkTab);
      setCreating(false);
      setEditingId(deepLinkAgent);
      setTab(parsed.tab);
      setFocusKnowledge(parsed.focusKnowledge);
    } else {
      toast.error("That agent no longer exists.");
    }
  }, [deepLinkAgent, deepLinkTab, agents.data]);

  const pendingVoiceName = pendingVoice
    ? voiceViews.find((v) => v.id === pendingVoice)?.name
    : undefined;
  const editing = agents.data?.find((a) => a.id === editingId) ?? null;
  const hasAgents = !!agents.data?.length;

  const handleGenerate = async () => {
    if (!roleDesc.trim() || !agentName.trim()) return;
    setIsGenerating(true);
    setLlmError(null);
    try {
      const created = await api.post<Agent>("/api/agents", {
        name: agentName.trim(),
        role_description: roleDesc.trim(),
      });
      await agents.mutate();
      setAgentName("");
      setRoleDesc("");
      openAgent(created.id);
      toast.success("Agent created. Review it and add your documents.");
    } catch (e) {
      if (e instanceof ApiError && e.code === "LLM_UNAVAILABLE")
        setLlmError(e.message);
      else toast.error(errMessage(e, "Could not create the agent. Try again."));
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDelete = async () => {
    if (!toDelete) return;
    try {
      await api.del(`/api/agents/${toDelete.id}`);
      if (editingId === toDelete.id) closeAgent();
      setToDelete(null);
      await agents.mutate();
      toast.success("Agent deleted.");
    } catch (e) {
      toast.error(errMessage(e, "Could not delete the agent."));
    }
  };

  const createBlocker = !agentName.trim()
    ? "Add a name first"
    : !roleDesc.trim()
      ? "Describe what the agent should do first"
      : "";

  const createForm = (
    <Card className="space-y-5">
      <div className="border-b border-border pb-4">
        <h2 className="flex items-center gap-2 font-display text-lg font-semibold text-text-primary">
          <Sparkles className="size-4 text-accent-primary" aria-hidden />
          Describe your agent
        </h2>
        <p className="mt-1 text-sm text-text-secondary">
          Say in a sentence or two what the agent should do, and we will write
          its instructions for you. You can change them later.
        </p>
      </div>
      <div>
        <label htmlFor="agent-name" className={label}>
          Agent name
        </label>
        <input
          id="agent-name"
          className={input}
          value={agentName}
          onChange={(e) => setAgentName(e.target.value)}
          placeholder="e.g. Sunrise Clinic receptionist"
        />
      </div>
      <div>
        <label htmlFor="agent-role" className={label}>
          What should it do?
        </label>
        <textarea
          id="agent-role"
          rows={3}
          className={`${input} resize-none`}
          value={roleDesc}
          onChange={(e) => setRoleDesc(e.target.value)}
          placeholder="e.g. Answers calls for a family clinic, shares timings and fees, and books appointments politely in Hindi and English."
        />
      </div>
      {llmError && (
        <div
          role="alert"
          className="rounded-xl border border-accent-warning/30 bg-accent-warning/10 p-4 text-sm"
        >
          <p className="font-medium text-text-primary">Setup needed</p>
          <p className="mt-1 text-text-secondary">
            BonClay cannot write the agent yet because no language model is
            connected. Ask the person who installed BonClay to set one up.
          </p>
          <details className="group mt-2">
            <summary className="inline-flex min-h-10 cursor-pointer list-none items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary">
              <ChevronDown
                className="size-3.5 transition-transform group-open:rotate-180"
                aria-hidden
              />{" "}
              Details
            </summary>
            <p className="mt-1 text-xs text-text-secondary">{llmError}</p>
            <p className="mt-1 text-xs text-text-secondary">
              Fix:{" "}
              <code className="rounded bg-bg-elevated px-1.5 py-0.5 font-mono">
                {LLM_FIX_TEXT}
              </code>
            </p>
          </details>
        </div>
      )}
      <div className="flex flex-wrap items-center justify-end gap-3">
        {createBlocker && (
          <p className="text-xs text-text-secondary">{createBlocker}</p>
        )}
        {hasAgents && (
          <Button
            variant="ghost"
            onClick={() => setCreating(false)}
            disabled={isGenerating}
          >
            Cancel
          </Button>
        )}
        <Button
          onClick={handleGenerate}
          loading={isGenerating}
          disabled={!!createBlocker}
          disabledReason={createBlocker || undefined}
          leftIcon={<Bot className="size-4" />}
        >
          {isGenerating ? "Creating" : "Create agent"}
        </Button>
      </div>
    </Card>
  );

  let body: React.ReactNode;
  if (agents.isLoading) {
    body = (
      <Card className="space-y-4">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-24 w-full" />
      </Card>
    );
  } else if (agents.error) {
    body = (
      <ErrorState
        title="Could not load your agents"
        message={agents.error.message}
        onRetry={() => agents.mutate()}
      />
    );
  } else if (editing) {
    const hint =
      tryAsking?.agentId === editing.id && tryAsking.questions.length > 0
        ? tryAsking.questions
        : null;
    body = (
      <>
        {hint && (
          <Card
            variant="solid"
            className="rounded-xl border-accent-primary/30 bg-accent-primary/5"
          >
            <p className="text-sm font-medium text-text-primary">
              Try asking your agent:
            </p>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-text-secondary">
              {hint.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          </Card>
        )}
        <AgentEditor
          key={editing.id}
          agent={editing}
          voices={voiceViews}
          focusKnowledge={focusKnowledge}
          tab={tab}
          onTabChange={(t) => {
            setTab(t);
            setFocusKnowledge(false);
            writeUrl(editing.id, t);
          }}
          presetVoiceId={
            pendingVoiceName ? (pendingVoice ?? undefined) : undefined
          }
          onPractice={openTest}
          onPhone={openCall}
          onBack={closeAgent}
          onDelete={setToDelete}
        />
      </>
    );
  } else if (!hasAgents) {
    body = (
      <>
        <EmptyState
          icon={<Bot />}
          title="No agents yet"
          description="Start from a ready-made template below, or describe your own agent in a sentence."
        />
        <TemplateGallery onCreated={handleTemplateCreated} />
        {createForm}
      </>
    );
  } else {
    body = (
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-6">
          {creating && createForm}
          <ul className="grid gap-4 sm:grid-cols-2">
            {agents.data!.map((a, i) => (
              <motion.li
                key={a.id}
                className="min-w-0"
                initial={reduceMotion ? false : { opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  delay: i * 0.04,
                  duration: 0.35,
                  ease: [0.22, 1, 0.36, 1],
                }}
              >
                <AgentCard
                  agent={a}
                  voiceName={voiceName(a.voice_id)}
                  phoneConnected={phoneConnected}
                  onEdit={(x) => openAgent(x.id)}
                  onPractice={openTest}
                  onPhone={openCall}
                  onDelete={setToDelete}
                />
              </motion.li>
            ))}
          </ul>
        </div>
        <aside aria-label="Start something new" className="lg:sticky lg:top-6">
          <Card variant="solid" className="space-y-4">
            <div>
              <h2 className="font-display text-lg font-semibold text-text-primary">
                Start something new
              </h2>
              <p className="mt-1 text-sm text-text-secondary">
                Begin from a ready-made template, or describe your own agent.
              </p>
            </div>
            {!creating && (
              <Button
                className="w-full"
                onClick={() => setCreating(true)}
                leftIcon={<Sparkles className="size-4" />}
              >
                Describe your own agent
              </Button>
            )}
            <TemplateGallery compact onCreated={handleTemplateCreated} />
          </Card>
        </aside>
      </div>
    );
  }

  return (
    <PageShell width="wide" className="space-y-6">
      {!editing && (
        <PageHeader
          title="Agents"
          description="Your voice agents. Talk to one to hear it, or set up a new one."
          actions={
            hasAgents && !creating ? (
              <Button
                onClick={() => setCreating(true)}
                leftIcon={<Plus className="size-4" />}
              >
                New agent
              </Button>
            ) : undefined
          }
        />
      )}
      {pendingVoiceName && !editing && (
        <div
          role="status"
          className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-accent-primary/30 bg-accent-primary/5 px-4 py-3 text-sm text-text-primary"
        >
          <span>Pick an agent to use {pendingVoiceName} with</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPendingVoice(null)}
          >
            Dismiss
          </Button>
        </div>
      )}
      {body}
      <TestCallDialog
        agent={testing}
        open={testOpen}
        onClose={() => setTestOpen(false)}
      />
      <CallDialog
        agent={calling}
        open={callOpen}
        onClose={() => setCallOpen(false)}
      />
      <ConfirmDialog
        open={!!toDelete}
        title={toDelete ? `Delete ${toDelete.name}?` : "Delete agent"}
        description="This removes the agent, its documents and its safety-check history. This cannot be undone."
        confirmLabel="Delete agent"
        keepLabel="Keep agent"
        onCancel={() => setToDelete(null)}
        onConfirm={handleDelete}
      />
    </PageShell>
  );
}
