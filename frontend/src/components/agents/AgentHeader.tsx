"use client";

import Link from "next/link";
import {
  ArrowLeft,
  Check,
  Circle,
  MessageCircle,
  Mic,
  Phone,
  PhoneCall,
  Trash2,
} from "lucide-react";
import { Badge, Button, Card } from "@/components/ui";
import { cn } from "@/lib/utils";
import type { Agent } from "@/types/api";
import { OverflowMenu } from "./OverflowMenu";
import {
  agentIsReady,
  setupSteps,
  talkHref,
  type SetupTab,
  useAgentTalkedTo,
  usePhoneConnected,
} from "./agentStatus";
import { languageNames } from "./languages";

interface Props {
  agent: Agent;
  /** The name as typed in the form, so the header follows the edit. */
  name?: string;
  voiceLabel: string;
  onBack: () => void;
  onDelete: (agent: Agent) => void;
  onPractice?: (agent: Agent) => void;
  onPhone?: (agent: Agent) => void;
  onGoToTab?: (tab: SetupTab) => void;
}

const actionClass =
  "inline-flex h-10 items-center justify-center gap-2 rounded-full px-4 text-sm font-medium select-none transition-[filter,background-color,border-color] duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary";

/** Sticky top of the agent page: name, what it can do now, setup progress and the main actions. */
export function AgentHeader({
  agent,
  name,
  voiceLabel,
  onBack,
  onDelete,
  onPractice,
  onPhone,
}: Props) {
  const phone = usePhoneConnected();
  const title = (name ?? agent.name).trim();
  const ready = agentIsReady(agent);

  return (
    <div className="space-y-3">
      <div className="z-20 -mx-4 border-b border-border bg-bg-base/95 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6 md:sticky md:top-0">
        <button
          type="button"
          onClick={onBack}
          className="mb-1.5 inline-flex items-center gap-1.5 rounded text-sm text-text-secondary hover:text-text-primary focus-visible:outline-2 focus-visible:outline-accent-primary"
        >
          <ArrowLeft className="size-4" aria-hidden /> All agents
        </button>
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2.5">
          <h1
            className="min-w-0 truncate font-display text-2xl font-medium tracking-[-0.02em] text-text-primary"
            data-testid="agent-title"
          >
            {title || "Unnamed agent"}
          </h1>
          <div className="flex flex-wrap items-center gap-2">
            <Link
              href={talkHref(agent.id)}
              className={cn(
                actionClass,
                "bg-brand text-[var(--color-on-accent)] hover:opacity-90",
              )}
            >
              <Mic className="size-4" aria-hidden /> Talk
            </Link>
            {onPractice && (
              <Button
                variant="outline"
                onClick={() => onPractice(agent)}
                leftIcon={<PhoneCall className="size-4" />}
              >
                Practice call
              </Button>
            )}
            {onPhone && (
              <Button
                variant="outline"
                onClick={() => onPhone(agent)}
                leftIcon={<Phone className="size-4" />}
              >
                Phone someone
              </Button>
            )}
            <OverflowMenu
              label="More actions"
              items={[
                {
                  label: "Delete agent",
                  icon: <Trash2 className="size-4" aria-hidden />,
                  danger: true,
                  onSelect: () => onDelete(agent),
                },
              ]}
            />
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={ready ? "success" : "warning"} dot>
          {ready ? "Ready" : "Needs documents"}
        </Badge>
        {phone === true && (
          <Badge tone="success">
            <Phone className="size-3" aria-hidden /> Phone connected
          </Badge>
        )}
        <Badge tone="neutral">
          <MessageCircle className="size-3" aria-hidden />{" "}
          {languageNames(agent.supported_languages)}
        </Badge>
        <Badge tone="neutral">
          <Mic className="size-3" aria-hidden /> {voiceLabel}
        </Badge>
      </div>
    </div>
  );
}

/** Setup progress and the steps still to do: sits in the rail beside the tabs. */
export function SetupChecklist({
  agent,
  onGoToTab,
}: {
  agent: Agent;
  onGoToTab?: (tab: SetupTab) => void;
}) {
  const phone = usePhoneConnected();
  const talked = useAgentTalkedTo(agent.id);
  const steps = setupSteps(agent, { talkedTo: talked, phoneConnected: phone });
  const doneCount = steps.filter((s) => s.done).length;

  return (
    <Card className="!p-0">
      <div className="space-y-2 border-b border-border px-4 py-3">
        <div className="flex items-baseline justify-between text-xs font-medium text-text-primary">
          <span>Setup</span>
          <span className="text-text-secondary">
            {doneCount} of {steps.length}
          </span>
        </div>
        <span
          aria-hidden
          className="block h-1.5 overflow-hidden rounded-full bg-bg-elevated"
        >
          <span
            className="block h-full rounded-full bg-accent-primary"
            style={{ width: `${(doneCount / steps.length) * 100}%` }}
          />
        </span>
      </div>
      <ul
        id="setup-steps"
        aria-label="Setup steps"
        className="divide-y divide-border"
      >
        {steps.map((s) => (
          <li
            key={s.id}
            className="flex items-center gap-3 px-4 py-2.5 text-sm"
          >
            {s.done ? (
              <Check
                className="size-4 shrink-0 text-accent-success"
                aria-hidden
              />
            ) : (
              <Circle className="size-4 shrink-0 text-text-muted" aria-hidden />
            )}
            <span
              className={cn(
                "flex-1",
                s.done ? "text-text-secondary" : "text-text-primary",
              )}
            >
              {s.label}
              <span className="sr-only">
                {s.done ? " (done)" : " (not done yet)"}
              </span>
            </span>
            {!s.done && onGoToTab && (
              <button
                type="button"
                onClick={() => onGoToTab(s.tab)}
                className="min-h-10 rounded px-2 text-xs font-medium text-accent-primary underline-offset-4 hover:underline focus-visible:outline-2 focus-visible:outline-accent-primary"
              >
                Go there
              </button>
            )}
          </li>
        ))}
      </ul>
    </Card>
  );
}
