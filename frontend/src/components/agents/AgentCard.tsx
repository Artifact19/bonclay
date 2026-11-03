"use client";

import Link from "next/link";
import {
  FileText,
  Languages,
  Mic,
  Pencil,
  Phone,
  PhoneCall,
  Trash2,
} from "lucide-react";
import { Badge, Button, Card } from "@/components/ui";
import { relativeTime } from "@/components/dashboard/format";
import { cn } from "@/lib/utils";
import type { Agent } from "@/types/api";
import { OverflowMenu } from "./OverflowMenu";
import { agentIsReady, talkHref } from "./agentStatus";
import { languageNames } from "./languages";

const snippet = (s: string, n = 110) =>
  s.length > n ? `${s.slice(0, n).trimEnd()}...` : s;

interface Props {
  agent: Agent;
  voiceName: string;
  /** Is a phone line set up? `undefined` while unknown. It is account-wide, so every card shows the same answer. */
  phoneConnected?: boolean;
  onEdit: (agent: Agent) => void;
  onPractice: (agent: Agent) => void;
  onPhone: (agent: Agent) => void;
  onDelete: (agent: Agent) => void;
}

/** One agent in the list: what tells it apart from the others, "Talk" as the main action, the rest behind "…". */
export function AgentCard({
  agent: a,
  voiceName,
  phoneConnected,
  onEdit,
  onPractice,
  onPhone,
  onDelete,
}: Props) {
  const ready = agentIsReady(a);
  const created = relativeTime(a.created_at);
  return (
    <Card variant="solid" interactive className="flex h-full flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <span
            aria-hidden
            className="flex size-10 shrink-0 items-center justify-center rounded-full bg-accent-primary/15 font-display text-base font-semibold text-accent-primary"
          >
            {(a.name.trim().charAt(0) || "?").toUpperCase()}
          </span>
          <div className="min-w-0">
            <h2 className="truncate font-display text-base font-semibold text-text-primary">
              {a.name}
            </h2>
            <p className="mt-1 line-clamp-2 text-sm text-text-secondary">
              {a.greeting ? snippet(a.greeting) : "No greeting yet."}
            </p>
          </div>
        </div>
        <Badge tone={ready ? "success" : "warning"} dot className="shrink-0">
          {ready ? "Ready" : "Needs documents"}
        </Badge>
      </div>

      <dl className="grid gap-x-4 gap-y-1.5 text-xs text-text-secondary xl:grid-cols-2">
        <div className="flex items-center gap-2">
          <dt className="sr-only">Languages</dt>
          <Languages className="size-3.5 shrink-0" aria-hidden />
          <dd className="truncate">{languageNames(a.supported_languages)}</dd>
        </div>
        <div className="flex items-center gap-2">
          <dt className="sr-only">Voice</dt>
          <Mic className="size-3.5 shrink-0" aria-hidden />
          <dd className="truncate">{voiceName}</dd>
        </div>
        <div className="flex items-center gap-2 xl:col-span-2">
          <dt className="sr-only">Documents</dt>
          <FileText className="size-3.5 shrink-0" aria-hidden />
          <dd>
            {a.document_count} document{a.document_count === 1 ? "" : "s"}
            {created ? (
              <span className="text-text-muted"> · Created {created}</span>
            ) : null}
          </dd>
        </div>
        {phoneConnected && (
          <div className="flex items-center gap-2">
            <dt className="sr-only">Phone</dt>
            <Phone
              className="size-3.5 shrink-0 text-accent-success"
              aria-hidden
            />
            <dd className="text-accent-success">Phone connected</dd>
          </div>
        )}
      </dl>

      <div className="mt-auto flex items-center gap-2 pt-1">
        <Link
          href={talkHref(a.id)}
          aria-label={`Talk to ${a.name}`}
          className={cn(
            "inline-flex h-10 items-center justify-center gap-2 rounded-full bg-brand px-5 text-sm font-medium text-[var(--color-on-accent)]",
            "transition-[filter] hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-primary",
          )}
        >
          <Mic className="size-4" aria-hidden /> Talk
        </Link>
        <Button
          variant="outline"
          onClick={() => onEdit(a)}
          aria-label={`Edit ${a.name}`}
          leftIcon={<Pencil className="size-4" />}
        >
          Edit
        </Button>
        <OverflowMenu
          className="ml-auto"
          label={`More actions for ${a.name}`}
          items={[
            {
              label: "Practice call",
              icon: <PhoneCall className="size-4" aria-hidden />,
              onSelect: () => onPractice(a),
            },
            {
              label: "Phone someone",
              icon: <Phone className="size-4" aria-hidden />,
              onSelect: () => onPhone(a),
            },
            {
              label: "Delete agent",
              icon: <Trash2 className="size-4" aria-hidden />,
              danger: true,
              onSelect: () => onDelete(a),
            },
          ]}
        />
      </div>
    </Card>
  );
}
