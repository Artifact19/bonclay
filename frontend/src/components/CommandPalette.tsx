"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { ArrowRight, Bot, CornerDownLeft, Search } from "lucide-react";
import { useAgents } from "@/hooks/useApi";
import { areaForPath, rememberArea } from "@/lib/areas";
import { APP_FONT_CLASSES } from "@/lib/appFonts";
import { cn } from "@/lib/utils";

export interface PaletteAction {
  id: string;
  label: string;
  /** Small grey text on the right, e.g. "Page" or "Agent". */
  hint: string;
  href: string;
  /** Extra words that should find this action. */
  keywords?: string;
  agent?: boolean;
}

/** Every dashboard page, from both areas. Labels match the page titles and the plain-language names used in the sidebar. */
export const PAGE_ACTIONS: PaletteAction[] = [
  { id: "go-home", label: "Go to Agents home", hint: "Page", href: "/dashboard", keywords: "home overview dashboard start" },
  { id: "go-studio-home", label: "Go to Studio home", hint: "Page", href: "/dashboard/studio/home", keywords: "home start make audio create" },
  { id: "go-agents", label: "Go to Agents", hint: "Page", href: "/dashboard/studio/agents", keywords: "assistant receptionist create new" },
  { id: "go-calls", label: "Go to Calls and bookings", hint: "Page", href: "/dashboard/calls", keywords: "phone appointments history" },
  { id: "go-talk", label: "Go to Talk to your agent", hint: "Page", href: "/dashboard/playground", keywords: "playground practice try test conversation" },
  { id: "go-documents", label: "Go to Documents", hint: "Page", href: "/dashboard/knowledge", keywords: "knowledge upload pdf files agent answers from" },
  { id: "go-results", label: "Go to Results", hint: "Page", href: "/dashboard/analytics", keywords: "analytics reports numbers stats" },
  { id: "go-safety", label: "Go to Safety check", hint: "Page", href: "/dashboard/eval", keywords: "evaluations eval test tricky callers" },
  { id: "go-approvals", label: "Go to Approvals", hint: "Page", href: "/dashboard/approvals", keywords: "bookings meetings requests waiting" },
  { id: "go-voices", label: "Go to Voices", hint: "Page", href: "/dashboard/voices", keywords: "library clone" },
  { id: "go-studio", label: "Go to Voice Studio", hint: "Page", href: "/dashboard/studio", keywords: "speech text to speech generate audio" },
  { id: "go-say", label: "Go to Say it once", hint: "Page", href: "/dashboard/say-it-once", keywords: "translate speak languages" },
  { id: "go-keys", label: "Go to API keys", hint: "Page", href: "/dashboard/keys", keywords: "developer token" },
  { id: "go-verify", label: "Go to Verify a clip", hint: "Page", href: "/verify", keywords: "check watermark receipt" },
  { id: "clone", label: "Clone a voice", hint: "Action", href: "/dashboard/voices?new=1", keywords: "new voice record my voice" },
  { id: "approve", label: "Approve bookings", hint: "Action", href: "/dashboard/approvals", keywords: "approve decline appointments meetings requests" },
];

/** Case-insensitive match: a substring scores higher the earlier it is; otherwise letters in order ("agnt" finds "Agents"). */
function scoreOne(text: string, token: string): number {
  const at = text.indexOf(token);
  if (at >= 0) return 1000 - at;
  let i = 0;
  for (const ch of text) if (ch === token[i] && ++i === token.length) return 100;
  return -1;
}

/** Every word typed must match the label or its keywords. Returns -1 for no match. */
export function scoreAction(a: PaletteAction, query: string): number {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return 0;
  const label = a.label.toLowerCase();
  const all = `${label} ${a.keywords ?? ""}`.toLowerCase();
  let total = 0;
  for (const t of tokens) {
    const s = Math.max(scoreOne(label, t), scoreOne(all, t) - 200);
    if (s < 0) return -1;
    total += s;
  }
  return total;
}

/** "12 Oct": tells apart agents that share a name. */
function shortDate(iso?: string): string {
  const d = iso ? new Date(iso) : null;
  return d && !Number.isNaN(d.getTime()) ? d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }) : "";
}

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  // Mounted only while open, so the agents list is fetched lazily and the form state resets every time.
  if (!open) return null;
  return <PaletteDialog onClose={onClose} />;
}

function PaletteDialog({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const agents = useAgents();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const uid = useId();
  const listId = `${uid}-list`;

  // Remember what had focus so it goes back there on close.
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    inputRef.current?.focus();
    return () => {
      if (before && document.contains(before)) before.focus();
    };
  }, []);

  const actions = useMemo<PaletteAction[]>(() => {
    const agentHint = (iso?: string) => {
      const d = shortDate(iso);
      return d ? `Agent, ${d}` : "Agent";
    };
    const agentActions = (agents.data ?? []).flatMap((a): PaletteAction[] => [
      { id: `talk-${a.id}`, label: `Talk to ${a.name}`, hint: agentHint(a.created_at), href: `/dashboard/playground?agent=${encodeURIComponent(a.id)}`, keywords: "practice try conversation", agent: true },
      { id: `edit-${a.id}`, label: `Edit ${a.name}`, hint: agentHint(a.created_at), href: `/dashboard/studio/agents?agent=${encodeURIComponent(a.id)}`, keywords: "settings change", agent: true },
    ]);
    return [...PAGE_ACTIONS.slice(0, 2), ...agentActions, ...PAGE_ACTIONS.slice(2)];
  }, [agents.data]);

  const results = useMemo(() => {
    if (!query.trim()) return actions;
    return actions
      .map((a, i) => ({ a, s: scoreAction(a, query), i }))
      .filter((r) => r.s >= 0)
      .sort((x, y) => y.s - x.s || x.i - y.i)
      .map((r) => r.a);
  }, [actions, query]);

  const current = results.length ? Math.min(active, results.length - 1) : -1;

  useEffect(() => {
    if (current < 0) return;
    listRef.current?.children[current]?.scrollIntoView?.({ block: "nearest" });
  }, [current]);

  const run = (a: PaletteAction | undefined) => {
    if (!a) return;
    onClose();
    // Going to a page of one area also switches to it (the sidebar follows the route); remembering it first stops
    // /dashboard from sending a Studio person back to Studio home.
    const area = areaForPath(a.href.split("?")[0]);
    if (area) rememberArea(area);
    router.push(a.href);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (results.length) setActive((current + 1) % results.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (results.length) setActive((current - 1 + results.length) % results.length);
    } else if (e.key === "Home" && results.length) {
      e.preventDefault();
      setActive(0);
    } else if (e.key === "End" && results.length) {
      e.preventDefault();
      setActive(results.length - 1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      run(results[current]);
    } else if (e.key === "Tab") {
      // Focus trap: the search box is the only stop in the dialog.
      e.preventDefault();
    }
  };

  const loading = agents.isLoading && !agents.data;

  return createPortal(
    <div className={`fixed inset-0 z-[70] flex items-start justify-center px-3 pt-[8vh] md:pt-[14vh] ${APP_FONT_CLASSES}`} onKeyDown={onKeyDown}>
      <div className="absolute inset-0 bg-black/50" aria-hidden onClick={onClose} />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Search and commands"
        className="relative flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-3xl border-[2px] border-[#141414] bg-[#FAF9F5] shadow-[0_8px_0_#141414]"
      >
        <div className="flex items-center gap-3 border-b-[1.5px] border-[#141414] px-4">
          <Search className="size-4 shrink-0 text-text-secondary" aria-hidden />
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={current >= 0 ? `${uid}-opt-${current}` : undefined}
            aria-autocomplete="list"
            aria-label="Search pages, agents and actions"
            autoComplete="off"
            spellCheck={false}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            placeholder="Search pages, agents and actions"
            className="h-12 min-w-0 flex-1 bg-transparent text-base text-text-primary outline-none placeholder:text-text-secondary"
          />
          <kbd className="hidden rounded-full border border-[#141414] bg-[#E2DDD3] px-2 py-0.5 text-[11px] font-semibold text-[#141414] md:block">Esc</kbd>
        </div>

        <ul ref={listRef} id={listId} role="listbox" aria-label="Results" className="min-h-0 flex-1 overflow-y-auto p-2 space-y-1">
          {results.map((a, i) => (
            <li
              key={a.id}
              id={`${uid}-opt-${i}`}
              role="option"
              aria-selected={i === current}
              onMouseMove={() => {
                if (i !== current) setActive(i);
                try { router.prefetch(a.href); } catch {}
              }}
              onMouseEnter={() => {
                try { router.prefetch(a.href); } catch {}
              }}
              onClick={() => run(a)}
              className={cn(
                "flex min-h-11 cursor-pointer items-center gap-3 rounded-full border border-transparent px-4 py-2 text-sm text-[#141414] transition-all",
                i === current ? "border-[#141414] bg-[#F3F59E] font-medium shadow-[0_2px_0_#141414]" : "hover:bg-[#E2DDD3]/60"
              )}
            >
              {a.agent ? (
                <Bot className="size-4 shrink-0 text-text-secondary" aria-hidden />
              ) : (
                <ArrowRight className="size-4 shrink-0 text-text-secondary" aria-hidden />
              )}
              <span className="min-w-0 flex-1 truncate">{a.label}</span>
              <span className="shrink-0 text-xs text-text-secondary">{a.hint}</span>
              {i === current && <CornerDownLeft className="hidden size-3.5 shrink-0 text-[#141414] md:block" aria-hidden />}
            </li>
          ))}
        </ul>

        <div role="status" className="border-t-[1.5px] border-[#141414] px-4 py-2.5 text-xs text-[#141414]/75 bg-[#E2DDD3]/50">
          {results.length === 0
            ? `Nothing matches "${query.trim()}".`
            : loading
              ? "Loading your agents..."
              : `${results.length} ${results.length === 1 ? "result" : "results"}. Use the arrow keys and Enter.`}
        </div>
      </div>
    </div>,
    document.body
  );
}
