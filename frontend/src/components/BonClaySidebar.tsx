"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ComponentType } from "react";
import useSWR from "swr";
import { AnimatePresence, motion } from "motion/react";
import { Menu, X, PanelLeftClose, PanelLeftOpen, LogOut, HardDrive, Cloud } from "lucide-react";
import {
  AgentsIcon,
  AnalyticsIcon,
  ApprovalsIcon,
  ApiKeysIcon,
  EvaluationsIcon,
  KnowledgeIcon,
  OverviewIcon,
  PlaygroundIcon,
  SayItOnceIcon,
  StudioIcon,
  VerifyIcon,
  VoicesIcon,
  type NavIconProps,
} from "./icons/NavIcons";
import { useHealth } from "@/hooks/useApi";
import { useArea } from "@/hooks/useArea";
import { AREAS, type Area } from "@/lib/areas";
import { AreaSwitch } from "./AreaSwitch";
import { api } from "@/lib/api";
import { useAuth } from "./AuthProvider";
import { Logo, LogoMark } from "./Logo";
import { ThemeToggle } from "./ThemeToggle";
import { ResourceIndicator } from "./resources/ResourceIndicator";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface NavItem { label: string; href: string; icon: ComponentType<NavIconProps>; badge?: "approvals" }
interface NavGroup { id: string; title: string | null; items: NavItem[] }

/** Calls and bookings: a phone handset with a voice bar (the nav icon set has none; same grid, stroke and active accent). */
function CallsIcon({ className, active }: NavIconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden className={className}>
      <path d="M5.2 4h3.1l1.5 3.8-2 1.3a10.5 10.5 0 0 0 5.1 5.1l1.3-2 3.8 1.5v3.1a1.7 1.7 0 0 1-1.8 1.7A14.6 14.6 0 0 1 3.5 5.8 1.7 1.7 0 0 1 5.2 4Z" />
      <g stroke={active ? "var(--area-accent, #F0A23A)" : "currentColor"}>
        <path d="M15 4.5a5 5 0 0 1 4.5 4.5" />
        <path d="M15 8a1.6 1.6 0 0 1 1 1" />
      </g>
    </svg>
  );
}

/** Campaigns: a megaphone (the nav icon set has none; same grid and stroke). */
function CampaignsIcon({ className, active }: NavIconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden className={className}>
      <path d="M4 10v4a1 1 0 0 0 1 1h2.5L14 19V5L7.5 9H5a1 1 0 0 0-1 1Z" />
      <g stroke={active ? "var(--area-accent, #F0A23A)" : "currentColor"}>
        <path d="M17.5 9.5a3.5 3.5 0 0 1 0 5" />
        <path d="M7.5 15l1 4.5" />
      </g>
    </svg>
  );
}

/** Each area has its own short list, grouped by the job a person is doing. A page that belongs to the other area is not
 *  listed here, but it is one tap away (the switch at the top), and the command palette finds every page. */
const NAV: Record<Area, NavGroup[]> = {
  studio: [
    { id: "home", title: null, items: [{ label: "Studio home", href: "/dashboard/studio/home", icon: OverviewIcon }] },
    {
      id: "make",
      title: "Make audio",
      items: [
        { label: "Voice Studio", href: "/dashboard/studio", icon: StudioIcon },
        { label: "Say it once", href: "/dashboard/say-it-once", icon: SayItOnceIcon },
      ],
    },
    {
      id: "library",
      title: "Voices and checks",
      items: [
        { label: "Voices", href: "/dashboard/voices", icon: VoicesIcon },
        // Public page outside the dashboard: anyone can check a clip, so it also works as a link to share.
        { label: "Verify a clip", href: "/verify", icon: VerifyIcon },
      ],
    },
  ],
  agents: [
    { id: "home", title: null, items: [{ label: "Agents home", href: "/dashboard", icon: OverviewIcon }] },
    {
      id: "setup",
      title: "Set up",
      items: [
        { label: "Agents", href: "/dashboard/studio/agents", icon: AgentsIcon },
        { label: "Documents", href: "/dashboard/knowledge", icon: KnowledgeIcon },
        // Voices is shared: an agent needs a voice too, so it is listed here as well and keeps you in Agents.
        { label: "Voices", href: "/dashboard/voices", icon: VoicesIcon },
      ],
    },
    {
      id: "run",
      title: "Try and run",
      items: [
        { label: "Talk to your agent", href: "/dashboard/playground", icon: PlaygroundIcon },
        { label: "Calls and bookings", href: "/dashboard/calls", icon: CallsIcon },
        { label: "Campaigns", href: "/dashboard/campaigns", icon: CampaignsIcon },
        { label: "Approvals", href: "/dashboard/approvals", icon: ApprovalsIcon, badge: "approvals" },
      ],
    },
    {
      id: "check",
      title: "How it is doing",
      items: [
        { label: "Results", href: "/dashboard/analytics", icon: AnalyticsIcon },
        { label: "Safety check", href: "/dashboard/eval", icon: EvaluationsIcon },
      ],
    },
  ],
};
/** In the footer of both areas. */
const SHARED_ITEMS: NavItem[] = [{ label: "API keys", href: "/dashboard/keys", icon: ApiKeysIcon }];
/** Every link of either area: "the most specific match wins" is decided against all of them. */
const NAV_ITEMS: NavItem[] = [...NAV.studio.flatMap((g) => g.items), ...NAV.agents.flatMap((g) => g.items), ...SHARED_ITEMS];

type ModelState = "ready" | "loading" | "offline";
interface ModelDot { name: string; state: ModelState }

/** The one line at the foot of the sidebar. Voice engines that are not loaded yet are normal (they load on first use). */
export type SystemState = "ready" | "starting" | "attention";
export const SYSTEM_LABEL: Record<SystemState, string> = { ready: "Ready", starting: "Starting", attention: "Needs attention" };
const SYSTEM_DOT: Record<SystemState, string> = {
  ready: "bg-accent-success",
  starting: "bg-accent-warning animate-pulse motion-reduce:animate-none",
  attention: "bg-accent-error",
};

/** apiDown: the server did not answer. `models` is undefined until the first answer; an engine that is still loading
 *  counts as starting; engines that are simply not loaded yet do not. */
export function systemState(apiDown: boolean, models: ModelDot[] | undefined): SystemState {
  if (apiDown) return "attention";
  if (models === undefined) return "starting";
  return models.some((m) => m.state === "loading") ? "starting" : "ready";
}

const MODEL_LABELS: Record<string, string> = {
  kokoro: "Kokoro",
  qwen3: "Qwen3",
  qwen3_tts: "Qwen3",
  qwen3_0_6b: "Qwen3",
  indicf5: "IndicF5",
  indic_f5: "IndicF5",
  indic_speak: "Indic-Speak",
  chatterbox_ml: "Chatterbox",
  chatterbox: "Chatterbox",
  dhvaani: "DhVaani",
};

function toState(v: unknown): ModelState {
  if (v === true) return "ready";
  if (typeof v === "string") {
    const s = v.toLowerCase();
    if (["ready", "loaded", "ok", "online", "available", "healthy"].includes(s)) return "ready";
    if (["loading", "warming", "downloading", "starting"].includes(s)) return "loading";
    return "offline";
  }
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    if ("status" in o) return toState(o.status);
    if ("loaded" in o) return toState(o.loaded);
    if ("ready" in o) return toState(o.ready);
  }
  return "offline";
}

/** Accepts {engines: [...]} (GET /api/models/status), {kokoro: "ready"}, {models: {...}}, or [{name, status}]. */
function normaliseModels(data: unknown): ModelDot[] {
  if (!data || typeof data !== "object") return [];
  const root = data as Record<string, unknown>;
  const src = (root.engines ?? root.models ?? data) as unknown;
  const label = (k: string) => MODEL_LABELS[k.toLowerCase()] ?? k;
  if (Array.isArray(src)) {
    return src
      .filter((m): m is Record<string, unknown> => !!m && typeof m === "object")
      .map((m) => ({ name: label(String(m.id ?? m.name ?? "model")), state: toState(m) }));
  }
  if (src && typeof src === "object") {
    return Object.entries(src as Record<string, unknown>).map(([k, v]) => ({ name: label(k), state: toState(v) }));
  }
  return [];
}

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(String(res.status));
  return res.json();
};

/** How many meeting requests wait for a decision (GET /api/meetings counts every status, whichever one is asked for). */
async function fetchPending(): Promise<number> {
  const body = await api.get<{ counts?: { pending?: number } }>("/api/meetings?status=pending");
  return Math.max(0, Number(body?.counts?.pending) || 0);
}

interface Props {
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

export function BonClaySidebar({ collapsed = false, onToggleCollapsed }: Props) {
  const pathname = usePathname();
  const { area, known, switchTo } = useArea();
  const { user, signOut } = useAuth();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [local, setLocal] = useState(true);

  const { data, error } = useSWR(`${API_URL}/api/models/status`, fetcher, {
    refreshInterval: 15000,
    shouldRetryOnError: false,
    revalidateOnFocus: false,
  });
  const apiDown = !!error;
  const health = useHealth();
  // No Supabase on the server: there is no account to sign in to, everything lives on this computer.
  const localMode = health.data?.supabase?.configured === false;
  // Signed statement of how the server is configured; "Offline, verified" only when it claims nothing leaves this computer.
  const attestation = useSWR<{ statement?: { claim?: string } }>(
    localMode ? "/api/privacy/attestation" : null,
    (url: string) => api.get(url),
    { revalidateOnFocus: false, shouldRetryOnError: false },
  );
  const offlineVerified = attestation.data?.statement?.claim === "configured_to_keep_data_on_this_machine";
  const models = apiDown || !data ? undefined : normaliseModels(data);
  const system = systemState(apiDown, models);
  const systemHint = apiDown
    ? "BonClay cannot reach its server. Check that it is running."
    : system === "starting"
      ? "A voice engine is starting up."
      : `Everything is ready. Voice engines load the first time they are used${models?.length ? ` (${models.filter((m) => m.state === "ready").length} of ${models.length} loaded now)` : ""}.`;

  // Requests waiting for the owner's decision. Cheap (one small list), refreshed every 45 s and when the tab is focused.
  // Only the Agents area has Approvals, so a Studio-only creator never makes this request.
  const approvals = useSWR<number>(known && area === "agents" ? "sidebar:approvals-pending" : null, fetchPending, {
    refreshInterval: 45000,
    revalidateOnFocus: true,
    focusThrottleInterval: 15000,
    shouldRetryOnError: false,
  });
  const pending = area === "agents" ? (approvals.data ?? 0) : 0;

  useEffect(() => {
    const read = () => {
      try {
        setLocal((localStorage.getItem("bonclay_storage_mode") ?? "local") === "local");
      } catch {
        /* default local */
      }
    };
    read();
    window.addEventListener("storageModeChanged", read);
    return () => window.removeEventListener("storageModeChanged", read);
  }, []);

  const toggleStorage = () => {
    const next = !local;
    setLocal(next);
    try {
      localStorage.setItem("bonclay_storage_mode", next ? "local" : "cloud");
    } catch {
      /* ignore */
    }
    window.dispatchEvent(new Event("storageModeChanged"));
  };

  // Close the mobile drawer on navigation.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setDrawerOpen(false), [pathname]);

  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  const isActive = (href: string) => {
    if (href === "/dashboard") return pathname === "/dashboard";
    if (!(pathname === href || pathname.startsWith(href + "/"))) return false;
    // the most specific matching item wins (Agents vs Voice Studio)
    return !NAV_ITEMS.some((i) => i.href.length > href.length && (pathname === i.href || pathname.startsWith(i.href + "/")));
  };

  const name =
    (user?.user_metadata?.name as string | undefined) ||
    (user?.user_metadata?.full_name as string | undefined) ||
    user?.email?.split("@")[0] ||
    (localMode ? (offlineVerified ? "Offline, verified" : "Local mode") : "Guest");

  const navLink = ({ label, href, icon: Icon, badge }: NavItem, compact: boolean, inDrawer: boolean) => {
    const active = isActive(href);
    const count = badge === "approvals" ? pending : 0;
    const countText = count > 99 ? "99+" : String(count);
    const waiting = `${count} waiting for your decision`;
    return (
      <Link
        key={href}
        href={href}
        title={compact ? (count ? `${label}: ${waiting}` : label) : undefined}
        aria-current={active ? "page" : undefined}
        className={cn(
          "group relative flex h-11 items-center rounded-lg md:h-9 text-sm font-medium transition-colors duration-150",
          compact ? "justify-center" : "gap-3 px-3",
          active ? "text-text-primary" : "text-text-secondary hover:text-text-primary"
        )}
      >
        {active && (
          <motion.span
            layoutId={inDrawer ? "nav-active-drawer" : "nav-active"}
            className="absolute inset-0 rounded-lg bg-[color-mix(in_srgb,var(--area-accent)_14%,transparent)]"
            transition={{ type: "spring", stiffness: 420, damping: 34 }}
          >
            <span aria-hidden className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-[var(--area-accent)]" />
          </motion.span>
        )}
        {!active && <span aria-hidden className="absolute inset-0 rounded-lg bg-bg-elevated opacity-0 transition-opacity group-hover:opacity-100" />}
        <Icon active={active} className="relative size-[19px] shrink-0" />
        {!compact && <span className="relative min-w-0 flex-1 truncate">{label}</span>}
        {count > 0 && (
          <span
            className={cn(
              "relative grid place-items-center rounded-full bg-accent-warning text-[11px] font-semibold leading-none text-black",
              compact ? "absolute right-1 top-0.5 h-4 min-w-4 px-1" : "h-5 min-w-5 px-1.5"
            )}
          >
            <span aria-hidden>{countText}</span>
            <span className="sr-only">{waiting}</span>
          </span>
        )}
      </Link>
    );
  };

  /** `compact` hides labels (desktop collapsed). Drawer always renders expanded. */
  const content = (compact: boolean, inDrawer: boolean) => (
    <div data-area={area} className={cn("flex h-full flex-col", inDrawer && "overflow-y-auto")}>
      <div className={cn("flex h-14 shrink-0 items-center border-b border-border", compact ? "justify-center" : "justify-between px-4")}>
        <Link href="/" aria-label="BonClay home" className="flex items-center">
          {compact ? <LogoMark size={28} /> : <Logo size={26} />}
        </Link>
        {inDrawer ? (
          <button onClick={() => setDrawerOpen(false)} aria-label="Close menu" className="grid size-11 place-items-center rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary">
            <X className="size-5" />
          </button>
        ) : null}
      </div>

      <div className={cn("shrink-0 pt-3", compact ? "px-1" : "px-3")}>
        <AreaSwitch area={area} onChange={switchTo} compact={compact} />
      </div>

      {/* In the drawer the whole panel scrolls (one scroll area), so the list and the footer are always reachable on a short phone. */}
      <nav aria-label="Main" className={cn("flex-1 px-2 pb-3 pt-4", !inDrawer && "overflow-y-auto overflow-x-hidden")}>
        {NAV[area].map((group, i) => (
          <div
            key={group.id}
            role="group"
            aria-label={group.title ?? `${AREAS.find((a) => a.id === area)?.label} home`}
            className={cn(i > 0 && (compact ? "mt-3 border-t border-border pt-3" : "mt-4 border-t border-border pt-3"))}
          >
            {group.title && !compact && (
              <p className="flex items-center gap-2 px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
                <span aria-hidden className="h-2.5 w-0.5 rounded-full bg-[var(--area-accent)]" />
                {group.title}
              </p>
            )}
            <div className="space-y-0.5">{group.items.map((item) => navLink(item, compact, inDrawer))}</div>
          </div>
        ))}
      </nav>

      <div className="shrink-0 space-y-2.5 border-t border-border p-3">
        {/* In both areas: the same API keys for everything. */}
        <div role="group" aria-label="Developers" className="space-y-0.5">
          {SHARED_ITEMS.map((item) => navLink(item, compact, inDrawer))}
        </div>

        {/* System status: one line, not one dot per voice engine */}
        <div
          role="status"
          aria-label={`System status: ${SYSTEM_LABEL[system]}`}
          title={systemHint}
          className={cn("flex items-center text-xs text-text-secondary", compact ? "justify-center" : "gap-2 px-1")}
        >
          <span aria-hidden className={cn("size-2 shrink-0 rounded-full", SYSTEM_DOT[system])} />
          {!compact ? <span>{SYSTEM_LABEL[system]}</span> : <span className="sr-only">{SYSTEM_LABEL[system]}</span>}
        </div>

        {/* Which models are loaded and how much memory they take (opens the "Models and memory" panel) */}
        <ResourceIndicator compact={compact} />

        {/* Storage */}
        <button
          onClick={toggleStorage}
          title={`Storage: ${local ? "local" : "cloud"}. Click to switch.`}
          className={cn(
            "flex h-9 w-full items-center rounded-lg border border-border text-xs font-medium text-text-secondary transition-colors hover:border-accent-primary/60 hover:text-text-primary",
            compact ? "justify-center" : "gap-2 px-3"
          )}
        >
          {local ? <HardDrive className="size-4" /> : <Cloud className="size-4 text-accent-secondary" />}
          {!compact && <span>{local ? "Local storage" : "Cloud storage"}</span>}
        </button>

        {/* User */}
        <div className={cn("flex items-center", compact ? "flex-col gap-2" : "gap-2.5")}>
          <div className="grid size-8 shrink-0 place-items-center rounded-full bg-brand text-xs font-semibold text-[var(--color-on-accent)]" aria-hidden>
            {name.charAt(0).toUpperCase()}
          </div>
          {!compact && (
            <div className="min-w-0 flex-1">
              <p className="truncate text-[13px] font-medium text-text-primary">{name}</p>
              <p className="truncate text-[11px] text-text-secondary">{user?.email ?? (localMode ? (offlineVerified ? "Configured so nothing leaves this computer" : "Data stays on this computer") : "Not signed in")}</p>
            </div>
          )}
          <ThemeToggle />
          {user ? (
            <button onClick={() => signOut()} aria-label="Sign out" title="Sign out" className="grid size-9 place-items-center rounded-lg text-text-secondary transition-colors hover:bg-bg-elevated hover:text-accent-error">
              <LogOut className="size-4" />
            </button>
          ) : localMode ? null : (
            <Link href="/login" className="text-xs font-medium text-accent-secondary">Sign in</Link>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop */}
      <aside
        className={cn(
          "fixed left-0 top-0 z-40 hidden h-screen flex-col border-r border-border bg-bg-base md:flex",
          "transition-[width] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]",
          collapsed ? "w-16" : "w-60"
        )}
      >
        {content(collapsed, false)}
        <button
          onClick={onToggleCollapsed}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!collapsed}
          className="absolute -right-3 top-[68px] grid size-6 place-items-center rounded-full border border-border bg-bg-surface text-text-secondary transition-colors hover:text-text-primary"
        >
          {collapsed ? <PanelLeftOpen className="size-3.5" /> : <PanelLeftClose className="size-3.5" />}
        </button>
      </aside>

      {/* Mobile top bar */}
      <div className="fixed inset-x-0 top-0 z-40 flex h-14 items-center gap-3 border-b border-border bg-bg-base/90 px-3 backdrop-blur md:hidden">
        <button onClick={() => setDrawerOpen(true)} aria-label="Open menu" aria-expanded={drawerOpen} className="grid size-10 place-items-center rounded-lg text-text-primary hover:bg-bg-elevated">
          <Menu className="size-5" />
        </button>
        <Logo size={24} />
        <span data-area={area} className="ml-1 rounded-full border border-[var(--area-accent)]/50 px-2.5 py-0.5 text-xs font-semibold text-[var(--area-accent)]">
          {AREAS.find((a) => a.id === area)?.label}
        </span>
      </div>

      {/* Mobile drawer */}
      <AnimatePresence>
        {drawerOpen && (
          <>
            <motion.div
              key="scrim"
              className="fixed inset-0 z-50 bg-black/60 md:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setDrawerOpen(false)}
            />
            <motion.aside
              key="drawer"
              role="dialog"
              aria-modal="true"
              aria-label="Navigation"
              className="fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] border-r border-border bg-bg-base md:hidden"
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            >
              {content(false, true)}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}

export default BonClaySidebar;
