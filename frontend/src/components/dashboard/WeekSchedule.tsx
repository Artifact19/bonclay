"use client";

import { useId, useState, type KeyboardEvent, type ReactNode } from "react";
import { CalendarClock, Check, ChevronLeft, ChevronRight, X } from "lucide-react";
import { toast } from "sonner";
import { Badge, Button, Card, EmptyState, ErrorState, Skeleton } from "@/components/ui";
import { useMeetingWeek } from "@/hooks/useApi";
import { api } from "@/lib/api";
import { errMessage } from "@/components/dashboard/format";
import type { MeetingCell, MeetingStatus, MeetingWeek } from "@/types/api";

const ROLE_LABEL: Record<string, string> = { student: "student", parent: "parent", faculty: "staff member", other: "" };

const HATCH = "repeating-linear-gradient(135deg, transparent 0 4px, var(--color-border) 4px 5px)";

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "Thu 22 Oct" (or "22 Oct") from an ISO date, the same in every locale. */
function shortDay(iso: string, withWeekday = true): string {
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  const day = `${d.getDate()} ${MONTHS[d.getMonth()]}`;
  return withWeekday ? `${WEEKDAYS[d.getDay()]} ${day}` : day;
}

/** The ISO date `days` after `iso` (local calendar days, so daylight saving cannot shift it). */
function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T12:00:00`);
  d.setDate(d.getDate() + days);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

const firstName = (name: string) => name.trim().split(/\s+/)[0] || name;

/** What a screen reader hears for a cell: "Thursday 11:30, asked for by Riya, pending". */
export function cellLabel(weekday: string, time: string, c: MeetingCell, actionable: boolean): string {
  const at = `${weekday} ${time}`;
  switch (c.state) {
    case "asked_for": return `${at}, asked for by ${firstName(c.request?.name ?? "someone")}, pending. Press to answer`;
    case "booked": return `${at}, booked for ${firstName(c.request?.name ?? "someone")}${c.request?.on_calendar ? ", on Google Calendar" : ""}. Press to open`;
    case "busy": {
      const why = c.reason ? `: ${c.reason}` : "";
      const sample = c.source === "sample" ? " (sample)" : "";
      return `${at}, busy${why}${sample}${actionable ? ". Press to make it free" : ""}`;
    }
    case "free": return `${at}, free${c.too_soon ? " (too soon for the agent to offer)" : ""}. Press to mark busy`;
    case "past": return `${at}, already past`;
    default: return `${at}, closed`;
  }
}

function Switch({ checked, onChange, label, description, busy }: {
  checked: boolean; onChange: (next: boolean) => void; label: string; description: ReactNode; busy?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-start gap-3">
      <button type="button" role="switch" aria-checked={checked} aria-labelledby={`${id}-l`} aria-describedby={`${id}-d`}
        disabled={busy} onClick={() => onChange(!checked)}
        className={`relative mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors disabled:opacity-60 [@media(pointer:coarse)]:h-7 [@media(pointer:coarse)]:w-12 ${checked ? "border-transparent bg-accent-primary" : "border-text-muted bg-bg-elevated"}`}>
        <span aria-hidden className={`inline-block size-4 rounded-full transition-transform motion-reduce:transition-none ${checked ? "translate-x-6 bg-[var(--color-on-accent)]" : "translate-x-1 bg-text-secondary"}`} />
      </button>
      <div className="min-w-0 text-sm">
        <p id={`${id}-l`} className="font-medium text-text-primary">{label}</p>
        <p id={`${id}-d`} className="mt-0.5 text-text-secondary">{description}</p>
      </div>
    </div>
  );
}

function Legend() {
  const swatch = "inline-block h-4 w-6 shrink-0 rounded";
  return (
    <ul aria-label="Legend" className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-secondary">
      <li className="flex items-center gap-1.5"><span aria-hidden className={`${swatch} border border-text-muted bg-bg-surface`} />Free</li>
      <li className="flex items-center gap-1.5"><span aria-hidden className={`${swatch} border border-border`} style={{ backgroundImage: HATCH }} />Busy</li>
      <li className="flex items-center gap-1.5"><span aria-hidden className={`${swatch} border-2 border-text-primary bg-bg-surface`} />Asked for</li>
      <li className="flex items-center gap-1.5"><span aria-hidden className={`${swatch} flex items-center justify-center bg-text-primary text-bg-base`}><Check className="size-3" /></span>Booked</li>
    </ul>
  );
}

/** Moves the focus between the grid's buttons with the arrow keys (the Tab key still visits each one). */
function onGridKey(e: KeyboardEvent<HTMLTableElement>) {
  const keys: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  const step = keys[e.key];
  const here = (e.target as HTMLElement).closest<HTMLElement>("[data-r]");
  if (!step || !here) return;
  let r = Number(here.dataset.r) + step[0];
  let c = Number(here.dataset.c) + step[1];
  for (let guard = 0; guard < 40; guard++, r += step[0], c += step[1]) {
    const next = e.currentTarget.querySelector<HTMLElement>(`[data-r="${r}"][data-c="${c}"]`);
    if (next) { e.preventDefault(); next.focus(); return; }
    if (r < 0 || c < 0 || r > 60 || c > 7) return;
  }
}

interface Selected { date: string; time: string; weekday: string; cell: MeetingCell }

export function WeekSchedule({ onDecision, onShowRequest }: {
  /** The list of requests below changed (one was approved or declined here). */
  onDecision?: () => void;
  /** Open the request's full card in the list. */
  onShowRequest?: (requestId: string, status: MeetingStatus) => void;
}) {
  const [start, setStart] = useState<string | undefined>(undefined);
  const { data, error, isLoading, mutate } = useMeetingWeek(start);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [deciding, setDeciding] = useState<"approve" | "decline" | null>(null);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);

  const openDays = data?.days.filter((d) => d.open) ?? [];
  const hasHours = !!data && openDays.length > 0 && data.times.length > 0;
  const currentMonday = data ? addDays(data.today, -((new Date(`${data.today}T12:00:00`).getDay() + 6) % 7)) : undefined;
  const isCurrent = !!data && data.start === currentMonday;
  const canGoBack = !!data && !!currentMonday && data.start > currentMonday;
  const canGoOn = !!data && addDays(data.start, 7) <= addDays(data.today, data.max_days_ahead);
  const goTo = (iso: string | undefined) => { setSelected(null); setDecisionError(null); setStart(iso); };

  const toggle = async (date: string, time: string, cell: MeetingCell) => {
    if (!data) return;
    const next: "busy" | "free" = cell.state === "free" ? "busy" : "free";
    const optimistic: MeetingCell = next === "busy" ? { state: "busy", reason: "Marked busy by you", source: "owner" } : { state: "free" };
    const before = data;
    const patched: MeetingWeek = { ...data, cells: { ...data.cells, [date]: { ...data.cells[date], [time]: optimistic } } };
    mutate(patched, { revalidate: false });
    try {
      await api.put("/api/meetings/slots", { date, time, state: next });
      mutate();
    } catch (e) {
      mutate(before, { revalidate: false });
      toast.error(errMessage(e, "Could not change that time."));
    }
  };

  const setSwitch = async (key: "availability_check" | "sample_schedule", value: boolean) => {
    setSaving(key);
    try {
      await api.put("/api/meetings/settings", { [key]: value });
      await mutate();
      toast.success(key === "availability_check"
        ? (value ? "The agent will check this schedule during calls." : "The agent will take the caller's preferred time for you to confirm.")
        : (value ? "Sample schedule is on." : "Sample schedule is off."));
    } catch (e) {
      toast.error(errMessage(e, "Could not save that."));
    } finally {
      setSaving(null);
    }
  };

  const decide = async (kind: "approve" | "decline") => {
    const id = selected?.cell.request?.request_id;
    if (!id) return;
    setDeciding(kind);
    setDecisionError(null);
    try {
      await api.post(`/api/meetings/${encodeURIComponent(id)}/${kind}`, {});
      toast.success(kind === "approve" ? `Approved the request from ${selected.cell.request?.name}` : `Declined the request from ${selected.cell.request?.name}`);
      setSelected(null);
      await mutate();
      onDecision?.();
    } catch (e) {
      setDecisionError(errMessage(e, "Could not save the decision."));
    } finally {
      setDeciding(null);
    }
  };

  const title = `${data?.label || "Head of department"}, this week`;
  const range = data && openDays.length ? `${shortDay(openDays[0].date, false)} to ${shortDay(openDays[openDays.length - 1].date, false)}` : "";
  const check = data?.settings.availability_check ?? true;
  const badCalendar = openDays.filter((d) => d.calendar_error);

  return (
    <Card variant="solid" role="region" className="space-y-4 p-5" aria-labelledby="week-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="week-title" className="text-[15px] font-semibold text-text-primary">{isCurrent || !data ? title : `${data.label || "Head of department"}, week of ${shortDay(data.start, false)}`}</h2>
            {data?.settings.sample_schedule && <Badge tone="neutral">Sample schedule</Badge>}
          </div>
          <p className="mt-0.5 text-sm text-text-secondary">
            {data ? `${data.slot_minutes}-minute slots` : "Loading the schedule"}
            {range && <span aria-live="polite"> · {range}</span>}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="outline" aria-label="Previous week" disabled={!canGoBack} onClick={() => data && goTo(addDays(data.start, -7))}>
            <ChevronLeft className="size-4" />
          </Button>
          <Button size="sm" variant="outline" disabled={isCurrent || !data} onClick={() => goTo(undefined)}>This week</Button>
          <Button size="sm" variant="outline" aria-label="Next week" disabled={!canGoOn} onClick={() => data && goTo(addDays(data.start, 7))}>
            <ChevronRight className="size-4" />
          </Button>
        </div>
      </div>

      {isLoading && !data && <Skeleton className="h-56 w-full" />}
      {error && !data && <ErrorState title="Could not load the schedule" message={errMessage(error, "Check that the API is running.")} onRetry={() => mutate()} />}
      {data && !hasHours && (
        <EmptyState icon={<CalendarClock />} title="No office hours are set"
          description="The agent has no times to offer. Office hours are part of the meeting settings kept with the meeting requests." />
      )}

      {data && hasHours && (
        <>
          {badCalendar.length > 0 && (
            <p role="status" className="text-sm text-accent-warning">
              Google Calendar could not be read for {badCalendar.map((d) => shortDay(d.date)).join(", ")}. Busy times from the calendar may be missing there.
            </p>
          )}
          <div className="relative -mx-1 overflow-x-auto px-1 pb-1">
            <table className="w-full min-w-[34rem] table-fixed border-separate border-spacing-1 text-xs" onKeyDown={onGridKey}>
              <caption className="sr-only">
                Schedule for the week of {shortDay(data.start)}. Each cell is a {data.slot_minutes}-minute slot: free, busy, asked for or booked.
              </caption>
              <colgroup>
                <col className="w-14" />
                {openDays.map((d) => <col key={d.date} />)}
              </colgroup>
              <thead>
                <tr>
                  <td />
                  {openDays.map((d) => (
                    <th key={d.date} scope="col" className="pb-1 text-center font-medium text-text-secondary">
                      <span aria-hidden>{shortDay(d.date)}</span>
                      <span className="sr-only">{d.weekday} {shortDay(d.date, false)}{d.full ? ", full for the day" : ""}</span>
                      {d.full && <span aria-hidden className="block text-[10px] font-semibold text-accent-warning">Full</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.times.map((t, r) => (
                  <tr key={t}>
                    <th scope="row" className="sticky left-0 z-[1] bg-bg-surface pr-1 text-right font-normal tabular-nums text-text-secondary">{t}</th>
                    {openDays.map((d, c) => {
                      const cell = data.cells[d.date]?.[t] ?? { state: "closed" as const };
                      const editable = cell.state === "free" || (cell.state === "busy" && (cell.source === "sample" || cell.source === "owner"));
                      const label = cellLabel(d.weekday, t, cell, editable);
                      const box = "grid h-9 w-full place-items-center rounded-md px-0.5 text-[11px] leading-3";
                      const focus = "focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent-secondary";
                      let body: React.ReactNode;
                      if (cell.state === "closed" || cell.state === "past") {
                        body = (
                          <span className={`${box} bg-bg-elevated/50 text-text-muted`} title={cell.reason}>
                            <span className="sr-only">{label}</span>
                          </span>
                        );
                      } else if (cell.state === "asked_for" || cell.state === "booked") {
                        const booked = cell.state === "booked";
                        body = (
                          <button type="button" data-r={r} data-c={c} aria-label={label}
                            aria-expanded={selected?.date === d.date && selected?.time === t}
                            onClick={() => { setDecisionError(null); setSelected({ date: d.date, time: t, weekday: d.weekday, cell }); }}
                            className={`${box} ${focus} cursor-pointer font-semibold ${booked ? "gap-0.5 bg-text-primary text-bg-base" : "border-2 border-text-primary bg-bg-surface text-text-primary"}`}>
                            <span aria-hidden className="flex items-center gap-0.5 truncate">
                              {booked && <Check className="size-3 shrink-0" />}
                              <span className="truncate">{firstName(cell.request?.name ?? "")}</span>
                            </span>
                          </button>
                        );
                      } else if (cell.state === "busy") {
                        const content = <span aria-hidden className="truncate text-text-secondary">Busy</span>;
                        body = editable ? (
                          <button type="button" data-r={r} data-c={c} aria-label={label} title={`${cell.reason ?? "Busy"}. Press to make it free`}
                            onClick={() => toggle(d.date, t, cell)}
                            className={`${box} ${focus} cursor-pointer border border-border hover:border-text-primary`} style={{ backgroundImage: HATCH }}>
                            {content}
                          </button>
                        ) : (
                          <span className={`${box} border border-border`} style={{ backgroundImage: HATCH }} title={cell.reason}>
                            {content}
                            <span className="sr-only">{label}</span>
                          </span>
                        );
                      } else {
                        body = (
                          <button type="button" data-r={r} data-c={c} aria-label={label} title="Free. Press to mark busy"
                            onClick={() => toggle(d.date, t, cell)}
                            className={`${box} ${focus} cursor-pointer border border-text-muted bg-bg-surface hover:border-text-primary`}>
                            {cell.too_soon && <span aria-hidden className="text-[10px] text-text-muted">soon</span>}
                          </button>
                        );
                      }
                      return <td key={d.date} className="p-0">{body}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Legend />
        </>
      )}

      {selected?.cell.request && (
        <div role="group" aria-label="Selected request" className="space-y-2 rounded-xl border border-border bg-bg-elevated p-3">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="min-w-0">
              <p className="text-sm font-semibold text-text-primary">
                {selected.cell.request.name}
                {ROLE_LABEL[selected.cell.request.role] ? <span className="font-normal text-text-secondary">, {ROLE_LABEL[selected.cell.request.role]}</span> : null}
              </p>
              <p className="text-sm text-text-secondary">
                {selected.cell.state === "booked" ? "Booked for " : "Asked for "}{shortDay(selected.date)}, {selected.time}
                {selected.cell.state === "booked" && selected.cell.request.on_calendar ? " (on Google Calendar)" : ""}
                {selected.cell.more ? ` · ${selected.cell.more} more request${selected.cell.more > 1 ? "s" : ""} for this time` : ""}
              </p>
            </div>
            <Badge tone={selected.cell.state === "booked" ? "success" : "warning"} dot>{selected.cell.state === "booked" ? "Approved" : "Waiting"}</Badge>
          </div>
          {decisionError && <p role="alert" className="text-sm text-accent-error">{decisionError}</p>}
          <div className="flex flex-wrap gap-2">
            {selected.cell.state === "asked_for" && (
              <>
                <Button size="sm" onClick={() => decide("approve")} disabled={deciding !== null}><Check className="size-4" />{deciding === "approve" ? "Approving..." : "Approve"}</Button>
                <Button size="sm" variant="outline" onClick={() => decide("decline")} disabled={deciding !== null}><X className="size-4" />{deciding === "decline" ? "Declining..." : "Decline"}</Button>
              </>
            )}
            <Button size="sm" variant="ghost" onClick={() => selected.cell.request && onShowRequest?.(selected.cell.request.request_id, selected.cell.state === "booked" ? "approved" : "pending")}>
              Show the full request
            </Button>
            <Button size="sm" variant="ghost" onClick={() => { setSelected(null); setDecisionError(null); }}>Close</Button>
          </div>
        </div>
      )}

      {data && (
        <div className="space-y-4 border-t border-border pt-4">
          <Switch checked={check} busy={saving === "availability_check"} onChange={(v) => setSwitch("availability_check", v)}
            label="Agent checks this schedule during calls"
            description={
              <>
                <span className={check ? "font-medium text-text-primary" : ""}>On: the agent offers only free times and suggests others.</span>{" "}
                <span className={check ? "" : "font-medium text-text-primary"}>Off: the agent takes the caller&apos;s preferred time and tells them the office will confirm.</span>
              </>
            } />
          <Switch checked={data.settings.sample_schedule} busy={saving === "sample_schedule"} onChange={(v) => setSwitch("sample_schedule", v)}
            label="Fill the week with a sample schedule"
            description="Made-up busy times so a demo looks real. They count as busy for the agent while it checks the schedule. Turn it off to show only your blocked times, your calendar and your own marks." />
        </div>
      )}
    </Card>
  );
}
