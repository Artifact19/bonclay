"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import {
  Check,
  ChevronDown,
  CircleCheck,
  Clock,
  Copy,
  Loader2,
  Phone,
  PhoneCall,
  PhoneIncoming,
  PhoneMissed,
  PhoneOff,
  PhoneOutgoing,
  Plus,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui";
import { Modal } from "@/components/voices/Modal";
import type {
  Agent,
  ExotelCallPlaced,
  ExotelCallState,
  ExotelCallStatus,
  ExotelInboundLink,
  ExotelStatus,
  SavedPhoneNumber,
} from "@/types/api";

export const POLL_MS = 2000;
const MAX_POLL_FAILURES = 4;
const TERMINAL: ExotelCallState[] = ["completed", "failed", "busy", "no-answer", "canceled"];

const fieldClass =
  "h-11 w-full rounded-lg border border-border bg-bg-base px-3 text-sm text-text-primary focus:border-accent-primary focus:outline-none disabled:opacity-50";
const codeClass = "rounded bg-bg-elevated px-1.5 py-0.5 font-mono text-xs";

/** Hide any long digit run (a phone number the provider echoed back) so a full number is never shown. */
export function maskDigits(text: string): string {
  return text.replace(/\+?\d[\d\s().-]{7,}\d/g, (m) => `••••${m.replace(/\D/g, "").slice(-2)}`);
}

/** Plain-language messages for the codes POST /api/phone/exotel/call (and the status poll) can answer with. */
export function callErrorMessage(e: unknown): string {
  const code = e instanceof ApiError ? e.code : "";
  const detail = e instanceof Error ? maskDigits(e.message) : "";
  switch (code) {
    case "CONSENT_REQUIRED":
      return "Only call someone who agreed to receive this call. Tick the box to confirm.";
    case "NUMBER_NOT_ALLOWED":
      return "That number is not saved. Add it first.";
    case "PHONE_NOT_CONFIGURED":
      return "Your phone line is not connected yet. Ask the person who installed BonClay to finish the phone setup (see Details).";
    case "RATE_LIMITED":
      return "Too many calls this hour. Wait a while and try again.";
    case "PHONE_ALL_ACCOUNTS_FAILED":
      return `Every phone account refused the call. ${detail}`.trim();
    case "BAD_NUMBER":
      return NUMBER_HINT;
    default:
      // EXOTEL_REFUSED, TWILIO_REFUSED, ... and EXOTEL_UNREACHABLE, TWILIO_UNREACHABLE, ...: one pair per provider.
      if (code.endsWith("_REFUSED")) return `The phone provider refused the call. ${detail}`.trim();
      if (code.endsWith("_UNREACHABLE")) {
        return "Could not reach the phone provider. Check this computer's internet connection and try again.";
      }
      return detail || "The call could not be placed. Try again.";
  }
}

const NUMBER_HINT = "Enter the number in international format, for example +919876543210.";

/** Plain-language messages for POST /api/phone/exotel/numbers and DELETE /api/phone/exotel/numbers/{id}. */
export function numberErrorMessage(e: unknown): string {
  const code = e instanceof ApiError ? e.code : "";
  switch (code) {
    case "CONSENT_REQUIRED":
      return "Tick the box to confirm this person agreed to get a call from an AI agent.";
    case "BAD_NUMBER":
      return "That doesn't look like a phone number. Try the format +91 98765 43210.";
    case "NUMBER_EXISTS":
      return "That number is already saved.";
    case "TOO_MANY_NUMBERS":
      return "You can save up to 20 numbers. Remove one first.";
    case "BAD_LABEL":
      return "Give the number a name of 1 to 40 characters.";
    case "NUMBER_NOT_FOUND":
      return "That number was already removed.";
    case "NUMBER_FROM_ENV":
      return "This number was set up by whoever installed BonClay, so it can only be removed there.";
    default:
      return "Something went wrong. Try again.";
  }
}

/** Loose client check: at least 10 digits once spaces, dashes, dots and brackets are gone. The server normalises. */
export function looksLikeNumber(raw: string): boolean {
  const s = raw.trim().replace(/[\s\-().]/g, "");
  return /^\+?\d{10,15}$/.test(s);
}

/** The number as E.164 (+919876543210), or null: spaces, dashes and brackets are ignored; Indian numbers may be
 *  written 9876543210, 09876543210, 919876543210 or +919876543210 (mirrors normalise_number in the backend). */
export function normaliseNumber(raw: string): string | null {
  const s = raw.replace(/[\s\-().]/g, "");
  let out: string;
  if (s.startsWith("+")) out = s;
  else if (!/^\d+$/.test(s)) return null;
  else if (s.startsWith("00")) out = `+${s.slice(2)}`;
  else if (s.length === 10) out = `+91${s}`;
  else if (s.length === 11 && s.startsWith("0")) out = `+91${s.slice(1)}`;
  else if (s.length === 12 && s.startsWith("91")) out = `+${s}`;
  else return null;
  return /^\+[1-9]\d{7,14}$/.test(out) ? out : null;
}

/** The name shown for a number; the backend labels numbers from its settings file with a technical sentence. */
function displayLabel(n: { label: string; source: "env" | "saved" }): string {
  return n.source === "env" && /\.env/i.test(n.label) ? "Pre-approved number" : n.label;
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.max(0, Math.floor(s % 60))).padStart(2, "0")}`;

interface LiveCall {
  sid: string;
  state: ExotelCallState | "calling";
  durationS: number | null;
  message: string | null;
  connectedAt: number | null;
  /** Which provider and account placed the call, e.g. "Twilio 2" (after any failover). */
  via: string | null;
}

const PROVIDER_NAMES: Record<string, string> = {
  exotel: "Exotel",
  twilio: "Twilio",
  plivo: "Plivo",
  vobiz: "Vobiz",
  telnyx: "Telnyx",
};

/** "Twilio 2": the provider and account of a placed call, from the dial answer. */
export function viaLabel(provider?: string, account?: string): string | null {
  if (!provider) return null;
  const name = PROVIDER_NAMES[provider] ?? provider;
  return account ? `${name} ${account}` : name;
}

function Line({ icon, tone, children }: { icon: ReactNode; tone: "neutral" | "good" | "bad"; children: ReactNode }) {
  const color = tone === "good" ? "text-accent-success" : tone === "bad" ? "text-accent-error" : "text-text-primary";
  return (
    <p className={`flex items-center gap-2 text-sm font-medium ${color}`}>
      <span aria-hidden className="grid size-5 place-items-center">
        {icon}
      </span>
      <span>{children}</span>
    </p>
  );
}

function CopyField({ text, label }: { text: string; label: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setState("copied");
      setTimeout(() => setState("idle"), 2000);
    } catch {
      setState("failed");
    }
  };
  return (
    <div>
      <div className="flex gap-2">
        <input
          readOnly
          value={text}
          aria-label={label}
          onFocus={(e) => e.target.select()}
          className={`${fieldClass} min-w-0 font-mono text-xs`}
        />
        <Button
          variant="outline"
          size="sm"
          className="h-11 shrink-0"
          onClick={copy}
          aria-label={`Copy ${label}`}
          leftIcon={state === "copied" ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
        >
          {state === "copied" ? "Copied" : "Copy"}
        </Button>
      </div>
      {state === "failed" && (
        <p role="alert" className="mt-1 text-xs text-accent-error">
          Could not copy. Select the address and copy it by hand.
        </p>
      )}
    </div>
  );
}

/** Technical setup behind a disclosure, for the person who installed BonClay. */
function SetupDetails({ status, id }: { status: ExotelStatus; id: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-border bg-bg-base">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
        className="flex min-h-11 w-full items-center justify-between gap-3 rounded-xl px-4 py-2 text-left text-sm font-medium text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-primary"
      >
        Details for your IT person
        <ChevronDown className={`size-4 shrink-0 text-text-secondary transition-transform ${open ? "rotate-180" : ""}`} aria-hidden />
      </button>
      {open && (
        <div id={id} className="space-y-3 border-t border-border px-4 py-4 text-sm text-text-secondary">
          {!status.configured && (
            <div>
              <p>These settings are missing:</p>
              <ul aria-label="Missing settings" className="mt-1 list-disc space-y-1 pl-5 text-text-primary">
                {status.missing.map((m) => (
                  <li key={m}>
                    <code className={codeClass}>{m}</code>
                  </li>
                ))}
              </ul>
              <p className="mt-2 text-xs">Add them to backend/.env and restart BonClay, then open this again.</p>
            </div>
          )}
          {status.configured && !status.inbound_ready && (
            <p>
              Set <code className={codeClass}>PUBLIC_BASE_URL</code> in backend/.env to the https address Exotel can reach (for
              example an ngrok tunnel), then restart.
            </p>
          )}
          <div>
            <p className="font-medium text-text-primary">Connecting the number in Exotel</p>
            <ol className="mt-1 list-decimal space-y-1 pl-5">
              <li>In Exotel, open App Bazaar and open (or create) the flow for your ExoPhone.</li>
              <li>Add a Voicebot applet to the flow.</li>
              <li>Paste the agent link into it and press Save.</li>
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children?: ReactNode }) {
  return (
    <li className="flex gap-3">
      <span
        aria-hidden
        className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full bg-bg-elevated text-xs font-semibold text-text-primary"
      >
        {n}
      </span>
      <div className="min-w-0 flex-1 space-y-2">
        <p className="text-sm font-medium text-text-primary">{title}</p>
        {children}
      </div>
    </li>
  );
}

/** "Receive calls": whether the line is ready, and three plain steps to put the agent on the clinic's number. */
function ReceivePanel({
  status,
  link,
  busy,
  error,
  onGetLink,
}: {
  status: ExotelStatus;
  link: ExotelInboundLink | null;
  busy: boolean;
  error: string | null;
  onGetLink: () => void;
}) {
  const ready = status.configured && status.inbound_ready;
  return (
    <div className="space-y-4">
      <div
        data-testid="receive-ready"
        className={`flex items-start gap-3 rounded-xl border p-3 text-sm ${
          ready ? "border-accent-success/30 bg-accent-success/10" : "border-border bg-bg-base"
        }`}
      >
        <span aria-hidden className={`mt-0.5 shrink-0 ${ready ? "text-accent-success" : "text-text-secondary"}`}>
          {ready ? <CircleCheck className="size-4" /> : <TriangleAlert className="size-4" />}
        </span>
        <div>
          <p className="font-medium text-text-primary">{ready ? "Ready to connect your number" : "Not ready to receive calls yet"}</p>
          <p className="mt-0.5 text-text-secondary">
            {ready
              ? "Follow the three steps below. Then phone your number to check that your agent answers."
              : "Your phone line needs a one-time setup first. Ask the person who installed BonClay to finish it, or open Details below."}
          </p>
        </div>
      </div>

      <ol aria-label="Steps to receive calls" className="space-y-4">
        <Step n={1} title="Get your agent's phone link">
          {!link && (
            <Button variant="outline" size="sm" className="h-11" onClick={onGetLink} loading={busy} disabled={!ready}>
              Get the link
            </Button>
          )}
          {error && (
            <p role="alert" className="text-xs text-accent-error">
              {error}
            </p>
          )}
          {link && (
            <div className="space-y-2">
              <CopyField text={link.stream_url} label="agent phone link" />
              <p className="text-xs text-text-secondary">
                Keep this link private. It works for {link.valid_days} days; get a new one after that.
              </p>
            </div>
          )}
        </Step>
        <Step n={2} title="Add it to your phone line">
          <p className="text-sm text-text-secondary">
            In your phone provider&apos;s dashboard, open the call flow for your business number, add a voice agent step and paste
            the link there. Save it.
          </p>
        </Step>
        <Step n={3} title="Phone your number to check">
          <p className="text-sm text-text-secondary">
            Call your business number from another phone. Your agent should answer.
            {status.caller_id_masked && (
              <>
                {" "}
                Your number: <span className="font-mono text-text-primary">{status.caller_id_masked}</span>
              </>
            )}
          </p>
        </Step>
      </ol>

      <SetupDetails status={status} id="receive-details" />
    </div>
  );
}

interface Props {
  agent: Pick<Agent, "id" | "name"> | null;
  open: boolean;
  onClose: () => void;
}

type Tab = "receive" | "phone";
const TABS: { id: Tab; label: string; icon: ReactNode }[] = [
  { id: "receive", label: "Receive calls", icon: <PhoneIncoming className="size-4" aria-hidden /> },
  { id: "phone", label: "Phone someone", icon: <PhoneOutgoing className="size-4" aria-hidden /> },
];

/**
 * Owner phone dialog with two tabs. "Receive calls": the steps that put the agent on the clinic's number. "Phone someone": the agent
 * phones a saved number (or one the installer approved) once the person has agreed, with the call's live status.
 */
export function CallDialog({ agent, open, onClose }: Props) {
  const agentId = agent?.id ?? null;
  const [tab, setTab] = useState<Tab>("receive");
  const tabRefs = useRef<Record<Tab, HTMLButtonElement | null>>({ receive: null, phone: null });
  const [status, setStatus] = useState<ExotelStatus | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [number, setNumber] = useState("");   // the chosen number from the list
  const [adding, setAdding] = useState(false);
  const [addLabel, setAddLabel] = useState("");
  const [addNum, setAddNum] = useState("");
  const [addBusy, setAddBusy] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [removing, setRemoving] = useState(false);
  const [numbersNote, setNumbersNote] = useState<string | null>(null);
  const [numbersError, setNumbersError] = useState<string | null>(null);
  const [agreed, setAgreed] = useState(false);
  const [placing, setPlacing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [call, setCall] = useState<LiveCall | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [reload, setReload] = useState(0);
  const [link, setLink] = useState<ExotelInboundLink | null>(null);
  const [linkBusy, setLinkBusy] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);

  // Opening the dialog starts clean and reads the phone setup; closing it (or leaving) drops everything in flight.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setStatus(null);
    setLoadError(null);
    setError(null);
    setCall(null);
    setAgreed(false);
    setPlacing(false);
    setAdding(false);
    setAddLabel("");
    setAddNum("");
    setAddError(null);
    setConfirmRemove(false);
    setNumbersNote(null);
    setNumbersError(null);
    setLink(null);
    setLinkError(null);
    setTab("receive");
    api
      .get<ExotelStatus>("/api/phone/exotel/status")
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        setNumber(s.allowed_numbers?.[0]?.number ?? "");
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e instanceof Error && e.message ? e.message : "Could not check the phone line.");
      });
    return () => {
      cancelled = true;
    };
  }, [open, reload]);

  const sid = call?.sid ?? null;
  const ended = call ? call.state !== "calling" && TERMINAL.includes(call.state) : false;

  // Poll the call about every 2 s until it ends, the dialog closes, or the status stops arriving.
  useEffect(() => {
    if (!open || !sid || ended) return;
    let cancelled = false;
    let failures = 0;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const r = await api.get<ExotelCallStatus>(`/api/phone/exotel/call/${encodeURIComponent(sid)}`);
        if (cancelled) return;
        failures = 0;
        setCall((prev) =>
          prev && prev.sid === sid
            ? {
                ...prev,
                state: r.status === "unknown" && prev.state !== "calling" ? prev.state : r.status,
                durationS: r.duration_s ?? prev.durationS,
                message: r.message ?? prev.message,
                connectedAt: r.status === "in-progress" ? prev.connectedAt ?? Date.now() : prev.connectedAt,
              }
            : prev
        );
        if (TERMINAL.includes(r.status)) return;
      } catch (e) {
        if (cancelled) return;
        failures += 1;
        const gone = e instanceof ApiError && e.code === "CALL_NOT_FOUND";
        if (gone || failures >= MAX_POLL_FAILURES) {
          setError(`Lost track of the call. ${callErrorMessage(e)}`);
          return;
        }
      }
      timer = setTimeout(tick, POLL_MS);
    };
    timer = setTimeout(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [open, sid, ended]);

  // Consent is for one call: once it has ended the box is ticked again for the next one.
  useEffect(() => {
    if (ended) setAgreed(false);
  }, [ended]);

  // The running timer while the call is connected.
  const connected = call?.state === "in-progress";
  useEffect(() => {
    if (!open || !connected) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [open, connected]);

  const hasList = (status?.allowed_numbers.length ?? 0) > 0;
  const chosen = status?.allowed_numbers.find((n) => n.number === number) ?? null;
  const showAdd = adding || !hasList;
  // A number from backend/.env is pre-approved: the owner confirmed that person's consent once, when adding it there.
  const preapproved = !showAdd && chosen?.source === "env";
  const consentOk = agreed || preapproved;
  const labelOk = addLabel.trim().length >= 1 && addLabel.trim().length <= 40;
  const numOk = looksLikeNumber(addNum);

  /** Re-read the status after the list changed, keeping the rest of the dialog as it is. */
  const refreshStatus = useCallback(async (select?: string) => {
    const s = await api.get<ExotelStatus>("/api/phone/exotel/status");
    setStatus(s);
    setNumber((cur) => {
      const want = select ?? cur;
      return s.allowed_numbers.some((n) => n.number === want) ? want : s.allowed_numbers[0]?.number ?? "";
    });
  }, []);

  const closeAdd = () => {
    setAdding(false);
    setAddLabel("");
    setAddNum("");
    setAddError(null);
  };

  /** Save the new number. The one consent box is what the server needs as "agreed". */
  const saveNumber = useCallback(async () => {
    if (!labelOk) return setAddError("Give the number a name of 1 to 40 characters.");
    if (!numOk) return setAddError("Enter at least 10 digits, for example +91 98765 43210.");
    if (!agreed) return setAddError("Tick the box to confirm this person agreed to get a call from an AI agent.");
    setAddBusy(true);
    setAddError(null);
    try {
      const saved = await api.post<SavedPhoneNumber>("/api/phone/exotel/numbers", {
        label: addLabel.trim(),
        number: addNum.trim(),
        agreed,
      });
      closeAdd();
      setNumbersError(null);
      setNumbersNote(`Saved ${saved.label}.`);
      setNumber(saved.number);
      try {
        await refreshStatus(saved.number);
      } catch {
        setNumbersError("Saved, but the list could not be refreshed. Close and open this again.");
      }
    } catch (e) {
      setAddError(numberErrorMessage(e));
    } finally {
      setAddBusy(false);
    }
  }, [labelOk, numOk, agreed, addLabel, addNum, refreshStatus]);

  const remove = useCallback(async () => {
    if (!chosen?.id) return;
    setRemoving(true);
    setNumbersError(null);
    try {
      await api.del(`/api/phone/exotel/numbers/${encodeURIComponent(chosen.id)}`);
      setNumbersNote(`Removed ${chosen.label}.`);
      setConfirmRemove(false);
      await refreshStatus();
    } catch (e) {
      setNumbersError(numberErrorMessage(e));
      if (e instanceof ApiError && e.code === "NUMBER_NOT_FOUND") {
        setConfirmRemove(false);
        void refreshStatus().catch(() => {});
      }
    } finally {
      setRemoving(false);
    }
  }, [chosen, refreshStatus]);

  const place = useCallback(async () => {
    const to = number;
    if (!agentId || !to) return;
    setPlacing(true);
    setError(null);
    setCall(null);
    try {
      const r = await api.post<ExotelCallPlaced>("/api/phone/exotel/call", {
        agent_id: agentId,
        to,
        caller_agreed: consentOk,
      });
      if (!r.call_sid) {
        setError("The call was sent, but its progress can't be shown. Check your phone.");
      } else {
        setNow(Date.now());
        setCall({
          sid: r.call_sid,
          state: "calling",
          durationS: null,
          message: null,
          connectedAt: null,
          via: viaLabel(r.provider, r.account),
        });
      }
    } catch (e) {
      setError(callErrorMessage(e));
    } finally {
      setPlacing(false);
    }
  }, [agentId, number, consentOk]);

  const getLink = useCallback(async () => {
    if (!agentId) return;
    setLinkBusy(true);
    setLinkError(null);
    try {
      setLink(await api.post<ExotelInboundLink>("/api/phone/exotel/inbound-link", { agent_id: agentId }));
    } catch (e) {
      setLinkError(callErrorMessage(e));
    } finally {
      setLinkBusy(false);
    }
  }, [agentId]);

  if (!agent) return null;

  const inCall = !!call && !ended;
  const canCall = !!status?.configured;
  const locked = placing || inCall || addBusy;

  const onTabKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    const i = TABS.findIndex((t) => t.id === tab);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    if (next < 0) return;
    e.preventDefault();
    setTab(TABS[next].id);
    tabRefs.current[TABS[next].id]?.focus();
  };

  let liveLine: ReactNode = null;
  if (call) {
    const elapsed = call.connectedAt ? Math.max(0, (now - call.connectedAt) / 1000) : 0;
    const msg = call.message ? `: ${maskDigits(call.message)}` : "";
    switch (call.state) {
      case "calling":
        liveLine = (
          <Line tone="neutral" icon={<Loader2 className="size-4 animate-spin" />}>
            Calling…
          </Line>
        );
        break;
      case "queued":
        liveLine = (
          <Line tone="neutral" icon={<Clock className="size-4" />}>
            Queued
          </Line>
        );
        break;
      case "ringing":
        liveLine = (
          <Line tone="neutral" icon={<PhoneIncoming className="size-4" />}>
            Ringing
          </Line>
        );
        break;
      case "in-progress":
        liveLine = (
          <Line tone="good" icon={<PhoneCall className="size-4" />}>
            Connected <span className="font-mono tabular-nums">{clock(elapsed)}</span>
          </Line>
        );
        break;
      case "completed": {
        const secs = call.durationS ?? (call.connectedAt ? elapsed : null);
        liveLine = (
          <Line tone="neutral" icon={<PhoneOff className="size-4" />}>
            Ended{secs != null ? ` (${clock(secs)})` : ""}
          </Line>
        );
        break;
      }
      case "busy":
        liveLine = (
          <Line tone="bad" icon={<PhoneMissed className="size-4" />}>
            Busy
            {msg}
          </Line>
        );
        break;
      case "no-answer":
        liveLine = (
          <Line tone="bad" icon={<PhoneMissed className="size-4" />}>
            No answer
            {msg}
          </Line>
        );
        break;
      case "canceled":
        liveLine = (
          <Line tone="bad" icon={<PhoneOff className="size-4" />}>
            Canceled
            {msg}
          </Line>
        );
        break;
      case "failed":
        liveLine = (
          <Line tone="bad" icon={<TriangleAlert className="size-4" />}>
            Failed
            {msg}
          </Line>
        );
        break;
      default:
        liveLine = (
          <Line tone="neutral" icon={<Loader2 className="size-4 animate-spin" />}>
            Checking the call…
          </Line>
        );
    }
  }

  // Why the main button is off, in one line.
  let hint: string | null = null;
  if (!inCall) {
    if (showAdd && (!labelOk || !numOk)) hint = "Enter a name and a phone number.";
    else if (!showAdd && !chosen) hint = "Choose who to call.";
    else if (!consentOk) hint = "Tick the box above to confirm consent.";
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      size="lg"
      title={`Phone line: ${agent.name}`}
      description="Let people phone this agent, or have it phone someone who agreed."
    >
      <div className="space-y-5">
        {!status && !loadError && (
          <p role="status" className="flex items-center gap-2 text-sm text-text-secondary">
            <Loader2 className="size-4 animate-spin" aria-hidden /> Checking your phone line…
          </p>
        )}

        {loadError && (
          <div role="alert" className="rounded-xl border border-accent-error/30 bg-accent-error/10 p-3 text-sm text-text-primary">
            {loadError}
            <div className="mt-2">
              <Button variant="outline" size="sm" className="h-11" onClick={() => setReload((n) => n + 1)}>
                Try again
              </Button>
            </div>
          </div>
        )}

        {status && (
          <p
            data-testid="line-status"
            className={`flex items-center gap-2 text-sm font-medium ${status.configured ? "text-accent-success" : "text-text-primary"}`}
          >
            <span aria-hidden className="grid size-5 place-items-center">
              {status.configured ? <CircleCheck className="size-4" /> : <TriangleAlert className="size-4 text-text-secondary" />}
            </span>
            {status.configured ? "Phone line connected" : "Phone line not connected yet"}
          </p>
        )}

        {status && (
          <div role="tablist" aria-label="Phone options" className="grid grid-cols-2 gap-1 rounded-xl bg-bg-elevated p-1">
            {TABS.map((t) => {
              const on = tab === t.id;
              return (
                <button
                  key={t.id}
                  ref={(el) => {
                    tabRefs.current[t.id] = el;
                  }}
                  type="button"
                  role="tab"
                  id={`phone-tab-${t.id}`}
                  aria-selected={on}
                  aria-controls={`phone-panel-${t.id}`}
                  tabIndex={on ? 0 : -1}
                  onClick={() => setTab(t.id)}
                  onKeyDown={onTabKey}
                  className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-2 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-primary ${
                    on ? "bg-bg-surface text-text-primary shadow-sm" : "text-text-secondary hover:text-text-primary"
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              );
            })}
          </div>
        )}

        {status && tab === "receive" && (
          <div role="tabpanel" id="phone-panel-receive" aria-labelledby="phone-tab-receive">
            <ReceivePanel status={status} link={link} busy={linkBusy} error={linkError} onGetLink={() => void getLink()} />
          </div>
        )}

        {status && tab === "phone" && !canCall && (
          <div role="tabpanel" id="phone-panel-phone" aria-labelledby="phone-tab-phone" className="space-y-4">
            <div className="rounded-xl border border-border bg-bg-base p-4 text-sm">
              <p className="flex items-center gap-2 font-medium text-text-primary">
                <Phone className="size-4 text-text-secondary" aria-hidden /> Your phone line is not connected yet
              </p>
              <p className="mt-2 text-text-secondary">
                Your agent can&apos;t phone anyone until the line is connected. Ask the person who installed BonClay to finish the
                phone setup, or open Details below.
              </p>
            </div>
            <SetupDetails status={status} id="phone-details" />
          </div>
        )}

        {status && tab === "phone" && canCall && (
          <div role="tabpanel" id="phone-panel-phone" aria-labelledby="phone-tab-phone">
            <form
              id="call-form"
              className="space-y-4"
              onSubmit={(e) => {
                e.preventDefault();
                if (!consentOk || locked) return;
                if (showAdd) void saveNumber();
                else void place();
              }}
            >
              {hasList && (
                <div>
                  <label htmlFor="call-number" className="mb-1.5 block text-xs font-medium text-text-secondary">
                    Number to call
                  </label>
                  <div className="flex gap-2">
                    <select
                      id="call-number"
                      value={number}
                      onChange={(e) => {
                        setNumber(e.target.value);
                        setConfirmRemove(false);
                        setAgreed(false);   // consent is for this person, not the next one
                      }}
                      disabled={locked}
                      className={fieldClass}
                    >
                      {status.allowed_numbers.map((n) => (
                        <option key={n.number} value={n.number}>
                          {displayLabel(n) ? `${displayLabel(n)}, ${n.masked}` : n.masked}
                        </option>
                      ))}
                    </select>
                    {chosen?.source === "saved" && chosen.id && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-11 w-11 shrink-0 px-0"
                        aria-label={`Remove ${chosen.label}`}
                        onClick={() => setConfirmRemove(true)}
                        disabled={locked || removing}
                      >
                        <Trash2 className="size-4" aria-hidden />
                      </Button>
                    )}
                  </div>
                  {chosen?.source === "env" && (
                    <p className="mt-1 text-xs text-text-secondary">
                      Pre-approved in backend/.env (PHONE_ALLOWED_NUMBERS): no tick needed. Change it there.
                    </p>
                  )}
                  {confirmRemove && chosen?.id && (
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-sm text-text-primary">
                      <span>Remove {chosen.label}?</span>
                      <Button type="button" size="sm" className="h-10" variant="outline" onClick={() => void remove()} loading={removing}>
                        Remove
                      </Button>
                      <Button type="button" size="sm" className="h-10" variant="ghost" onClick={() => setConfirmRemove(false)} disabled={removing}>
                        Keep
                      </Button>
                    </div>
                  )}
                  {!adding && (
                    <button
                      type="button"
                      aria-expanded={false}
                      aria-controls="add-number-panel"
                      onClick={() => {
                        setAdding(true);
                        setNumbersNote(null);
                      }}
                      disabled={locked}
                      className="mt-1 inline-flex min-h-11 items-center gap-1 text-sm font-medium text-accent-primary hover:underline disabled:opacity-50"
                    >
                      <Plus className="size-4" aria-hidden /> Add a number
                    </button>
                  )}
                </div>
              )}

              {showAdd && (
                <div id="add-number-panel" className="space-y-3 rounded-xl border border-border bg-bg-base p-4">
                  {!hasList && (
                    <p className="text-sm text-text-secondary">Add the number you want to call. Only saved numbers can be called.</p>
                  )}
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div>
                      <label htmlFor="add-number-label" className="mb-1.5 block text-xs font-medium text-text-secondary">
                        Name
                      </label>
                      <input
                        id="add-number-label"
                        maxLength={40}
                        placeholder="Arnav"
                        value={addLabel}
                        onChange={(e) => setAddLabel(e.target.value)}
                        disabled={locked}
                        className={fieldClass}
                      />
                    </div>
                    <div>
                      <label htmlFor="add-number-phone" className="mb-1.5 block text-xs font-medium text-text-secondary">
                        Phone number
                      </label>
                      <input
                        id="add-number-phone"
                        type="tel"
                        inputMode="tel"
                        autoComplete="off"
                        placeholder="+91 98765 43210"
                        value={addNum}
                        onChange={(e) => setAddNum(e.target.value)}
                        disabled={locked}
                        className={fieldClass}
                      />
                    </div>
                  </div>
                  {addError && (
                    <p role="alert" className="text-xs text-accent-error">
                      {addError}
                    </p>
                  )}
                  {hasList && (
                    <Button type="button" size="sm" variant="ghost" className="h-10" onClick={closeAdd} disabled={addBusy}>
                      Cancel
                    </Button>
                  )}
                </div>
              )}

              <div aria-live="polite" aria-atomic="true" data-testid="numbers-status" className="text-xs">
                {numbersNote && <p className="text-accent-success">{numbersNote}</p>}
              </div>
              {numbersError && (
                <p role="alert" className="text-xs text-accent-error">
                  {numbersError}
                </p>
              )}

              <label className="flex min-h-11 items-start gap-3 rounded-xl border border-border bg-bg-base p-3 text-sm text-text-primary">
                <input
                  type="checkbox"
                  checked={consentOk}
                  onChange={(e) => setAgreed(e.target.checked)}
                  disabled={locked || preapproved}
                  className="mt-0.5 size-5 shrink-0 accent-[var(--color-accent)]"
                />
                <span>This person has agreed to get a call from an AI agent</span>
              </label>
              {preapproved && (
                <p className="-mt-2 text-xs text-text-secondary">Pre-approved: confirmed when the number was added to backend/.env.</p>
              )}
            </form>
          </div>
        )}

        {/* Stays in view at the bottom while the content above scrolls (phone sheets). */}
        {status && tab === "phone" && canCall && (
          <div
            data-testid="call-footer"
            className="sticky bottom-[-1.25rem] z-10 -mx-5 -mb-5 space-y-2 border-t border-border bg-[hsl(var(--au-surface))] px-5 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 sm:-mx-6 sm:px-6"
          >
            <div aria-live="polite" aria-atomic="true" data-testid="call-status">
              {liveLine}
              {call?.via && (
                <p data-testid="call-via" className="mt-0.5 text-xs text-text-secondary">
                  via {call.via}
                </p>
              )}
            </div>
            {error && (
              <div role="alert" className="rounded-xl border border-accent-error/30 bg-accent-error/10 p-3 text-sm text-text-primary">
                {error}
              </div>
            )}
            {hint && !addBusy && !placing && (
              <p id="call-hint" className="text-xs text-text-secondary">
                {hint}
              </p>
            )}
            <Button
              type="submit"
              form="call-form"
              size="lg"
              className="h-12 w-full sm:w-auto"
              loading={placing || addBusy}
              disabled={!consentOk || inCall || (showAdd ? !labelOk || !numOk : !chosen)}
              aria-describedby={hint ? "call-hint" : undefined}
              leftIcon={showAdd ? undefined : <Phone className="size-4" />}
            >
              {showAdd ? "Save number" : ended ? "Call again" : "Call"}
            </Button>
          </div>
        )}
      </div>
    </Modal>
  );
}
