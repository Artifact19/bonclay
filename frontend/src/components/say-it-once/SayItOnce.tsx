"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { AudioLines, Check } from "lucide-react";
import { Button, Card, EmptyState, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useVoices } from "@/hooks/useApi";
import { toView } from "@/components/voices/voiceApi";
import { cn } from "@/lib/utils";
import { LanguageCard } from "./LanguageCard";
import { ReviewList } from "./ReviewList";
import { ProgressBar, formatDuration, formatEta } from "./progress";
import type { Capability, Estimate, ReviewItem, SayItem, SayJob, Speed } from "./types";

export const DEFAULT_SENTENCE = "Your appointment is confirmed for Thursday at four.";
export const MAX_CHARS = 200;
export const MAX_LANGUAGES = 4;

/** Shown first; the rest sit behind "More languages". Names are used until the server's list arrives. */
const MAIN: [string, string][] = [
  ["hi", "Hindi"], ["mr", "Marathi"], ["ta", "Tamil"], ["bn", "Bengali"], ["te", "Telugu"],
  ["gu", "Gujarati"], ["en", "English"], ["fr", "French"], ["es", "Spanish"],
];
const MAIN_CODES = new Set(MAIN.map(([c]) => c));

function errText(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  return e instanceof Error ? e.message : "Something went wrong. Try again.";
}

/** How a language would be spoken at the chosen speed. */
function routeOf(c: Capability | undefined, speed: Speed) {
  if (!c) return undefined;
  return speed === "fast" ? c.fast ?? { ...c, available: false } : c;
}

function kindLabel(c: Capability | undefined, speed: Speed): string | null {
  const r = routeOf(c, speed);
  if (!r) return null;
  if (!r.available) return "not available";
  return r.voice_kind === "your_voice" ? "your voice" : "preset voice";
}

const SPEEDS: { value: Speed; label: string; hint: string }[] = [
  { value: "own", label: "Your voice (slower)", hint: "Your own cloned voice where an engine can speak the language; other languages use preset voices." },
  { value: "fast", label: "Fast (preset voices)", hint: "Preset voices only: the quickest, and the lightest on this laptop." },
];

export function SayItOnce({ pollMs = 1000, estimateDebounceMs = 350 }: { pollMs?: number; estimateDebounceMs?: number }) {
  const voices = useVoices();
  const consented = useMemo(() => (voices.data ?? []).map(toView).filter((v) => v.consent), [voices.data]);

  const [voiceId, setVoiceId] = useState<string>("");
  const [text, setText] = useState(DEFAULT_SENTENCE);
  const [picked, setPicked] = useState<string[]>(["hi", "fr"]);
  const [more, setMore] = useState(false);
  const [speed, setSpeed] = useState<Speed>("own");
  const [reviewFirst, setReviewFirst] = useState(false);
  const [caps, setCaps] = useState<Record<string, Capability>>({});
  const [capsError, setCapsError] = useState<string | null>(null);
  const [estimate, setEstimate] = useState<Estimate | null>(null);

  // The running (or finished) job, and the single-language re-speaks that replace one card of it.
  const [job, setJob] = useState<SayJob | null>(null);
  const [overrides, setOverrides] = useState<Record<string, SayItem>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [cardEdits, setCardEdits] = useState<Record<string, string>>({});
  // What the last run was made from, so "Re-speak" and "Retry" use the same sentence, voice and speed.
  const ran = useRef<{ voiceId: string; text: string; speed: Speed } | null>(null);

  // The review step: translations the user can read and edit before anything is spoken.
  const [translating, setTranslating] = useState(false);
  const [review, setReview] = useState<ReviewItem[] | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const activeVoice = voiceId && consented.some((v) => v.id === voiceId) ? voiceId : consented[0]?.id ?? "";
  const activeLanguage = consented.find((v) => v.id === activeVoice)?.language ?? "";

  // What each language will sound like for the chosen voice.
  useEffect(() => {
    if (!activeVoice) return;
    let cancelled = false;
    api
      .get<{ languages: Capability[] }>(`/api/say-it-once/capabilities?voice_id=${encodeURIComponent(activeVoice)}`, { retries: 0 })
      .then((r) => {
        if (cancelled) return;
        setCaps(Object.fromEntries((r?.languages ?? []).map((c) => [c.language, c])));
        setCapsError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setCaps({});
        setCapsError(errText(e));
      });
    return () => {
      cancelled = true;
    };
  }, [activeVoice]);

  const chips = useMemo(() => {
    const all = Object.values(caps);
    if (!all.length) return MAIN.map(([language, language_name]) => ({ language, language_name }) as Capability);
    const main = MAIN.map(([c]) => caps[c]).filter(Boolean);
    return more ? [...main, ...all.filter((c) => !MAIN_CODES.has(c.language))] : main;
  }, [caps, more]);

  // A language the chosen speed cannot speak is left out, however it got picked.
  const chosen = useMemo(
    () => picked.filter((c) => routeOf(caps[c], speed)?.available !== false),
    [picked, caps, speed]
  );

  const toggle = (code: string) =>
    setPicked((p) => (p.includes(code) ? p.filter((c) => c !== code) : p.length >= MAX_LANGUAGES ? p : [...p, code]));

  const running = job?.state === "running";
  const working = busy || translating;
  const locked = working || !!review;
  const canRun =
    !working && !!activeVoice && chosen.length > 0 && text.trim().length > 0 && text.length <= MAX_CHARS;

  // The upfront estimate: how long this sentence takes in each chosen language on this laptop.
  useEffect(() => {
    if (!activeVoice || chosen.length === 0 || !text.trim() || locked) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      api
        .post<Estimate>(
          "/api/say-it-once/estimate",
          { voice_id: activeVoice, text: text.trim(), languages: chosen, source_language: "en", speed },
          { retries: 0 }
        )
        .then((r) => !cancelled && setEstimate(r))
        .catch(() => !cancelled && setEstimate(null));
    }, estimateDebounceMs);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [activeVoice, chosen, text, speed, locked, estimateDebounceMs]);

  const poll = useCallback(
    async (id: string, onSnap: (out: SayJob) => void) => {
      for (;;) {
        const out = await api.get<SayJob>(`/api/say-it-once/${encodeURIComponent(id)}`, { retries: 2 });
        if (!alive.current) return;
        onSnap(out);
        if (out.state !== "running") return;
        await new Promise((r) => setTimeout(r, pollMs));
        if (!alive.current) return;
      }
    },
    [pollMs]
  );

  /** Say it: translate and speak every language, one after another. `approved` carries reviewed translations. */
  const run = async (approved?: Record<string, string>) => {
    setBusy(true);
    setError(null);
    setJob(null);
    setOverrides({});
    setCardEdits({});
    const languages = approved && review ? review.map((r) => r.language) : chosen;
    const sentence = text.trim();
    try {
      const started = await api.post<{ job_id: string; items: { language: string; status: string }[] }>(
        "/api/say-it-once",
        {
          voice_id: activeVoice, text: sentence, languages, source_language: "en", speed,
          ...(approved ? { translations: approved } : {}),
        },
        { retries: 0 }
      );
      ran.current = { voiceId: activeVoice, text: sentence, speed };
      setJobId(started.job_id);
      setJob({
        state: "running", speed, progress: 0, done: 0, failed: 0, total: started.items.length,
        items: started.items.map((i, n) => {
          const route = routeOf(caps[i.language], speed);
          return {
            language: i.language, language_name: caps[i.language]?.language_name ?? i.language.toUpperCase(),
            status: "queued", stage: "waiting", progress: 0, queue_position: n + 1, eta_s: null,
            text: approved?.[i.language] ?? null, voice_kind: route?.voice_kind ?? null,
            engine: route?.engine ?? null, engine_name: route?.engine_name ?? null,
            attribution: route?.attribution ?? null, audio_url: null, receipt_id: null, seconds: null,
          };
        }),
      });
      await poll(started.job_id, (out) => setJob(out));
    } catch (e) {
      if (alive.current) {
        setError(errText(e));
        setJob(null);
      }
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  /** Re-speak (or retry) one language, from the text on its card, without translating again. */
  const respeak = async (language: string) => {
    const base = (job?.items ?? []).map((i) => overrides[i.language] ?? i).find((i) => i.language === language);
    const basis = ran.current;
    if (!base || !basis) return;
    const value = (cardEdits[language] ?? base.text ?? "").trim();
    setBusy(true);
    setError(null);
    setOverrides((o) => ({
      ...o,
      [language]: {
        ...base, status: "queued", stage: "waiting", progress: 0, queue_position: 1, eta_s: null, error: null,
        audio_url: null, receipt_id: null, seconds: null, text: value || base.text,
      },
    }));
    try {
      const started = await api.post<{ job_id: string }>(
        "/api/say-it-once",
        {
          voice_id: basis.voiceId, text: basis.text, languages: [language], source_language: "en", speed: basis.speed,
          ...(value ? { translations: { [language]: value } } : {}),
        },
        { retries: 0 }
      );
      await poll(started.job_id, (out) => {
        const next = out.items.find((i) => i.language === language);
        if (next) setOverrides((o) => ({ ...o, [language]: next }));
        if (out.state !== "running") setCardEdits((c) => Object.fromEntries(Object.entries(c).filter(([k]) => k !== language)));
      });
    } catch (e) {
      if (alive.current) {
        setOverrides((o) => ({ ...o, [language]: { ...base, status: "failed", stage: "failed", progress: base.progress ?? 0, error: errText(e) } }));
      }
    } finally {
      if (alive.current) setBusy(false);
    }
  };

  const cancel = async () => {
    if (!jobId) return;
    try {
      const out = await api.post<SayJob>(`/api/say-it-once/${encodeURIComponent(jobId)}/cancel`, {}, { retries: 0 });
      if (alive.current && out?.items) setJob(out);
    } catch (e) {
      if (alive.current) setError(errText(e));
    }
  };

  const translate = async () => {
    setTranslating(true);
    setError(null);
    setJob(null);
    try {
      const out = await api.post<{ items: ReviewItem[] }>(
        "/api/say-it-once/translate",
        { voice_id: activeVoice, text: text.trim(), languages: chosen, source_language: "en", speed },
        { retries: 0 }
      );
      if (!alive.current) return;
      setReview(out.items);
      setEdits({});
    } catch (e) {
      if (alive.current) setError(errText(e));
    } finally {
      if (alive.current) setTranslating(false);
    }
  };

  const backToEdit = () => {
    setReview(null);
    setEdits({});
    setError(null);
  };

  const approved = useMemo(() => {
    const out: Record<string, string> = {};
    for (const r of review ?? []) {
      if (!r.available) continue;
      const v = (edits[r.language] ?? r.translation).trim();
      if (v) out[r.language] = v;
    }
    return out;
  }, [review, edits]);
  const missing = (review ?? []).filter((r) => r.available && !approved[r.language]);
  const canSpeak = !busy && !!review && Object.keys(approved).length > 0 && missing.length === 0;

  const cards = (job?.items ?? []).map((i) => overrides[i.language] ?? i);
  const showEstimate = !locked && !!estimate && estimate.items.some((i) => i.available);

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-6 md:py-8">
      <PageHeader
        title="Say it once, hear it everywhere"
        description="Type one sentence. BonClay translates it and speaks it in up to four languages, one after another, with each language's text and voice together."
      />

      {voices.isLoading ? (
        <Skeleton className="h-64 rounded-2xl" />
      ) : voices.error ? (
        <ErrorState message={voices.error.message} onRetry={() => void voices.mutate()} />
      ) : consented.length === 0 ? (
        <EmptyState
          icon={<AudioLines />}
          title="You need a voice with a recorded consent"
          description="Say it once speaks in your own voice, so it only uses voices you cloned with the consent step."
          action={
            <Link href="/voices/new" className="inline-flex h-10 items-center rounded-full bg-brand px-5 text-sm font-medium text-[var(--color-on-accent)]">
              Clone a voice
            </Link>
          }
        />
      ) : (
        <div className="space-y-6">
          <Card variant="solid" className="space-y-5">
            <div>
              <label htmlFor="sio-voice" className="mb-1.5 block text-sm font-medium text-text-primary">
                Voice
              </label>
              <select
                id="sio-voice"
                value={activeVoice}
                onChange={(e) => setVoiceId(e.target.value)}
                disabled={locked}
                className="h-11 w-full rounded-xl border border-border bg-bg-base px-3 text-sm text-text-primary sm:w-80"
              >
                {consented.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name}
                  </option>
                ))}
              </select>
              {speed === "own" && activeLanguage === "en" && chosen.some((c) => c === "mr") && (
                <p data-testid="marathi-voice-note" className="mt-2 max-w-xl text-xs text-text-secondary">
                  For clear Marathi, use a voice recorded in Marathi: this one was recorded in English, so its Marathi can
                  sound unclear. <Link href="/voices/new" className="underline">Record a voice in Marathi</Link> (pick the मराठी
                  script).
                </p>
              )}
            </div>

            <div>
              <label htmlFor="sio-text" className="mb-1.5 block text-sm font-medium text-text-primary">
                Sentence
              </label>
              <textarea
                id="sio-text"
                value={text}
                maxLength={MAX_CHARS}
                rows={2}
                onChange={(e) => setText(e.target.value)}
                disabled={locked}
                className="w-full resize-y rounded-xl border border-border bg-bg-base px-3 py-2.5 text-[15px] text-text-primary"
              />
              <p className="mt-1 text-right text-xs tabular-nums text-text-secondary">
                {text.length} / {MAX_CHARS}
              </p>
            </div>

            <fieldset>
              <legend className="mb-1.5 text-sm font-medium text-text-primary">Speed</legend>
              <div className="grid gap-2 sm:grid-cols-2">
                {SPEEDS.map((s) => (
                  <label
                    key={s.value}
                    className={cn(
                      "flex cursor-pointer items-start gap-3 rounded-xl border px-3.5 py-3 text-sm transition-colors",
                      speed === s.value ? "border-accent-primary bg-accent-primary/10" : "border-border hover:border-text-primary",
                      locked && "cursor-not-allowed opacity-60"
                    )}
                  >
                    <input
                      type="radio"
                      name="sio-speed"
                      value={s.value}
                      checked={speed === s.value}
                      disabled={locked}
                      onChange={() => setSpeed(s.value)}
                      className="mt-1 accent-[var(--color-accent-primary)]"
                    />
                    <span>
                      <span className="block font-medium text-text-primary">{s.label}</span>
                      <span className="block text-xs text-text-secondary">{s.hint}</span>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            <fieldset>
              <legend className="mb-1.5 text-sm font-medium text-text-primary">
                Languages <span className="font-normal text-text-secondary">(up to {MAX_LANGUAGES})</span>
              </legend>
              <div className="flex flex-wrap gap-2">
                {chips.map((c) => {
                  const on = picked.includes(c.language);
                  const route = routeOf(caps[c.language], speed);
                  const unavailable = route ? !route.available : false;
                  const label = kindLabel(caps[c.language], speed);
                  return (
                    <button
                      key={c.language}
                      type="button"
                      aria-pressed={on && !unavailable}
                      disabled={locked || (!on && picked.length >= MAX_LANGUAGES) || unavailable}
                      onClick={() => toggle(c.language)}
                      title={route?.reason ?? undefined}
                      className={cn(
                        "inline-flex min-h-10 items-center gap-2 rounded-full border px-3.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50",
                        on && !unavailable
                          ? "border-accent-primary bg-accent-primary/15 text-text-primary"
                          : "border-border text-text-secondary hover:border-text-primary hover:text-text-primary"
                      )}
                    >
                      {on && !unavailable && <Check className="size-3.5 text-accent-secondary" aria-hidden />}
                      <span className="font-medium">{c.language_name}</span>
                      {label && (
                        <span
                          data-testid={`chip-kind-${c.language}`}
                          className={cn("text-xs", label === "your voice" ? "text-accent-secondary" : "text-text-secondary")}
                        >
                          {label}
                        </span>
                      )}
                    </button>
                  );
                })}
                {Object.keys(caps).length > MAIN.length && (
                  <button
                    type="button"
                    onClick={() => setMore((m) => !m)}
                    className="min-h-10 rounded-full px-3 text-sm font-medium text-accent-secondary hover:underline"
                  >
                    {more ? "Fewer languages" : "More languages"}
                  </button>
                )}
              </div>
              {capsError && <p className="mt-2 text-xs text-accent-error">Could not check which voice each language gets: {capsError}</p>}
            </fieldset>

            {showEstimate && (
              <div data-testid="estimate" className="rounded-xl bg-bg-elevated px-4 py-3 text-sm text-text-secondary">
                <ul className="space-y-0.5">
                  {estimate.items
                    .filter((i) => i.available)
                    .map((i) => (
                      <li key={i.language}>
                        {i.voice_kind === "your_voice" ? "Your voice" : "A preset voice"} in {i.language_name} takes{" "}
                        <span className="font-medium text-text-primary">{formatDuration(i.est_s)}</span> on this laptop.
                      </li>
                    ))}
                </ul>
                {estimate.items.filter((i) => i.available).length > 1 && (
                  <p className="mt-1.5 font-medium text-text-primary">
                    All together {formatDuration(estimate.total_s)}, one language at a time.
                  </p>
                )}
              </div>
            )}

            <div>
              {review ? (
                <div className="flex flex-wrap items-center gap-3">
                  <Button size="lg" onClick={() => void run(approved)} loading={busy} disabled={!canSpeak}>
                    Speak all
                  </Button>
                  <Button size="lg" variant="outline" onClick={backToEdit} disabled={busy}>
                    Back to edit
                  </Button>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    size="lg"
                    onClick={() => void (reviewFirst ? translate() : run())}
                    loading={working}
                    disabled={!canRun}
                  >
                    Say it
                  </Button>
                  {running && (
                    <Button size="lg" variant="outline" onClick={() => void cancel()} disabled={!!job?.cancelling}>
                      Cancel
                    </Button>
                  )}
                  <label className="inline-flex min-h-10 cursor-pointer items-center gap-2 text-sm text-text-secondary">
                    <input
                      type="checkbox"
                      checked={reviewFirst}
                      disabled={locked}
                      onChange={(e) => setReviewFirst(e.target.checked)}
                      className="size-4 accent-[var(--color-accent-primary)]"
                    />
                    Review translations before speaking
                  </label>
                </div>
              )}
              {reviewFirst && !review && (
                <p className="mt-2 max-w-[68ch] text-xs text-text-secondary">
                  Say it will translate first, show you each translation to read and fix, then speak when you say so.
                </p>
              )}
              {error && (
                <p role="alert" className="mt-3 text-sm text-accent-error">
                  {error}
                </p>
              )}
            </div>
          </Card>

          {review && (
            <ReviewList
              items={review}
              edits={edits}
              disabled={busy}
              onEdit={(language, value) => setEdits((e) => ({ ...e, [language]: value }))}
            />
          )}

          {job && <OverallProgress job={job} />}

          {job && (
            <section aria-label="Results" className="grid grid-cols-1 gap-4 md:grid-cols-2">
              {cards.map((item) => (
                <LanguageCard
                  key={item.language}
                  item={item}
                  text={cardEdits[item.language] ?? item.text ?? ""}
                  onTextChange={(v) => setCardEdits((c) => ({ ...c, [item.language]: v }))}
                  onRespeak={() => void respeak(item.language)}
                  canAct={!working}
                />
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  );
}

/** The whole job at a glance: a bar with a percentage, how many languages are done and an honest time left. */
function OverallProgress({ job }: { job: SayJob }) {
  const total = job.total ?? job.items.length;
  const done = job.done ?? job.items.filter((i) => i.status === "done").length;
  const progress = job.progress ?? 0;
  const eta = job.state === "running" && !job.slow ? formatEta(job.eta_s) : null;
  const failed = job.failed ?? 0;
  const headline =
    job.state === "running"
      ? job.cancelling
        ? "Stopping after the language being spoken"
        : "Working on it"
      : job.state === "cancelled"
        ? "Stopped"
        : job.state === "failed"
          ? "Something went wrong"
          : "All done";
  return (
    <Card variant="solid" data-testid="overall" className="space-y-2.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="text-sm font-medium text-text-primary" role="status">
          {headline}
        </p>
        <p className="text-sm tabular-nums text-text-secondary">
          <span data-testid="overall-pct" className="font-display text-xl font-medium text-text-primary">
            {Math.round(progress)}%
          </span>
        </p>
      </div>
      <ProgressBar value={progress} label="Overall progress" testId="overall-progress" className="h-2.5" />
      <p className="flex flex-wrap gap-x-4 text-sm text-text-secondary" data-testid="overall-detail">
        <span>
          {done} of {total} {total === 1 ? "language" : "languages"} done
          {failed > 0 ? `, ${failed} failed` : ""}
        </span>
        {eta && <span>{eta}</span>}
        {job.state === "running" && job.slow && <span>Taking longer than estimated</span>}
        {job.state !== "running" && job.elapsed_s ? <span>Took {Math.round(job.elapsed_s)} s</span> : null}
      </p>
    </Card>
  );
}

export default SayItOnce;
