"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Send, Square, Volume2 } from "lucide-react";
import { toast } from "sonner";
import { api, ApiError } from "@/lib/api";
import { Button, Skeleton, ErrorState } from "@/components/ui";
import { useKokoroVoices, useVoices } from "@/hooks/useApi";
import WaveformPlayer from "@/components/audio/WaveformPlayer";

interface Source {
  index: number;
  score: number;
  text: string;
}

interface AskError {
  error: string;
  code: string;
  details: unknown;
}

const TTS_LIMIT = 2000;

/** Strip markdown noise and cap at the server's 2000 char limit, ending on a sentence boundary. */
function plainSpeech(raw: string): { text: string; truncated: boolean } {
  const text = raw.replace(/[*_`#>]+/g, "").replace(/\s+/g, " ").trim();
  if (text.length <= TTS_LIMIT) return { text, truncated: false };
  const head = text.slice(0, TTS_LIMIT);
  let cut = -1;
  const re = /[.!?]["')\]]*(?=\s|$)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(head))) cut = m.index + m[0].length;
  if (cut < TTS_LIMIT / 4) cut = head.lastIndexOf(" ");
  return { text: head.slice(0, cut > 0 ? cut : TTS_LIMIT).trim(), truncated: true };
}

type Status = "idle" | "retrieving" | "streaming";

export default function AskPanel({ docId, docName }: { docId: string | null; docName?: string }) {
  const { data: kokoro, isLoading: kLoading } = useKokoroVoices();
  const { data: cloned, isLoading: cLoading } = useVoices();

  const [question, setQuestion] = useState("");
  const [voice, setVoice] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [answer, setAnswer] = useState("");
  const [asked, setAsked] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [showSources, setShowSources] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [audio, setAudio] = useState<Blob | null>(null);
  const [ttsError, setTtsError] = useState<string | null>(null);
  const [shortened, setShortened] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  const answerRef = useRef<HTMLDivElement>(null);

  const kokoroList = useMemo(() => (Array.isArray(kokoro) ? kokoro : []), [kokoro]);
  const clonedList = useMemo(() => (Array.isArray(cloned) ? cloned : []), [cloned]);

  useEffect(() => {
    if (!voice && kokoroList.length) setVoice(`kokoro:${kokoroList[0].id}`);
  }, [voice, kokoroList]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
      ttsAbortRef.current?.abort();
    },
    []
  );

  // New document selected: drop the previous conversation.
  useEffect(() => {
    abortRef.current?.abort();
    ttsAbortRef.current?.abort();
    setStatus("idle");
    setAnswer("");
    setAsked("");
    setSources([]);
    setError(null);
    setAudio(null);
    setTtsError(null);
    setShortened(false);
    setShowSources(false);
  }, [docId]);

  const busy = status !== "idle";
  const canAsk = !!docId && question.trim().length > 1 && !busy;

  async function ask() {
    if (!docId || !canAsk) return;
    const q = question.trim();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    ttsAbortRef.current?.abort();
    setStatus("retrieving");
    setAnswer("");
    setAsked(q);
    setSources([]);
    setShowSources(false);
    setError(null);
    setAudio(null);
    setTtsError(null);
    setShortened(false);

    let acc = "";
    let streamError: string | null = null;
    const body: Record<string, unknown> = { doc_id: docId, question: q };
    if (voice.startsWith("clone:")) body.voice_id = voice.slice(6);

    try {
      await api.sse("/api/rag/ask", body, {
        signal: ctrl.signal,
        onEvent: (event, data) => {
          if (event === "token") {
            acc += String(data);
            setStatus("streaming");
            setAnswer(acc);
          } else if (event === "sources") {
            setSources(data as Source[]);
          } else if (event === "done") {
            const final = (data as { answer: string }).answer;
            if (final) {
              acc = final;
              setAnswer(final);
            }
          } else if (event === "error") {
            streamError = (data as AskError).error;
          }
        },
      });
      if (streamError) throw new Error(streamError);
      if (!acc.trim()) throw new Error("The document returned no answer. Try rephrasing the question.");
    } catch (e) {
      if (ctrl.signal.aborted || (e instanceof ApiError && e.code === "aborted")) {
        // Stopped by the user: keep whatever streamed so far.
      } else {
        const message = e instanceof Error ? e.message : "Something went wrong while answering.";
        setError(message);
        toast.error(message);
      }
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null;
      setStatus("idle");
      requestAnimationFrame(() => answerRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    }
  }

  async function speak() {
    const { text, truncated } = plainSpeech(answer);
    if (!text || !voice) return;
    ttsAbortRef.current?.abort();
    const ctrl = new AbortController();
    ttsAbortRef.current = ctrl;
    setSpeaking(true);
    setTtsError(null);
    setShortened(truncated);
    setAudio(null);
    try {
      let blob: Blob;
      if (voice.startsWith("kokoro:")) {
        blob = await api.blob(
          "/api/kokoro/generate",
          { voice: voice.slice(7), text, speed: 1.0 },
          { signal: ctrl.signal }
        );
      } else {
        const v = clonedList.find((x) => x.id === voice.slice(6));
        const fd = new FormData();
        fd.append("text", text);
        fd.append("voice_id", voice.slice(6));
        fd.append("language", v?.language || "en");
        fd.append("mode", "fast");
        blob = await api.blob("/api/voices/generate", fd, { signal: ctrl.signal });
      }
      if (blob.size === 0) throw new Error("The voice returned empty audio.");
      setAudio(blob);
    } catch (e) {
      if (ctrl.signal.aborted || (e instanceof ApiError && e.code === "aborted")) return;
      const message = e instanceof Error ? e.message : "Could not generate speech.";
      setTtsError(message);
      toast.error(message);
    } finally {
      if (ttsAbortRef.current === ctrl) ttsAbortRef.current = null;
      setSpeaking(false);
    }
  }

  const voicesLoading = kLoading || cLoading;
  const hasAnswer = answer.length > 0;
  const streaming = status === "streaming";

  return (
    <div className="space-y-6">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void ask();
        }}
        className="space-y-3"
      >
        <label htmlFor="rag-question" className="sr-only">
          Question about {docName ?? "the selected document"}
        </label>
        <textarea
          id="rag-question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void ask();
            }
          }}
          disabled={!docId}
          rows={3}
          maxLength={2000}
          placeholder={docId ? `Ask anything about ${docName ?? "this document"}` : "Select a document to start asking"}
          className="w-full resize-y rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3.5 py-3 text-sm text-[var(--color-text-primary)] outline-none placeholder:text-[var(--color-text-secondary)] focus:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/40 disabled:cursor-not-allowed disabled:opacity-60"
        />

        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="min-w-0 flex-1">
            <label htmlFor="rag-voice" className="mb-1 block text-xs text-[var(--color-text-secondary)]">
              Answer voice
            </label>
            <select
              id="rag-voice"
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              disabled={voicesLoading && !voice}
              className="min-h-11 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 text-sm text-[var(--color-text-primary)] outline-none focus:border-[var(--color-accent)] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]/40 disabled:opacity-60"
            >
              {voicesLoading && !voice && <option value="">Loading voices</option>}
              {!voicesLoading && kokoroList.length === 0 && clonedList.length === 0 && (
                <option value="">No voices available</option>
              )}
              {kokoroList.length > 0 && (
                <optgroup label="Kokoro voices">
                  {kokoroList.map((v) => (
                    <option key={v.id} value={`kokoro:${v.id}`}>
                      {v.name}
                      {v.language ? ` (${v.language})` : ""}
                    </option>
                  ))}
                </optgroup>
              )}
              {clonedList.length > 0 && (
                <optgroup label="My cloned voices">
                  {clonedList.map((v) => (
                    <option key={v.id} value={`clone:${v.id}`}>
                      {v.name}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>

          {busy ? (
            <Button
              variant="outline"
              size="lg"
              onClick={() => abortRef.current?.abort()}
              leftIcon={<Square className="size-3.5 fill-current" aria-hidden />}
            >
              Stop
            </Button>
          ) : (
            <Button type="submit" size="lg" disabled={!canAsk} leftIcon={<Send className="size-4" aria-hidden />}>
              Ask
            </Button>
          )}
        </div>
        <p className="hidden text-xs text-[var(--color-text-secondary)] sm:block">Ctrl or Cmd + Enter to send</p>
      </form>

      <div ref={answerRef} aria-live="polite" aria-busy={busy}>
        {!hasAnswer && !error && status === "idle" && (
          <p className="text-sm text-[var(--color-text-secondary)]">
            {docId
              ? "Answers appear here as they are written, with the passages they came from."
              : "Pick or upload a document, then ask a question about it."}
          </p>
        )}

        {status === "retrieving" && (
          <div className="space-y-2.5" role="status" aria-label="Searching the document">
            <p className="text-sm text-[var(--color-text-secondary)]">Searching the document for relevant passages</p>
            {[92, 78, 54].map((w) => (
              <Skeleton key={w} className="h-3.5" style={{ width: `${w}%` }} />
            ))}
          </div>
        )}

        {(hasAnswer || error) && (
          <div className="space-y-4">
            {asked && <p className="text-sm font-medium text-[var(--color-text-primary)]">{asked}</p>}

            {hasAnswer && (
              <p className="max-w-[68ch] whitespace-pre-wrap text-[15px] leading-relaxed text-[var(--color-text-primary)]">
                {answer}
                {streaming && (
                  <span
                    aria-hidden
                    className="ml-0.5 inline-block h-[1.05em] w-[2px] translate-y-[3px] animate-pulse bg-[var(--color-accent)] motion-reduce:animate-none"
                  />
                )}
              </p>
            )}

            {error && (
              <ErrorState
                title="Could not answer"
                message={error}
                onRetry={busy ? undefined : () => void ask()}
                className="py-6"
              />
            )}

            {sources.length > 0 && (
              <div className="rounded-lg border border-[var(--color-border)]">
                <button
                  type="button"
                  aria-expanded={showSources}
                  aria-controls="rag-sources"
                  onClick={() => setShowSources((s) => !s)}
                  className="flex min-h-11 w-full items-center justify-between gap-2 rounded-lg px-3.5 text-left text-sm text-[var(--color-text-primary)] outline-none hover:bg-white/[0.03] focus-visible:ring-2 focus-visible:ring-[var(--color-accent)]"
                >
                  <span>
                    Sources <span className="text-[var(--color-text-secondary)] tabular-nums">({sources.length})</span>
                  </span>
                  <ChevronDown
                    className={`size-4 text-[var(--color-text-secondary)] transition-transform duration-200 motion-reduce:transition-none ${
                      showSources ? "rotate-180" : ""
                    }`}
                    aria-hidden
                  />
                </button>
                {showSources && (
                  <ol id="rag-sources" className="space-y-3 border-t border-[var(--color-border)] px-3.5 py-3">
                    {sources.map((s, i) => (
                      <li key={i} className="text-sm">
                        <p className="text-xs font-medium text-[var(--color-text-secondary)]">
                          [{i + 1}] Passage {s.index + 1} - match {Math.round(s.score * 100)}%
                        </p>
                        <p className="mt-1 max-w-[68ch] leading-relaxed text-[var(--color-text-secondary)]">
                          {s.text}
                        </p>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )}

            {hasAnswer && !busy && (
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant="outline"
                    size="lg"
                    onClick={() => void speak()}
                    disabled={!voice}
                    loading={speaking}
                    leftIcon={<Volume2 className="size-4" aria-hidden />}
                  >
                    {speaking ? "Generating audio" : audio ? "Speak again" : "Speak answer"}
                  </Button>
                  {speaking && (
                    <Button variant="ghost" size="sm" onClick={() => ttsAbortRef.current?.abort()}>
                      Cancel
                    </Button>
                  )}
                </div>
                {shortened && (
                  <p className="text-xs text-[var(--color-text-secondary)]" role="status">
                    The answer was shortened to fit the 2,000 character voice limit, so the audio covers only the first part.
                  </p>
                )}
                {ttsError && (
                  <p className="text-sm text-red-300" role="alert">
                    {ttsError}
                  </p>
                )}
                {audio && <WaveformPlayer src={audio} filename="answer.wav" autoPlay />}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
