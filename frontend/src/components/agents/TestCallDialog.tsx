"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, PhoneCall, Play } from "lucide-react";
import { api, ApiError, API_URL } from "@/lib/api";
import { AudioQueue } from "@/lib/audioQueue";
import { AiGeneratedTag } from "@/components/audio/AiGeneratedTag";
import { Badge, Button } from "@/components/ui";
import { Modal } from "@/components/voices/Modal";
import { errMessage } from "@/components/dashboard/format";
import { LLM_FIX_TEXT } from "@/lib/llm";
import { langAttr } from "@/lib/lang";
import type { Agent } from "@/types/api";

export type ScenarioId = "normal" | "not_in_doc" | "trick" | "hinglish" | "custom";

export const TEST_SCENARIOS: { id: ScenarioId; label: string; turns: number }[] = [
  { id: "normal", label: "Ask a normal question from your document", turns: 2 },
  { id: "not_in_doc", label: "Ask something not in your document", turns: 2 },
  { id: "trick", label: "Try to trick the agent", turns: 2 },
  { id: "hinglish", label: "Speak Hinglish", turns: 2 },
  { id: "custom", label: "Custom", turns: 1 },
];

interface CallerResponse {
  text: string;
  voice: string;
  audio_base64: string;
  turns_total: number;
}

interface VerdictResponse {
  verdict: string;
  label: string;
  ok: boolean | null;
}

interface DoneEvent {
  agent_text: string;
  conversation_id?: string;
  sources_used?: number;
  grounded?: boolean;
}

export interface CallTurn {
  callerText: string;
  callerVoice: string;
  heard?: string;
  agentText: string;
  sourcesUsed: number;
  verdict?: VerdictResponse;
  /** BCP-47 language of what was said in this turn (the `lang` of the transcript text); unset when it is not known. */
  lang?: string;
}

const absolute = (u: string) => (u.startsWith("http") || u.startsWith("data:") || u.startsWith("blob:") ? u : `${API_URL}${u}`);
const PLAY_TIMEOUT_MS = 30000;

function base64ToBlob(b64: string, type = "audio/wav"): Blob {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type });
}

/** Plays one clip and resolves when it ended (or could not play: a blocked autoplay must not stall the test). */
function playClip(url: string, stop: { current: (() => void) | null }): Promise<void> {
  return new Promise((resolve) => {
    const audio = new Audio(url);
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      stop.current = null;
      resolve();
    };
    const timer = setTimeout(done, PLAY_TIMEOUT_MS);
    stop.current = () => {
      try {
        audio.pause();
      } catch {
        /* already torn down */
      }
      done();
    };
    audio.addEventListener("ended", done);
    audio.addEventListener("error", done);
    try {
      const p = audio.play();
      if (p && typeof p.catch === "function") p.catch(done);
    } catch {
      done();
    }
  });
}

type Phase = "idle" | "caller" | "thinking" | "speaking" | "judging" | "done";

const PHASE_TEXT: Record<Phase, string> = {
  idle: "",
  caller: "The caller is speaking",
  thinking: "The agent is thinking",
  speaking: "The agent is answering",
  judging: "Checking the answer",
  done: "",
};

interface Props {
  agent: Agent | null;
  open: boolean;
  onClose: () => void;
}

/**
 * Owner "Test call": a simulated caller (a Kokoro preset voice, never the agent's own) speaks a scripted line, it goes
 * through the normal spoken-turn endpoint as real audio, the agent's reply is played with the AI-generated label, and a
 * short verdict says what happened. The conversation is flagged as a test call so Analytics ignores it.
 */
export function TestCallDialog({ agent, open, onClose }: Props) {
  const [scenario, setScenario] = useState<ScenarioId>("normal");
  const [customText, setCustomText] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [turns, setTurns] = useState<CallTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [llmHint, setLlmHint] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const queueRef = useRef<AudioQueue | null>(null);
  const stopClipRef = useRef<(() => void) | null>(null);

  const stopAll = useCallback(() => {
    abortRef.current?.abort();
    queueRef.current?.stop();
    stopClipRef.current?.();
  }, []);

  // Closing the dialog (or leaving the page) ends the call and silences it.
  useEffect(() => {
    if (!open) return;
    return () => stopAll();
  }, [open, stopAll]);

  if (!agent) return null;
  const running = phase !== "idle" && phase !== "done";
  const meta = TEST_SCENARIOS.find((s) => s.id === scenario)!;
  const noDocs = !agent.document_count && (scenario === "normal" || scenario === "hinglish");

  const update = (index: number, patch: Partial<CallTurn>) =>
    setTurns((prev) => prev.map((t, i) => (i === index ? { ...t, ...patch } : t)));

  const start = async () => {
    stopAll();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setTurns([]);
    setError(null);
    setLlmHint(false);
    let conversationId: string | null = null;
    try {
      for (let i = 0; i < meta.turns; i++) {
        if (ctrl.signal.aborted) return;
        setPhase("caller");
        const caller = await api.post<CallerResponse>(
          `/api/agents/${agent.id}/test-call/caller`,
          { scenario, turn: i, custom_text: scenario === "custom" ? customText : undefined },
          { signal: ctrl.signal }
        );
        // The Hinglish scenario is Hindi in Roman letters; otherwise an agent set up for one language speaks it.
        const lang = scenario === "hinglish" ? langAttr("hinglish") : agent.supported_languages?.length === 1 ? langAttr(agent.supported_languages[0]) : undefined;
        setTurns((prev) => [...prev, { callerText: caller.text, callerVoice: caller.voice, agentText: "", sourcesUsed: 0, lang }]);
        const wav = base64ToBlob(caller.audio_base64);
        await playClip(`data:audio/wav;base64,${caller.audio_base64}`, stopClipRef);
        if (ctrl.signal.aborted) return;

        setPhase("thinking");
        const queue = new AudioQueue();
        queueRef.current = queue;
        let done: DoneEvent | null = null;
        let failure: string | null = null;
        let spoken = "";
        const fd = new FormData();
        fd.append("audio", wav, "caller.wav");
        fd.append("test_call", "true");
        if (conversationId) fd.append("conversation_id", conversationId);
        await api.sse(`/api/conversations/${agent.id}/turn/stream`, fd, {
          signal: ctrl.signal,
          onEvent: (event, raw) => {
            const d = raw as Record<string, unknown>;
            if (event === "transcript") {
              update(i, { heard: String(d.user_text ?? "") });
            } else if (event === "sentence") {
              queue.add(absolute(String(d.audio_url)));
              setPhase("speaking");
              if (!d.filler) {
                spoken = `${spoken} ${String(d.text ?? "")}`.trim();
                update(i, { agentText: spoken });
              }
            } else if (event === "done") {
              done = raw as DoneEvent;
              queue.close();
            } else if (event === "error") {
              failure = String(d.message ?? "The agent did not respond.");
              if (d.code === "LLM_UNAVAILABLE") setLlmHint(true);
              queue.close();
            }
          },
        });
        if (failure) throw new Error(failure);
        if (!done) throw new Error("The reply was cut off. Try again.");
        const finished: DoneEvent = done;
        conversationId = finished.conversation_id ?? conversationId;
        update(i, { agentText: finished.agent_text, sourcesUsed: finished.sources_used ?? 0 });
        await queue.finished();
        if (ctrl.signal.aborted) return;

        setPhase("judging");
        const verdict = await api.post<VerdictResponse>(
          `/api/agents/${agent.id}/test-call/verdict`,
          { scenario, caller_text: caller.text, agent_text: finished.agent_text, sources_used: finished.sources_used ?? 0 },
          { signal: ctrl.signal }
        );
        update(i, { verdict });
      }
      setPhase("done");
    } catch (e) {
      if (e instanceof ApiError && e.code === "aborted") return;
      if (e instanceof ApiError && e.code === "LLM_UNAVAILABLE") setLlmHint(true);
      setError(errMessage(e, "The practice call failed. Try again."));
      setPhase("idle");
    }
  };

  const close = () => {
    stopAll();
    setPhase("idle");
    onClose();
  };

  return (
    <Modal
      open={open}
      onClose={close}
      size="lg"
      title={`Practice call: ${agent.name}`}
      description="A simulated caller phones your agent so you can hear how it answers. Practice calls are not counted in your results."
    >
      <div className="space-y-5">
        <div>
          <label htmlFor="test-scenario" className="mb-1.5 block text-xs font-medium text-text-secondary">
            Scenario
          </label>
          <select
            id="test-scenario"
            value={scenario}
            onChange={(e) => setScenario(e.target.value as ScenarioId)}
            disabled={running}
            className="h-10 w-full rounded-lg border border-border bg-bg-base px-3 text-sm text-text-primary focus:border-accent-primary focus:outline-none disabled:opacity-50"
          >
            {TEST_SCENARIOS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        {scenario === "custom" && (
          <div>
            <label htmlFor="test-custom" className="mb-1.5 block text-xs font-medium text-text-secondary">
              What the caller says
            </label>
            <textarea
              id="test-custom"
              rows={2}
              maxLength={300}
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              disabled={running}
              placeholder="e.g. Can I reschedule my appointment to Friday?"
              className="w-full resize-none rounded-lg border border-border bg-bg-base px-3 py-2 text-sm text-text-primary focus:border-accent-primary focus:outline-none disabled:opacity-50"
            />
          </div>
        )}

        {noDocs && (
          <p className="text-xs text-text-secondary">
            This agent has no documents yet, so it will have nothing to answer from. Add one on the agent&apos;s Basics tab first.
          </p>
        )}

        <div className="flex items-center justify-between gap-3">
          <p role="status" className="flex items-center gap-2 text-xs text-text-secondary">
            {running && <Loader2 className="size-3.5 animate-spin" aria-hidden />}
            {PHASE_TEXT[phase]}
          </p>
          <Button
            onClick={start}
            loading={running}
            disabled={running || (scenario === "custom" && !customText.trim())}
            disabledReason={scenario === "custom" && !customText.trim() ? "Type what the caller should say first" : undefined}
            leftIcon={<Play className="size-4" />}
          >
            {phase === "done" || turns.length ? "Call again" : "Start practice call"}
          </Button>
        </div>

        {error && (
          <div role="alert" className="rounded-xl border border-accent-error/30 bg-accent-error/10 p-3 text-sm text-text-primary">
            {error}
            {llmHint && (
              <details className="mt-1">
                <summary className="min-h-8 cursor-pointer text-xs font-medium text-text-secondary hover:text-text-primary">Details for your IT person</summary>
                <p className="mt-1 text-xs text-text-secondary">
                  Fix: <code className="rounded bg-bg-elevated px-1.5 py-0.5 font-mono">{LLM_FIX_TEXT}</code>
                </p>
              </details>
            )}
          </div>
        )}

        {turns.length > 0 && (
          <ol className="space-y-4" aria-label="Practice call transcript">
            {turns.map((t, i) => (
              <li key={i} className="space-y-2 rounded-xl border border-border bg-bg-base p-3">
                <div className="flex items-start gap-2 text-sm">
                  <PhoneCall className="mt-0.5 size-4 shrink-0 text-text-secondary" aria-hidden />
                  <div className="min-w-0">
                    <p className="text-xs text-text-secondary">
                      Simulated caller <span className="font-mono">({t.callerVoice})</span>
                    </p>
                    <p lang={t.lang} className="text-text-primary">{t.callerText}</p>
                    {t.heard && t.heard !== t.callerText && (
                      <p className="text-xs text-text-secondary">
                        Agent heard: &ldquo;<span lang={t.lang}>{t.heard}</span>&rdquo;
                      </p>
                    )}
                  </div>
                </div>
                {t.agentText && (
                  <div className="pl-6 text-sm">
                    <p className="flex items-center gap-2 text-xs text-text-secondary">
                      {agent.name} <AiGeneratedTag />
                    </p>
                    <p lang={t.lang} className="text-text-primary">{t.agentText}</p>
                  </div>
                )}
                {t.verdict && (
                  <div className="pl-6">
                    <Badge tone={t.verdict.ok === true ? "success" : t.verdict.ok === false ? "warning" : "neutral"}>
                      {t.verdict.label}
                    </Badge>
                  </div>
                )}
              </li>
            ))}
          </ol>
        )}
      </div>
    </Modal>
  );
}
