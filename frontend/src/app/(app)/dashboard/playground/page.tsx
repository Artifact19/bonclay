"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { motion, useReducedMotion } from "motion/react";
import { toast } from "sonner";
import { Gauge, Headphones, Loader2, Mic, Phone } from "lucide-react";
import { api, ApiError, API_URL } from "@/lib/api";
import { langAttr } from "@/lib/lang";
import { useAgents, useAgentTemplates } from "@/hooks/useApi";
import { useHandsFree, type HandsFreeStatus } from "@/hooks/useHandsFree";
import { LLM_FIX_TEXT } from "@/lib/llm";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { AiGeneratedTag } from "@/components/audio/AiGeneratedTag";
import { AudioQueue } from "@/lib/audioQueue";
import { PerformancePanel } from "@/components/dashboard/PerformancePanel";
import { ToolCallChips } from "@/components/dashboard/ToolCallChips";
import type { ToolCall } from "@/types/api";
import { Modal } from "@/components/voices/Modal";
import { CallDialog } from "@/components/agents/CallDialog";
import {
  ReplyTiming,
  hasTimingInfo,
} from "@/components/playground/ReplyTiming";
import {
  SuggestedQuestions,
  TypeQuestion,
} from "@/components/playground/QuestionBar";
import { suggestedQuestionsFor } from "@/components/playground/suggestions";
import { VoiceMark } from "@/components/playground/VoiceMark";
import { languageNames } from "@/components/agents/languages";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";

interface ChatMessage {
  id: string;
  role: "user" | "agent";
  text: string;
  latencyMs?: number;
  audioUrl?: string;
  engineName?: string;
  llmProvider?: string | null;
  llmModel?: string | null;
  routeReason?: string;
  timings?: Record<string, number>;
  grounded?: boolean;
  sourcesUsed?: number;
  /** The reply is still arriving sentence by sentence. */
  streaming?: boolean;
  /** Spoken as it arrived, so the final audio is not auto-played again. */
  streamed?: boolean;
  firstAudioMs?: number;
  /** What the agent did with its tools while answering. */
  toolCalls?: ToolCall[];
  /** BCP-47 language of the text, when the server said which (the `lang` of the bubble). */
  lang?: string;
}

interface TranscriptEvent {
  user_text: string;
  language?: string;
}

interface SentenceEvent {
  index: number;
  text: string;
  audio_url: string;
  /** A short "one moment" clip played while the answer is slow: not part of the reply text. */
  filler?: boolean;
}

/** `done` of POST /api/conversations/{agent}/turn/stream. */
interface StreamDoneEvent {
  turn_id: string;
  user_text?: string;
  agent_text: string;
  agent_audio_url?: string;
  timings?: Record<string, number>;
  total_ms?: number;
  engine_name?: string;
  llm_provider?: string | null;
  llm_model?: string | null;
  route_reason?: string;
  grounded?: boolean;
  sources_used?: number;
  conversation_id?: string;
  tool_calls?: ToolCall[];
}

interface StreamErrorEvent {
  code: string;
  message: string;
}

type Phase = "transcribing" | "thinking" | "speaking";

interface TurnResponse {
  conversation_id?: string;
  user_text: string;
  agent_text: string;
  latency_ms?: number;
  agent_audio_url?: string;
  engine_name?: string;
  llm_provider?: string | null;
  llm_model?: string | null;
  route_reason?: string;
  timings?: Record<string, number>;
  grounded?: boolean;
  sources_used?: number;
  tool_calls?: ToolCall[];
}

// MediaRecorder output differs by browser (Chrome: webm, Safari: mp4); the backend checks extension vs content.
const recordingExt = (type: string) =>
  type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";

/** The query parameter other pages use to open the Playground on one agent: /dashboard/playground?agent=<id>. */
const AGENT_PARAM = "agent";

const HANDS_FREE_KEY = "bonclay.handsFree";

/** What the hands-free status line says: text for every state, so it never relies on colour. */
function handsFreeLabel(
  status: HandsFreeStatus,
  thinking: boolean,
  speaking: boolean,
): string {
  if (status === "starting") return "Starting the microphone…";
  if (status === "error") return "Microphone unavailable";
  if (status === "hearing") return "Hearing you…";
  if (speaking) return "Speaking… (talk to interrupt)";
  if (thinking) return "Thinking…";
  return "Listening…";
}

const absolute = (u: string) =>
  u.startsWith("http") || u.startsWith("blob:") || u.startsWith("data:")
    ? u
    : `${API_URL}${u}`;

export default function PlaygroundPage() {
  // useSearchParams needs a Suspense boundary for static rendering
  return (
    <Suspense fallback={null}>
      <Playground />
    </Suspense>
  );
}

function Playground() {
  const motionReduced = useReducedMotion();
  const linkedAgentId = useSearchParams()?.get(AGENT_PARAM) ?? "";
  const agentsQuery = useAgents();
  const templatesQuery = useAgentTemplates();
  const agents = agentsQuery.data ?? [];
  const agentsLoading = agentsQuery.isLoading;
  const [pickedAgentId, setSelectedAgentId] = useState("");
  // The agent the user picked; else the one a link asked for (?agent=<id>); else the newest.
  const selectedAgentId = agents.some((a) => a.id === pickedAgentId)
    ? pickedAgentId
    : agents.some((a) => a.id === linkedAgentId)
      ? linkedAgentId
      : (agents[0]?.id ?? "");
  const selectedAgent = agents.find((a) => a.id === selectedAgentId) ?? null;
  // A message the server did not label: an agent set up for one language speaks it; with several, say nothing.
  const fallbackLang =
    selectedAgent?.supported_languages?.length === 1
      ? langAttr(selectedAgent.supported_languages[0])
      : undefined;
  const questions = useMemo(
    () => suggestedQuestionsFor(selectedAgent, templatesQuery?.data),
    [selectedAgent, templatesQuery?.data],
  );
  // Two agents can share a name (a template used twice): add the day it was made so the list tells them apart.
  const optionLabel = (a: (typeof agents)[number]) => {
    if (agents.filter((o) => o.name === a.name).length < 2) return a.name;
    const made = new Date(a.created_at);
    return Number.isNaN(made.getTime())
      ? a.name
      : `${a.name} (made ${made.toLocaleDateString(undefined, { day: "numeric", month: "short" })})`;
  };
  const selectedIdRef = useRef(selectedAgentId);
  selectedIdRef.current = selectedAgentId;
  const [conversationId, setConversationId] = useState<string | null>(null);
  const conversationIdRef = useRef<string | null>(null);
  conversationIdRef.current = conversationId;
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [phase, setPhase] = useState<Phase>("transcribing");
  const [perfOpen, setPerfOpen] = useState(false);
  const [callOpen, setCallOpen] = useState(false);
  const [handsFree, setHandsFree] = useState(false);
  // The agent's voice is still coming out of the speakers (it can outlast the request that produced it).
  const [audioActive, setAudioActive] = useState(false);

  const queueRef = useRef<AudioQueue | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  const reduced = useRef(false);
  const meterRef = useRef<HTMLDivElement>(null);
  const liveRef = useRef({ processing: false, phase: "transcribing" as Phase });

  useEffect(() => {
    reduced.current = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
  }, []);

  useEffect(() => {
    liveRef.current = { processing: isProcessing, phase };
  });

  // Hands-free is remembered across visits. (Read after mount so the server and first client render agree.)
  useEffect(() => {
    try {
      if (localStorage.getItem(HANDS_FREE_KEY) === "1") setHandsFree(true);
    } catch {
      /* storage blocked: it just starts off */
    }
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({
      behavior: reduced.current ? "auto" : "smooth",
    });
  }, [messages, isProcessing]);

  // Release the mic if the page unmounts mid-recording, and stop a reply that is still playing or arriving.
  useEffect(
    () => () => {
      recorderRef.current?.stream?.getTracks().forEach((t) => t.stop());
      queueRef.current?.stop();
      abortRef.current?.abort();
    },
    [],
  );

  /** The original request/response turn: one JSON answer once the whole reply audio is ready. */
  const submitBlockingTurn = useCallback(
    async (fd: FormData) => {
      const data = await api.upload<TurnResponse>(
        `/api/conversations/${selectedAgentId}/turn`,
        fd,
      );
      if (data.conversation_id && !conversationId)
        setConversationId(data.conversation_id);
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", text: data.user_text },
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: data.agent_text,
          latencyMs: data.latency_ms,
          audioUrl: data.agent_audio_url
            ? absolute(data.agent_audio_url)
            : undefined,
          engineName: data.engine_name,
          llmProvider: data.llm_provider,
          llmModel: data.llm_model,
          routeReason: data.route_reason,
          timings: data.timings,
          grounded: data.grounded,
          sourcesUsed: data.sources_used,
          toolCalls: data.tool_calls,
        },
      ]);
    },
    [conversationId, selectedAgentId],
  );

  const submitTurn = useCallback(
    async (audio: Blob) => {
      setIsProcessing(true);
      setPhase("transcribing");
      queueRef.current?.stop();
      abortRef.current?.abort();
      setAudioActive(false);
      const ctrl = new AbortController();
      const queue = new AudioQueue();
      abortRef.current = ctrl;
      queueRef.current = queue;
      const agentMsgId = crypto.randomUUID();
      let agentShown = false;
      let finished = false;
      // The language the server heard: the reply is in it too.
      let spokenLang: string | undefined;
      const patchAgent = (
        patch:
          Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>),
      ) =>
        setMessages((prev) =>
          prev.map((m) =>
            m.id === agentMsgId
              ? { ...m, ...(typeof patch === "function" ? patch(m) : patch) }
              : m,
          ),
        );

      const onEvent = (event: string, raw: unknown) => {
        if (event === "transcript") {
          const d = raw as TranscriptEvent;
          spokenLang = langAttr(d.language);
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: "user",
              text: d.user_text,
              lang: spokenLang,
            },
          ]);
          setPhase("thinking");
        } else if (event === "sentence") {
          // The first sentence starts playing as soon as it arrives; the rest queue up behind it, in order.
          const d = raw as SentenceEvent;
          queue.add(absolute(d.audio_url));
          setPhase("speaking");
          setAudioActive(true);
          if (d.filler) return; // heard, but not shown: the reply text starts with the first real sentence
          if (!agentShown) {
            agentShown = true;
            setMessages((prev) => [
              ...prev,
              {
                id: agentMsgId,
                role: "agent",
                text: d.text,
                streaming: true,
                streamed: true,
                lang: spokenLang,
              },
            ]);
          } else {
            patchAgent((m) => ({ text: `${m.text} ${d.text}` }));
          }
        } else if (event === "done") {
          const d = raw as StreamDoneEvent;
          finished = true;
          queue.close();
          if (d.conversation_id)
            setConversationId((prev) => prev ?? d.conversation_id ?? null);
          const final: Partial<ChatMessage> = {
            text: d.agent_text,
            streaming: false,
            latencyMs: d.total_ms ?? d.timings?.total_ms,
            firstAudioMs: d.timings?.first_audio_ms,
            audioUrl: d.agent_audio_url
              ? absolute(d.agent_audio_url)
              : undefined,
            engineName: d.engine_name,
            llmProvider: d.llm_provider,
            llmModel: d.llm_model,
            routeReason: d.route_reason,
            timings: d.timings,
            grounded: d.grounded,
            sourcesUsed: d.sources_used,
            toolCalls: d.tool_calls,
          };
          if (agentShown) patchAgent(final);
          else
            setMessages((prev) => [
              ...prev,
              {
                id: agentMsgId,
                role: "agent",
                text: d.agent_text,
                streamed: true,
                lang: spokenLang,
                ...final,
              },
            ]);
        } else if (event === "error") {
          const d = raw as StreamErrorEvent;
          finished = true;
          queue.close();
          if (agentShown) patchAgent({ streaming: false });
          if (d.code === "LLM_UNAVAILABLE")
            toast.error(d.message, { description: LLM_FIX_TEXT });
          else
            toast.error(d.message || "The agent did not respond. Try again.");
        }
      };

      try {
        const fd = new FormData();
        fd.append("audio", audio, `audio.${recordingExt(audio.type)}`);
        if (conversationId) fd.append("conversation_id", conversationId);
        try {
          await api.sse(
            `/api/conversations/${selectedAgentId}/turn/stream`,
            fd,
            { onEvent, signal: ctrl.signal },
          );
          if (!finished && !ctrl.signal.aborted) {
            if (agentShown) patchAgent({ streaming: false });
            toast.error("The reply was cut off. Try again.");
          }
        } catch (e) {
          // An older backend has no stream endpoint (plain 404 / 405 from the framework): use the original turn.
          const missing =
            e instanceof ApiError &&
            (e.status === 404 || e.status === 405) &&
            e.code === `http_${e.status}`;
          if (!missing || agentShown) throw e;
          await submitBlockingTurn(fd);
        }
      } catch (e) {
        if (agentShown) patchAgent({ streaming: false });
        if (e instanceof ApiError && e.code === "aborted") return;
        if (e instanceof ApiError && e.code === "LLM_UNAVAILABLE")
          toast.error(e.message, { description: LLM_FIX_TEXT });
        else
          toast.error(
            e instanceof Error
              ? e.message
              : "The agent did not respond. Try again.",
          );
      } finally {
        if (abortRef.current === ctrl) setIsProcessing(false);
        // However the request ended, nothing more is coming: the speaking state ends when the last clip does.
        queue.close();
        void queue.finished().then(() => {
          if (queueRef.current === queue) setAudioActive(false);
        });
      }
    },
    [conversationId, selectedAgentId, submitBlockingTurn],
  );

  /** Cut the agent off and drop the request it was answering (barge-in, or hands-free turned off). */
  const interruptAgent = useCallback(() => {
    queueRef.current?.stop();
    abortRef.current?.abort();
    setAudioActive(false);
    setIsProcessing(false);
  }, []);

  /** A typed question: the agent's text answer, no microphone and no speakers needed. */
  const sendText = useCallback(async (text: string) => {
    const agentId = selectedIdRef.current;
    if (!agentId || !text.trim() || liveRef.current.processing) return;
    queueRef.current?.stop();
    abortRef.current?.abort();
    setAudioActive(false);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", text },
    ]);
    setIsProcessing(true);
    setPhase("thinking");
    try {
      // The same turn as a spoken question, in the same conversation; the text skips speech recognition.
      const fd = new FormData();
      fd.append("text", text);
      if (conversationIdRef.current)
        fd.append("conversation_id", conversationIdRef.current);
      const data = await api.upload<TurnResponse>(
        `/api/conversations/${agentId}/turn`,
        fd,
        { signal: ctrl.signal },
      );
      if (selectedIdRef.current !== agentId) return; // the user moved to another agent meanwhile
      if (data.conversation_id && !conversationIdRef.current)
        setConversationId(data.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "agent",
          text: data.agent_text,
          latencyMs: data.latency_ms,
          audioUrl: data.agent_audio_url
            ? absolute(data.agent_audio_url)
            : undefined,
          engineName: data.engine_name,
          llmProvider: data.llm_provider,
          llmModel: data.llm_model,
          routeReason: data.route_reason,
          timings: data.timings,
          grounded: data.grounded,
          sourcesUsed: data.sources_used,
          toolCalls: data.tool_calls,
        },
      ]);
    } catch (e) {
      if (e instanceof ApiError && e.code === "aborted") return;
      if (e instanceof ApiError && e.code === "LLM_UNAVAILABLE")
        toast.error(e.message, { description: LLM_FIX_TEXT });
      else
        toast.error(
          e instanceof Error
            ? e.message
            : "The agent did not respond. Try again.",
        );
    } finally {
      if (abortRef.current === ctrl) setIsProcessing(false);
    }
  }, []);

  const { status: handsFreeStatus } = useHandsFree({
    enabled: handsFree,
    isPlaying: () =>
      !!queueRef.current?.busy ||
      (liveRef.current.processing && liveRef.current.phase === "speaking"),
    onUtterance: (audio) => {
      if (selectedAgentId) void submitTurn(audio);
    },
    onBargeIn: interruptAgent,
    onLevel: (level) => {
      if (meterRef.current)
        meterRef.current.style.transform = `scaleX(${level})`;
    },
    onError: (message) => {
      setHandsFree(false);
      toast.error(message);
    },
  });

  const toggleHandsFree = () => {
    const next = !handsFree;
    setHandsFree(next);
    try {
      localStorage.setItem(HANDS_FREE_KEY, next ? "1" : "0");
    } catch {
      /* storage blocked: the choice just lasts this visit */
    }
    if (next) stopRecording();
    else interruptAgent(); // the microphone is released by the hook; a reply still arriving is dropped too
  };

  const startRecording = async () => {
    if (isRecording || isProcessing || handsFree) return;
    if (!selectedAgentId) {
      toast.error("Choose an agent to talk to first.");
      return;
    }
    queueRef.current?.stop(); // speaking over the agent cuts it off
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = rec;
      chunksRef.current = [];
      rec.ondataavailable = (ev) =>
        ev.data.size > 0 && chunksRef.current.push(ev.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        if (blob.size > 0) void submitTurn(blob);
      };
      rec.start();
      setIsRecording(true);
    } catch {
      toast.error(
        "Microphone access was blocked. Allow it in your browser settings to talk.",
      );
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === " " || e.key === "Enter") && !e.repeat) {
      e.preventDefault();
      void startRecording();
    }
  };
  const onKeyUp = (e: React.KeyboardEvent) => {
    if (e.key === " " || e.key === "Enter") stopRecording();
  };

  const pickAgent = (id: string) => {
    interruptAgent();
    setSelectedAgentId(id);
    setConversationId(null);
    setMessages([]);
  };
  const hasAgents = agents.length > 0 || agentsLoading;
  const actionBtn =
    "inline-flex items-center justify-center gap-2 rounded-lg border border-border text-sm text-text-secondary transition-colors hover:bg-bg-elevated hover:text-text-primary focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)] disabled:cursor-not-allowed disabled:opacity-40";

  return (
    <div className="mx-auto flex h-[calc(100dvh-4rem)] w-full max-w-7xl flex-col px-4 py-4 sm:px-6 md:py-6">
      <div className="grid min-h-0 flex-1 gap-5 lg:grid-cols-[17rem_minmax(0,1fr)]">
        <aside
          className="hidden min-h-0 lg:flex lg:flex-col"
          aria-label="Agent picker"
        >
          <Card
            variant="solid"
            padded={false}
            className="flex min-h-0 flex-1 flex-col overflow-hidden"
          >
            <div className="flex items-center justify-between px-4 pb-2 pt-4">
              <h2 className="font-display text-base text-text-primary">
                Your agents
              </h2>
              <Badge>{agents.length}</Badge>
            </div>
            <ul
              role="list"
              aria-label="Agents"
              className="min-h-0 flex-1 overflow-y-auto px-2 pb-2"
            >
              {agentsLoading &&
                [0, 1, 2].map((i) => (
                  <li
                    key={i}
                    className="flex items-center gap-3 px-2 py-3"
                    aria-hidden="true"
                  >
                    <div className="h-9 w-9 shrink-0 animate-pulse rounded-[10px] bg-bg-elevated" />
                    <div className="flex-1 space-y-2">
                      <div className="h-3 w-2/3 animate-pulse rounded bg-bg-elevated" />
                      <div className="h-2.5 w-full animate-pulse rounded bg-bg-elevated" />
                    </div>
                  </li>
                ))}
              {!agentsLoading && !agents.length && (
                <li className="px-3 py-4 text-sm text-text-secondary">
                  {agentsQuery.error
                    ? "Could not load agents. "
                    : "No agents yet. "}
                  <Link
                    className="underline underline-offset-2"
                    href="/dashboard/studio/agents"
                  >
                    Agent Studio
                  </Link>{" "}
                  is where you make one.
                </li>
              )}
              {agents.map((a) => {
                const selected = a.id === selectedAgentId;
                return (
                  <li key={a.id} className="relative">
                    <button
                      type="button"
                      aria-pressed={selected}
                      onClick={() => pickAgent(a.id)}
                      className={`relative flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)] ${
                        selected ? "bg-bg-elevated" : "hover:bg-bg-elevated/60"
                      }`}
                    >
                      {selected && (
                        <motion.span
                          layoutId="agent-rail-mark"
                          className="absolute inset-y-2 left-0 w-[3px] rounded-full bg-[#F0A23A]"
                          transition={
                            motionReduced
                              ? { duration: 0 }
                              : { type: "spring", stiffness: 400, damping: 32 }
                          }
                          aria-hidden="true"
                        />
                      )}
                      <VoiceMark name={a.name} size={36} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm font-medium text-text-primary">
                          {optionLabel(a)}
                        </span>
                        <span className="line-clamp-1 block text-xs text-text-secondary">
                          {a.greeting?.trim() || "No greeting yet"}
                        </span>
                        {a.supported_languages?.length > 0 && (
                          <span className="mt-0.5 line-clamp-1 block text-[10px] uppercase tracking-wide text-text-muted">
                            {languageNames(a.supported_languages)}
                          </span>
                        )}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
            <div className="space-y-2 border-t border-border p-3">
              <button
                type="button"
                onClick={() => setCallOpen(true)}
                disabled={!selectedAgentId}
                className={`${actionBtn} h-9 w-full`}
              >
                <Phone className="h-4 w-4" aria-hidden="true" />
                Phone someone
              </button>
              <button
                type="button"
                onClick={() => setPerfOpen(true)}
                disabled={!selectedAgentId}
                className={`${actionBtn} h-9 w-full`}
              >
                <Gauge className="h-4 w-4" aria-hidden="true" />
                Usual reply time
              </button>
              <Link
                href="/dashboard/studio/agents"
                className="block pt-1 text-center text-xs text-text-secondary underline-offset-2 hover:text-text-primary hover:underline"
              >
                Set up agents
              </Link>
            </div>
          </Card>
        </aside>

        <main className="flex min-h-0 flex-col gap-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div className="min-w-0">
              <h1 className="font-display text-xl tracking-tight text-text-primary sm:text-2xl">
                Talk to your agent
              </h1>
              <p className="mt-1 text-sm text-text-secondary">
                Ask your agent a question by typing or speaking, and see how it
                answers.
              </p>
            </div>
            <div className="flex items-center gap-2 lg:hidden">
              <label htmlFor="agent" className="text-sm text-text-secondary">
                Talk to
              </label>
              <select
                id="agent"
                value={selectedAgentId}
                disabled={agentsLoading || !agents.length}
                onChange={(e) => pickAgent(e.target.value)}
                className="h-10 min-w-0 flex-1 rounded-lg border border-border bg-bg-surface px-3 text-sm text-text-primary focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)] sm:w-auto sm:min-w-[12rem] sm:max-w-[18rem] sm:flex-none"
              >
                {agentsLoading && <option value="">Loading agents</option>}
                {!agentsLoading && !agents.length && (
                  <option value="">
                    {agentsQuery.error
                      ? "Could not load agents"
                      : "No agents yet"}
                  </option>
                )}
                {agents.map((a) => (
                  <option key={a.id} value={a.id}>
                    {optionLabel(a)}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => setCallOpen(true)}
                disabled={!selectedAgentId}
                className={`${actionBtn} h-10 w-10 shrink-0`}
              >
                <Phone className="h-4 w-4" aria-hidden="true" />
                <span className="sr-only">Phone someone</span>
              </button>
              <button
                type="button"
                onClick={() => setPerfOpen(true)}
                disabled={!selectedAgentId}
                className={`${actionBtn} h-10 w-10 shrink-0`}
              >
                <Gauge className="h-4 w-4" aria-hidden="true" />
                <span className="sr-only">Usual reply time</span>
              </button>
            </div>
          </div>
          <CallDialog
            agent={selectedAgent}
            open={callOpen}
            onClose={() => setCallOpen(false)}
          />
          <Modal
            open={perfOpen}
            onClose={() => setPerfOpen(false)}
            title="Usual reply time"
            description="How long your agent usually takes to answer, and how long its slowest replies take."
            size="lg"
          >
            <PerformancePanel
              agentId={selectedAgentId}
              hideHeading
              className="!border-0 !bg-transparent !p-0"
            />
          </Modal>

          <Card
            variant="solid"
            padded={false}
            className="relative flex min-h-0 flex-1 flex-col overflow-hidden"
          >
            <div
              className="flex-1 space-y-5 overflow-y-auto p-4 sm:p-6"
              role="log"
              aria-live="polite"
              aria-label="Conversation"
            >
              {messages.length === 0 && !isProcessing ? (
                <div className="flex h-full items-center">
                  {selectedAgent ? (
                    <motion.div
                      key={selectedAgentId}
                      initial={motionReduced ? false : { opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{
                        duration: motionReduced ? 0 : 0.4,
                        ease: [0.16, 1, 0.3, 1],
                      }}
                      className="w-full max-w-2xl"
                    >
                      <div className="flex items-center gap-3">
                        <VoiceMark name={selectedAgent.name} size={48} />
                        <span className="text-sm font-medium text-text-primary">
                          {selectedAgent.name}
                        </span>
                      </div>
                      <blockquote className="mt-4 font-display text-[clamp(1.375rem,2.4vw,2rem)] leading-snug tracking-[-0.02em] text-text-primary">
                        {selectedAgent.greeting?.trim() ||
                          "Namaste. How can I help?"}
                      </blockquote>
                      <p className="mt-4 text-sm text-text-secondary">
                        <span className="font-medium text-text-primary">
                          Ready when you are
                        </span>
                        . Type a question below, tap one of these, or hold the
                        microphone and speak.
                      </p>
                      <p className="mb-2 mt-5 text-xs font-medium uppercase tracking-wide text-text-muted">
                        Try asking
                      </p>
                      <SuggestedQuestions
                        className="justify-start"
                        questions={questions}
                        onPick={(q) => void sendText(q)}
                        disabled={isProcessing || handsFree}
                      />
                    </motion.div>
                  ) : (
                    <div className="max-w-md">
                      <p className="text-sm font-medium text-text-primary">
                        {hasAgents
                          ? "Ready when you are"
                          : "Create an agent to start talking"}
                      </p>
                      <p className="mt-1 text-sm text-text-secondary">
                        {hasAgents ? (
                          <>
                            Type a question below, tap one of these, or hold the
                            microphone and speak.
                          </>
                        ) : (
                          <>
                            {agentsQuery.error
                              ? `${agentsQuery.error.message} `
                              : ""}
                            Design an agent in{" "}
                            <Link
                              className="underline underline-offset-2"
                              href="/dashboard/studio/agents"
                            >
                              Agent Studio
                            </Link>
                            , then come back to talk to it.
                          </>
                        )}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                messages.map((m) => (
                  <motion.div
                    key={m.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                    className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}
                  >
                    {m.role === "agent" ? (
                      <VoiceMark
                        name={selectedAgent?.name ?? "Agent"}
                        size={32}
                      />
                    ) : (
                      <div
                        className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-bg-elevated text-[10px] font-semibold text-text-secondary"
                        aria-hidden="true"
                      >
                        You
                      </div>
                    )}
                    <div
                      className={`flex min-w-0 max-w-[85%] flex-col gap-2 sm:max-w-[75%] ${m.role === "user" ? "items-end" : "items-start"}`}
                    >
                      <div
                        className={`px-4 py-2.5 text-[15px] leading-relaxed text-text-primary ${
                          m.role === "user"
                            ? "rounded-2xl rounded-tr-md bg-[#F0A23A]/15"
                            : "rounded-2xl rounded-tl-md border border-border bg-bg-base"
                        }`}
                      >
                        <span lang={m.lang ?? fallbackLang}>{m.text}</span>
                      </div>
                      {m.role === "agent" && !m.streaming && (
                        <ToolCallChips calls={m.toolCalls} />
                      )}
                      {m.role === "agent" && m.streaming && (
                        <div
                          className="flex items-center gap-2 text-xs text-[var(--color-text-secondary)]"
                          role="status"
                        >
                          <span
                            className="size-1.5 rounded-full bg-[#F0A23A] motion-safe:animate-pulse"
                            aria-hidden="true"
                          />
                          Speaking
                          <AiGeneratedTag />
                        </div>
                      )}
                      {m.role === "agent" &&
                        !m.streaming &&
                        hasTimingInfo(m) && <ReplyTiming info={m} />}
                      {m.role === "agent" && m.audioUrl && !m.streaming && (
                        <WaveformPlayer
                          src={m.audioUrl}
                          height={44}
                          autoPlay={
                            !m.streamed &&
                            m.id === messages[messages.length - 1].id
                          }
                          filename="agent-reply.wav"
                          className="w-full"
                        />
                      )}
                    </div>
                  </motion.div>
                ))
              )}
              {isProcessing && phase !== "speaking" && (
                <div
                  className="flex items-center gap-3 text-sm text-[var(--color-text-secondary)]"
                  role="status"
                >
                  <Loader2
                    className="h-4 w-4 motion-safe:animate-spin"
                    aria-hidden="true"
                  />
                  {phase === "thinking" ? "Thinking" : "Listening and thinking"}
                </div>
              )}
              <div ref={endRef} />
            </div>

            <div className="border-t border-border p-3 sm:p-4">
              <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap sm:gap-3">
                <button
                  type="button"
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onMouseLeave={stopRecording}
                  onTouchStart={startRecording}
                  onTouchEnd={stopRecording}
                  onKeyDown={onKeyDown}
                  onKeyUp={onKeyUp}
                  onBlur={stopRecording}
                  disabled={isProcessing || !selectedAgentId || handsFree}
                  aria-pressed={isRecording}
                  aria-label={
                    isRecording ? "Recording. Release to send." : "Hold to talk"
                  }
                  className="relative grid h-12 w-12 shrink-0 touch-none select-none place-items-center rounded-full text-[var(--color-on-accent)] transition-transform duration-200 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[var(--color-accent-cyan)]"
                  style={{
                    background: "#F0A23A",
                    color: "#141008",
                    transform: isRecording ? "scale(1.08)" : undefined,
                  }}
                >
                  {isRecording && (
                    <span
                      className="absolute inset-0 rounded-full bg-white/25 motion-safe:animate-ping"
                      aria-hidden="true"
                    />
                  )}
                  <Mic className="relative h-5 w-5" />
                </button>
                <TypeQuestion
                  onSend={(t) => void sendText(t)}
                  disabled={!selectedAgentId || handsFree || isRecording}
                  busy={isProcessing}
                />
                <button
                  type="button"
                  onClick={toggleHandsFree}
                  disabled={!selectedAgentId}
                  aria-pressed={handsFree}
                  className={`inline-flex h-10 shrink-0 items-center gap-1.5 rounded-full border px-3 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)] disabled:cursor-not-allowed disabled:opacity-40 ${
                    handsFree
                      ? "border-[#F0A23A] bg-[#F0A23A]/15 text-[var(--color-text-primary)]"
                      : "border-border text-text-secondary hover:bg-bg-elevated hover:text-text-primary"
                  }`}
                >
                  <Headphones className="h-4 w-4" aria-hidden="true" />
                  Hands-free
                  <span className="text-xs font-medium">
                    {handsFree ? "On" : "Off"}
                  </span>
                </button>
              </div>
              <div className="mt-3 flex flex-col items-center gap-2">
                {handsFree && (
                  <div
                    className="h-1.5 w-40 overflow-hidden rounded-full bg-bg-elevated"
                    aria-hidden="true"
                  >
                    <div
                      ref={meterRef}
                      className="h-full origin-left rounded-full bg-[#F0A23A]"
                      style={{ transform: "scaleX(0)" }}
                    />
                  </div>
                )}
                <p
                  className="text-center text-xs text-text-secondary"
                  role="status"
                  aria-live="polite"
                >
                  {handsFree
                    ? handsFreeLabel(
                        handsFreeStatus,
                        isProcessing && phase !== "speaking",
                        audioActive,
                      )
                    : isRecording
                      ? "Recording. Release to send."
                      : "Hold to talk"}
                </p>
              </div>
            </div>
          </Card>
        </main>
      </div>
    </div>
  );
}
