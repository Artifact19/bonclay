"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Reorder, useDragControls, motion } from "motion/react";
import { toast } from "sonner";
import {
  ArrowDown,
  ArrowUp,
  GripVertical,
  MessagesSquare,
  Plus,
  Trash2,
  X,
} from "lucide-react";
import { useKokoroVoices, useVoices } from "@/hooks/useApi";
import {
  WaveformPlayer,
  type WaveformControls,
} from "@/components/audio/WaveformPlayer";
import { Button, EmptyState } from "@/components/ui";
import { GenerateButton, type GenerateState } from "./GenerateButton";
import { errorMessage, generateBlob } from "./studioApi";

const PALETTE = [
  "#F0A23A",
  "#E0782A",
  "#F59E0B",
  "#10B981",
  "#F43F5E",
  "#3B82F6",
  "#D946EF",
  "#84CC16",
];
const MAX_SPEAKERS = PALETTE.length;
const uid = () => Math.random().toString(36).slice(2, 9);

interface Speaker {
  id: string;
  name: string;
  /** "preset:Ryan" or "cloned:<voice id>" */
  voice: string;
  color: string;
}
interface Line {
  id: string;
  speakerId: string;
  text: string;
}

const field =
  "w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2 text-sm text-[var(--color-text-primary)] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--color-accent-cyan)]";

export function DialogueBuilder() {
  const { data: voices } = useVoices();
  const cloned = useMemo(() => (Array.isArray(voices) ? voices : []), [voices]);
  const { data: kokoroVoices } = useKokoroVoices();
  const kokoro = useMemo(
    () => (Array.isArray(kokoroVoices) ? kokoroVoices : []),
    [kokoroVoices],
  );

  const [speakers, setSpeakers] = useState<Speaker[]>([
    {
      id: "s1",
      name: "Speaker 1",
      voice: "preset:af_heart",
      color: PALETTE[0],
    },
    {
      id: "s2",
      name: "Speaker 2",
      voice: "preset:am_michael",
      color: PALETTE[1],
    },
  ]);
  const [activeId, setActiveId] = useState("s1");
  const [lines, setLines] = useState<Line[]>([
    { id: uid(), speakerId: "s1", text: "" },
  ]);
  const [state, setState] = useState<GenerateState>("idle");
  const [genError, setGenError] = useState<string | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const controlRef = useRef<WaveformControls | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(
    () => () => {
      abort.current?.abort();
    },
    [],
  );

  const speakerById = (id: string) => speakers.find((s) => s.id === id);
  const voiceLabel = (key: string) => {
    if (key.startsWith("cloned:"))
      return cloned.find((v) => v.id === key.slice(7))?.name ?? "Saved voice";
    return kokoro.find((v) => v.id === key.slice(7))?.name ?? key.slice(7);
  };

  const addSpeaker = () => {
    if (speakers.length >= MAX_SPEAKERS) return;
    const color =
      PALETTE.find((c) => !speakers.some((s) => s.color === c)) ?? PALETTE[0];
    const used = new Set(speakers.map((s) => s.voice));
    const free = kokoro.find((v) => !used.has(`preset:${v.id}`)) ?? kokoro[0];
    const sp: Speaker = {
      id: uid(),
      name: `Speaker ${speakers.length + 1}`,
      voice: `preset:${free?.id ?? "af_heart"}`,
      color,
    };
    setSpeakers([...speakers, sp]);
    setActiveId(sp.id);
  };

  const removeSpeaker = (id: string) => {
    if (speakers.length <= 1) return;
    const rest = speakers.filter((s) => s.id !== id);
    setSpeakers(rest);
    setLines((ls) =>
      ls.map((l) => (l.speakerId === id ? { ...l, speakerId: rest[0].id } : l)),
    );
    if (activeId === id) setActiveId(rest[0].id);
  };

  const updateSpeaker = (id: string, patch: Partial<Speaker>) =>
    setSpeakers((ss) => ss.map((s) => (s.id === id ? { ...s, ...patch } : s)));

  const move = (idx: number, dir: -1 | 1) => {
    const j = idx + dir;
    if (j < 0 || j >= lines.length) return;
    const next = [...lines];
    [next[idx], next[j]] = [next[j], next[idx]];
    setLines(next);
  };

  const filled = lines.filter((l) => l.text.trim());
  const active = speakerById(activeId) ?? speakers[0];

  const generate = async () => {
    if (!filled.length) return;
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setState("working");
    setGenError(null);
    try {
      const body = {
        lines: filled.map((l) => {
          const sp = speakerById(l.speakerId) ?? speakers[0];
          const cloned_ = sp.voice.startsWith("cloned:");
          return {
            text: l.text.trim(),
            speaker: sp.voice.slice(sp.voice.indexOf(":") + 1),
            type: cloned_ ? "cloned" : "preset",
          };
        }),
      };
      const out = await generateBlob(
        "/api/generate-dialogue",
        body,
        ctl.signal,
      );
      setBlob(out);
      setState("done");
      toast.success("Dialogue ready.");
    } catch (e) {
      // Stop is not a failure: no toast, no error row. A newer request replacing this one leaves the state to that one.
      const stopped =
        ctl.signal.aborted ||
        (e as { code?: string })?.code === "aborted" ||
        (e as { name?: string })?.name === "AbortError";
      if (abort.current === ctl) setState("idle");
      if (!stopped) {
        const message = errorMessage(
          e,
          "Dialogue generation failed. Try again.",
        );
        setGenError(message);
        toast.error(message);
      }
    }
  };

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_22rem]">
      <section
        aria-label="Dialogue script"
        className="min-w-0 space-y-5 rounded-2xl border border-border bg-bg-surface p-5"
      >
        {/* Speakers */}
        <div>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-medium text-[var(--color-text-primary)]">
              Speakers
            </h2>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={addSpeaker}
              disabled={speakers.length >= MAX_SPEAKERS}
              leftIcon={<Plus className="h-3.5 w-3.5" />}
            >
              Add speaker
            </Button>
          </div>
          <Reorder.Group
            axis="x"
            values={speakers}
            onReorder={setSpeakers}
            className="flex flex-wrap gap-2"
          >
            {speakers.map((s) => (
              <Reorder.Item
                key={s.id}
                value={s}
                className={`flex cursor-grab items-center gap-2 rounded-full border py-1 pl-3 pr-1.5 text-sm active:cursor-grabbing ${
                  s.id === activeId
                    ? "bg-[hsl(var(--au-text-primary)/0.08)]"
                    : ""
                }`}
                style={{ borderColor: s.color }}
              >
                <button
                  type="button"
                  onClick={() => setActiveId(s.id)}
                  aria-pressed={s.id === activeId}
                  className="flex items-center gap-2 text-[var(--color-text-primary)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ background: s.color }}
                    aria-hidden="true"
                  />
                  {s.name}
                  <span className="hidden text-xs text-[var(--color-text-secondary)] sm:inline">
                    {voiceLabel(s.voice)}
                  </span>
                </button>
                {speakers.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeSpeaker(s.id)}
                    aria-label={`Remove ${s.name}`}
                    className="grid h-6 w-6 place-items-center rounded-full text-[var(--color-text-secondary)] hover:bg-[hsl(var(--au-text-primary)/0.1)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </Reorder.Item>
            ))}
          </Reorder.Group>

          {/* Selected speaker editor */}
          <div className="mt-3 grid gap-2 rounded-xl border border-[var(--color-border)] p-3 sm:grid-cols-2">
            <div>
              <label
                htmlFor="sp-name"
                className="mb-1 block text-xs text-[var(--color-text-secondary)]"
              >
                Name
              </label>
              <input
                id="sp-name"
                className={field}
                value={active.name}
                maxLength={24}
                onChange={(e) =>
                  updateSpeaker(active.id, { name: e.target.value })
                }
              />
            </div>
            <div>
              <label
                htmlFor="sp-voice"
                className="mb-1 block text-xs text-[var(--color-text-secondary)]"
              >
                Voice
              </label>
              <select
                id="sp-voice"
                className={field}
                value={active.voice}
                onChange={(e) =>
                  updateSpeaker(active.id, { voice: e.target.value })
                }
              >
                <optgroup label="Preset voices">
                  {kokoro.length === 0 && (
                    <option value={active.voice}>
                      {voiceLabel(active.voice)}
                    </option>
                  )}
                  {kokoro.map((v) => (
                    <option key={v.id} value={`preset:${v.id}`}>
                      {v.name}
                      {v.language ? ` - ${v.language}` : ""}
                      {v.gender ? ` (${v.gender})` : ""}
                    </option>
                  ))}
                </optgroup>
                {cloned.length > 0 && (
                  <optgroup label="Your cloned voices">
                    {cloned.map((v) => (
                      <option key={v.id} value={`cloned:${v.id}`}>
                        {v.name}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </div>
          </div>
        </div>

        {/* Lines */}
        <div>
          <h2 className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">
            Lines
          </h2>
          <Reorder.Group
            axis="y"
            values={lines}
            onReorder={setLines}
            className="space-y-2"
          >
            {lines.map((l, i) => (
              <LineRow
                key={l.id}
                line={l}
                index={i}
                total={lines.length}
                speakers={speakers}
                color={speakerById(l.speakerId)?.color ?? PALETTE[0]}
                onChange={(patch) =>
                  setLines((ls) =>
                    ls.map((x) => (x.id === l.id ? { ...x, ...patch } : x)),
                  )
                }
                onRemove={() =>
                  setLines((ls) =>
                    ls.length > 1 ? ls.filter((x) => x.id !== l.id) : ls,
                  )
                }
                onMove={(d) => move(i, d)}
              />
            ))}
          </Reorder.Group>
          <Button
            type="button"
            variant="outline"
            className="mt-2 w-full border-dashed"
            onClick={() =>
              setLines((ls) => [
                ...ls,
                { id: uid(), speakerId: activeId, text: "" },
              ])
            }
            leftIcon={<Plus className="h-4 w-4" />}
          >
            Add line for {active.name}
          </Button>
        </div>

        <GenerateButton
          state={state}
          error={genError}
          onStop={() => abort.current?.abort()}
          idleLabel={`Generate dialogue (${filled.length} ${filled.length === 1 ? "line" : "lines"})`}
          disabled={!filled.length}
          disabledReason="Type a line for at least one speaker first."
          onGenerate={generate}
          onPlay={() => controlRef.current?.play()}
        />
      </section>

      <section
        aria-label="Dialogue output"
        className="min-w-0 self-start xl:sticky xl:top-6"
      >
        <div className="rounded-2xl border border-border bg-bg-surface p-5">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Output
          </p>
          {blob ? (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            >
              <WaveformPlayer
                src={blob}
                height={110}
                autoPlay
                controlRef={controlRef}
                filename="bonclay-dialogue.wav"
              />
              <p className="mt-2 text-sm text-[var(--color-text-secondary)]">
                Lines are stitched in order with a short pause between turns.
              </p>
            </motion.div>
          ) : (
            <EmptyState
              className="min-h-[16rem]"
              icon={<MessagesSquare aria-hidden="true" />}
              title="Your dialogue appears here"
              description="Add speakers, write their lines, drag to reorder, then generate one stitched recording."
            />
          )}
        </div>
      </section>
    </div>
  );
}

function LineRow({
  line,
  index,
  total,
  speakers,
  color,
  onChange,
  onRemove,
  onMove,
}: {
  line: Line;
  index: number;
  total: number;
  speakers: Speaker[];
  color: string;
  onChange: (p: Partial<Line>) => void;
  onRemove: () => void;
  onMove: (d: -1 | 1) => void;
}) {
  const controls = useDragControls();
  const btn =
    "grid h-8 w-8 place-items-center rounded-md text-[var(--color-text-secondary)] hover:bg-[hsl(var(--au-text-primary)/0.08)] disabled:opacity-30 focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]";
  return (
    <Reorder.Item
      value={line}
      dragListener={false}
      dragControls={controls}
      className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-2.5"
    >
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onPointerDown={(e) => controls.start(e)}
          aria-label={`Drag line ${index + 1} to reorder`}
          className={`${btn} cursor-grab touch-none active:cursor-grabbing`}
        >
          <GripVertical className="h-4 w-4" />
        </button>
        <span
          className="h-3 w-3 shrink-0 rounded-full"
          style={{ background: color }}
          aria-hidden="true"
        />
        <select
          aria-label={`Speaker for line ${index + 1}`}
          value={line.speakerId}
          onChange={(e) => onChange({ speakerId: e.target.value })}
          className="h-8 min-w-0 flex-1 rounded-md border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-2 text-sm text-[var(--color-text-primary)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
        >
          {speakers.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          className={btn}
          onClick={() => onMove(-1)}
          disabled={index === 0}
          aria-label="Move line up"
        >
          <ArrowUp className="h-4 w-4" />
        </button>
        <button
          type="button"
          className={btn}
          onClick={() => onMove(1)}
          disabled={index === total - 1}
          aria-label="Move line down"
        >
          <ArrowDown className="h-4 w-4" />
        </button>
        <button
          type="button"
          className={btn}
          onClick={onRemove}
          disabled={total === 1}
          aria-label="Delete line"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
      <textarea
        aria-label={`Text for line ${index + 1}`}
        value={line.text}
        onChange={(e) => onChange({ text: e.target.value.slice(0, 500) })}
        rows={2}
        placeholder="What does this speaker say?"
        className="mt-2 block w-full resize-y rounded-md bg-transparent px-1 text-sm leading-relaxed text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)] focus:outline-none"
      />
    </Reorder.Item>
  );
}
