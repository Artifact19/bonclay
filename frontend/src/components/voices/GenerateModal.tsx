"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Sparkles, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import WaveformPlayer from "@/components/audio/WaveformPlayer";
import { Button } from "@/components/ui";
import { Modal } from "./Modal";
import { generateSpeech, type VoiceView } from "./voiceApi";
import { MAX_GENERATE_CHARS, languageLabel } from "./validate";

interface Props {
  voice: VoiceView | null;
  onClose: () => void;
  /** Start with this line already in the box (used by "Preview"). */
  initialText?: string;
  /** Preview mode: a short sample line, worded as a preview rather than a full script. */
  preview?: boolean;
}

export function GenerateModal({ voice, onClose, initialText = "", preview = false }: Props) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audio, setAudio] = useState<Blob | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Reset whenever a different voice is opened.
  const voiceId = voice?.id;
  useEffect(() => {
    setText(initialText);
    setAudio(null);
    setError(null);
    setBusy(false);
    setSlow(false);
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voiceId]);

  const close = () => {
    abortRef.current?.abort();
    onClose();
  };

  const remaining = MAX_GENERATE_CHARS - text.length;
  const canGenerate = !!text.trim() && !busy && remaining >= 0;

  const run = async () => {
    if (!voice || !canGenerate) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setBusy(true);
    setError(null);
    setAudio(null);
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), 2500);
    try {
      const blob = await generateSpeech(voice, text.trim(), ctrl.signal);
      setAudio(blob);
      toast.success(`Speech generated with ${voice.name}`);
    } catch (e) {
      if (ctrl.signal.aborted) return;
      const msg = e instanceof Error ? e.message : "Generation failed.";
      setError(msg);
      toast.error("Could not generate speech", { description: msg });
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  };

  const counterTone =
    remaining < 0 ? "text-red-400" : remaining < 200 ? "text-amber-300" : "text-[var(--color-text-secondary)]";

  return (
    <Modal
      open={!!voice}
      onClose={close}
      size="lg"
      title={voice ? (preview ? `Preview ${voice.name}` : `Speak as ${voice.name}`) : "Generate speech"}
      description={
        voice
          ? preview
            ? `${languageLabel(voice.language)}. Press the button to hear this line. The first one can take a minute while the voice loads.`
            : languageLabel(voice.language)
          : undefined
      }
    >
      <div className="space-y-4">
        <div>
          <label htmlFor="gen-text" className="mb-2 block text-sm font-medium text-[var(--color-text-primary)]">
            What should this voice say?
          </label>
          <textarea
            id="gen-text"
            data-autofocus
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
            }}
            rows={6}
            aria-describedby="gen-count"
            placeholder="Type or paste your script here."
            className="w-full resize-y rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-bg))] p-3.5 text-base leading-relaxed text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)]/70 focus:border-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40"
          />
          <div className="mt-1.5 flex items-center justify-between text-xs">
            <span className="text-[var(--color-text-secondary)]">{preview ? "You can change the line" : "Ctrl+Enter to generate"}</span>
            <span id="gen-count" aria-live="polite" className={`tabular-nums ${counterTone}`}>
              {text.length.toLocaleString()} / {MAX_GENERATE_CHARS.toLocaleString()}
            </span>
          </div>
        </div>

        {error && (
          <div role="alert" className="flex items-start gap-2 rounded-xl border border-red-400/25 bg-red-400/10 p-3 text-sm text-red-300">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <Button variant="ghost" onClick={close}>
            Close
          </Button>
          <Button onClick={run} disabled={!canGenerate} aria-busy={busy}>
            {busy ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                {slow ? "Loading model, one moment" : "Generating"}
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                {preview ? (audio ? "Play again" : "Play preview") : audio ? "Generate again" : "Generate"}
              </>
            )}
          </Button>
        </div>

        {audio && (
          <div className="rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-bg))] p-4">
            <WaveformPlayer
              src={audio}
              filename={`${(voice?.name ?? "voice").replace(/[^\w-]+/g, "_")}.wav`}
              autoPlay
            />
          </div>
        )}
      </div>
    </Modal>
  );
}
