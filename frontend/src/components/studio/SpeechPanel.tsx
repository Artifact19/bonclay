"use client";

import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { toast } from "sonner";
import { AudioLines, Keyboard, Languages, Wand2 } from "lucide-react";
import { coverageText, indianCoverage } from "@/components/voices/coverage";
import { API_URL, api } from "@/lib/api";
import { AudioQueue } from "@/lib/audioQueue";
import { SentenceQueue, type StreamedSentence } from "@/lib/sentenceQueue";
import {
  useIndicSpeak,
  useKokoroVoices,
  useModelStatus,
  useVoices,
} from "@/hooks/useApi";
import type { IndicSpeakInfo, ModelStatus } from "@/types/api";
import { useGenerationStore } from "@/stores";
import { langAttr } from "@/lib/lang";
import {
  WaveformPlayer,
  type WaveformControls,
} from "@/components/audio/WaveformPlayer";
import { Badge, Button, Card, EmptyState } from "@/components/ui";
import { GenerateButton, type GenerateState } from "./GenerateButton";
import { DevanagariKeyboard } from "./DevanagariKeyboard";
import {
  DHVAANI_CREDIT,
  DHVAANI_LABEL,
  FALLBACK_SAMPLES,
  INDIC_CODES,
  MAX_CHARS,
  absoluteAudioUrl,
  errorMessage,
  fetchClipBlob,
  languageName,
  fetchSamples,
  generateBlob,
  type SamplePrompt,
  type StreamedGeneration,
} from "./studioApi";
import styles from "./studio.module.css";

type EngineId = "kokoro" | "indic" | "clone" | "dhvaani";

const ENGINES: { id: EngineId; label: string; hint: string }[] = [
  { id: "kokoro", label: "Kokoro", hint: "Fast presets, English and more" },
  {
    id: "indic",
    label: "Indic-Speak",
    hint: "22 Indian languages, preset voices",
  },
  { id: "clone", label: "Cloned voice", hint: "Speak in a voice you saved" },
  // Listed only when DhVaani is set up on this server (see dhvaaniAvailable).
  {
    id: "dhvaani",
    label: DHVAANI_LABEL,
    hint: "Your saved voice in Hindi, Marathi, Tamil and more",
  },
];

/** The credit ARTPARK-IISc asks for, wherever DhVaani audio is offered. */
function DhvaaniCredit() {
  return (
    <p
      data-testid="dhvaani-credit"
      className="text-xs text-[var(--color-text-secondary)]"
    >
      {DHVAANI_CREDIT}
    </p>
  );
}

/** Measured: the Indic-Speak worker took 16 s to load on an Apple M4 (docs/RESULTS.md); the first load after a restart can take longer. */
export const INDIC_LOAD_SECONDS = 20;

const field =
  "w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-primary)] px-3 py-2.5 text-sm text-[var(--color-text-primary)] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-[var(--color-accent-cyan)]";

/** The Indic Open Model License requires this notice wherever Indic-Speak output reaches a user interface. */
function IndicAttribution({ info }: { info: IndicSpeakInfo }) {
  return (
    <p className="text-xs text-[var(--color-text-secondary)]">
      {info.attribution}{" "}
      <a
        className="underline underline-offset-2"
        href={info.model_url}
        target="_blank"
        rel="noreferrer"
      >
        Model card
      </a>
    </p>
  );
}

export function SpeechPanel() {
  const {
    data: clonedVoices,
    error: voicesError,
    isLoading: voicesLoading,
  } = useVoices();
  const { data: kokoroVoices, isLoading: kokoroLoading } = useKokoroVoices();
  const { data: indicInfo } = useIndicSpeak();
  const { data: modelStatus } = useModelStatus();
  const {
    currentAudio,
    setCurrentAudio,
    startGeneration,
    finishGeneration,
    isGenerating,
  } = useGenerationStore();

  const [engine, setEngine] = useState<EngineId>("kokoro");
  const [manualEngine, setManualEngine] = useState(false);
  const [autoRoutedFrom, setAutoRoutedFrom] = useState<EngineId | null>(null);
  const [kokoroVoice, setKokoroVoice] = useState("");
  const [kokoroTouched, setKokoroTouched] = useState(false);
  const [clonedId, setClonedId] = useState("");
  const [indicVoice, setIndicVoice] = useState("");
  const [indicTouched, setIndicTouched] = useState(false);
  const [speed, setSpeed] = useState(1);
  const searchParams = useSearchParams();
  const [text, setText] = useState("");
  const [typing, setTyping] = useState(false);
  const [detected, setDetected] = useState<string | null>(null);
  // Hindi typed in Roman letters ("kal subah aana"): the Hindi voices need Devanagari, so offer to convert it.
  const [hinglish, setHinglish] = useState(false);
  // The text before and after the last Hinglish conversion, so it can be undone until the user edits the result.
  const [converted, setConverted] = useState<{
    before: string;
    after: string;
  } | null>(null);
  const [showKeys, setShowKeys] = useState(false);
  const [transliterating, setTransliterating] = useState(false);
  const [samples, setSamples] = useState<SamplePrompt[]>(FALLBACK_SAMPLES);
  const [done, setDone] = useState(false);
  // True while the Indic-Speak worker (3.8B parameters) is being loaded for this request, before any speech is made.
  const [indicLoading, setIndicLoading] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioEngine, setAudioEngine] = useState<EngineId | null>(null);
  // True when the audio on screen was spoken by DhVaani (a saved voice in an Indian language): its credit goes with it.
  const [audioDhvaani, setAudioDhvaani] = useState(false);
  // A cloned-voice render streams sentence by sentence: what is being rendered now, and (once it is done) whether the
  // clip on screen was already heard while it rendered (then the waveform must not play it again) and its receipt.
  const [streamLabel, setStreamLabel] = useState("");
  const [audioStreamed, setAudioStreamed] = useState(false);
  const [receiptId, setReceiptId] = useState<string | null>(null);
  // Set when a paste, a sample or a conversion was longer than the limit and had to be cut; cleared on the next edit.
  const [trimmed, setTrimmed] = useState<{
    kept: number;
    total: number;
  } | null>(null);
  // Why the last attempt failed (shown beside Generate with a Retry); cleared when a new attempt starts or the script changes.
  const [genError, setGenError] = useState<string | null>(null);
  // The language of the script the audio on screen was made from, for the `lang` of the text shown under it.
  const [audioLang, setAudioLang] = useState<string | undefined>(undefined);

  const taRef = useRef<HTMLTextAreaElement>(null);
  const controlRef = useRef<WaveformControls | null>(null);
  const typingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const loadPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const playerRef = useRef<AudioQueue | null>(null);

  const cloned = useMemo(
    () => (Array.isArray(clonedVoices) ? clonedVoices : []),
    [clonedVoices],
  );
  const kokoro = useMemo(
    () => (Array.isArray(kokoroVoices) ? kokoroVoices : []),
    [kokoroVoices],
  );
  const indicVoices = useMemo(() => indicInfo?.voices ?? [], [indicInfo]);
  const indicAvailable = indicInfo?.available === true;
  // DhVaani clones a saved voice into Indian languages; it is an optional install (./bonclay setup --dhvaani).
  const dhvaaniAvailable =
    modelStatus?.engines?.some(
      (e) => e.id === "dhvaani" && e.available === true,
    ) === true;
  // Hindi may be set to clone with Chatterbox instead (INDIC_CLONE_ENGINE=chatterbox); then DhVaani never speaks Hindi.
  const hindiByChatterbox =
    modelStatus?.indic_clone?.hindi_engine === "chatterbox_ml";
  const spokenByDhvaani = (code: string) =>
    INDIC_CODES.has(code) && !(code === "hi" && hindiByChatterbox);
  const usesSavedVoice = engine === "clone" || engine === "dhvaani";
  // Two plain choices up front; the exact engines sit under "Advanced".
  const voiceKind: "standard" | "mine" = usesSavedVoice ? "mine" : "standard";
  const coverage = useMemo(
    () => indianCoverage(modelStatus?.engines),
    [modelStatus],
  );
  const engineTiles = useMemo(
    () => ENGINES.filter((e) => e.id !== "dhvaani" || dhvaaniAvailable),
    [dhvaaniAvailable],
  );
  // Indic-Speak speaks a few hundred characters at a time (it is a slow, large model); the other engines take MAX_CHARS.
  const limit =
    engine === "indic" && indicInfo?.max_chars
      ? Math.min(MAX_CHARS, indicInfo.max_chars)
      : MAX_CHARS;
  const indicGroups = useMemo(() => {
    const code =
      detected && indicVoices.some((v) => v.language_code === detected)
        ? detected
        : "hi";
    return {
      code,
      mine: indicVoices.filter((v) => v.language_code === code),
      others: indicVoices.filter((v) => v.language_code !== code),
    };
  }, [indicVoices, detected]);

  // Kokoro's English voices garble Indian-language text; its Hindi voice is the one that can pronounce it. Until the
  // user picks a voice themselves, follow the language of the text.
  useEffect(() => {
    if (!kokoro.length) return;
    if (kokoroTouched && kokoroVoice) return;
    const hindi = kokoro.find((v) => v.language_code === "h");
    const want =
      detected && INDIC_CODES.has(detected) && hindi ? hindi.id : kokoro[0].id;
    if (want !== kokoroVoice) setKokoroVoice(want);
  }, [kokoro, detected, kokoroTouched, kokoroVoice]);
  useEffect(() => {
    if (!clonedId && cloned.length) setClonedId(cloned[0].id);
  }, [cloned, clonedId]);
  useEffect(() => {
    if (indicTouched || !indicVoices.length) return;
    const pick =
      indicVoices.find(
        (v) => v.language_code === indicGroups.code && v.gender === "female",
      ) ?? indicVoices[0];
    if (pick && pick.id !== indicVoice) setIndicVoice(pick.id);
  }, [indicVoices, indicGroups.code, indicTouched, indicVoice]);

  // Deep link from the voices page: /dashboard/studio?voice=<id>
  useEffect(() => {
    const v = searchParams.get("voice");
    if (v) {
      setClonedId(v);
      setEngine("clone");
      setManualEngine(true);
    }
  }, [searchParams]);

  // Onboarding sample prompts (falls back to local set if the route is unavailable).
  useEffect(() => {
    let live = true;
    fetchSamples()
      .then((s) => {
        if (live && s.length) setSamples(s);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  // Debounced language detection.
  useEffect(() => {
    const t = text.trim();
    if (t.length < 4) {
      setDetected(null);
      setHinglish(false);
      return;
    }
    const ctl = new AbortController();
    const id = setTimeout(async () => {
      try {
        const r = await api.post<{ language?: string; hinglish?: boolean }>(
          "/api/indic/detect",
          { text: t },
          { signal: ctl.signal, retries: 0 },
        );
        if (r?.language) setDetected(r.language);
        setHinglish(r?.hinglish === true);
      } catch {
        /* detection is advisory; stay silent */
      }
    }, 600);
    return () => {
      clearTimeout(id);
      ctl.abort();
    };
  }, [text]);

  // Auto-route to Indic-Speak for Indic text (only when it is set up and the text fits); restore when the user types
  // something else.
  const textLength = text.length;
  useEffect(() => {
    if (!detected || manualEngine) return;
    const fits = textLength <= (indicInfo?.max_chars ?? MAX_CHARS);
    if (
      INDIC_CODES.has(detected) &&
      engine !== "indic" &&
      indicAvailable &&
      fits
    ) {
      setAutoRoutedFrom(engine);
      setEngine("indic");
      toast.info(
        `${languageName(detected)} detected. Using the Indian-language voices.`,
      );
    } else if (
      !INDIC_CODES.has(detected) &&
      autoRoutedFrom &&
      engine === "indic"
    ) {
      setEngine(autoRoutedFrom);
      setAutoRoutedFrom(null);
    }
  }, [
    detected,
    manualEngine,
    engine,
    autoRoutedFrom,
    indicAvailable,
    indicInfo,
    textLength,
  ]);

  // Revoke object URLs we create.
  const prevUrl = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      if (loadPollRef.current) clearInterval(loadPollRef.current);
      abortRef.current?.abort();
      playerRef.current?.stop();
    };
  }, []);

  const onTextChange = (v: string) => {
    // Never cut silently: say how much was kept, until the next edit.
    if (v.length > limit) {
      setText(v.slice(0, limit));
      setTrimmed({ kept: limit, total: v.length });
    } else {
      setText(v);
      setTrimmed(null);
    }
    setGenError(null);
    setDone(false);
    setTyping(true);
    if (typingTimer.current) clearTimeout(typingTimer.current);
    typingTimer.current = setTimeout(() => setTyping(false), 900);
  };

  const insertAtCaret = useCallback(
    (s: string) => {
      const ta = taRef.current;
      const start = ta?.selectionStart ?? text.length;
      const end = ta?.selectionEnd ?? text.length;
      onTextChange(text.slice(0, start) + s + text.slice(end));
      requestAnimationFrame(() => {
        ta?.focus();
        ta?.setSelectionRange(start + s.length, start + s.length);
      });
    },
    [text, limit],
  );

  const backspace = useCallback(() => {
    const ta = taRef.current;
    const start = ta?.selectionStart ?? text.length;
    const end = ta?.selectionEnd ?? text.length;
    if (start === end && start === 0) return;
    const from = start === end ? start - 1 : start;
    onTextChange(text.slice(0, from) + text.slice(end));
    requestAnimationFrame(() => {
      ta?.focus();
      ta?.setSelectionRange(from, from);
    });
  }, [text]);

  const looksLatin = text.trim().length > 0 && !/[^\u0000-ɏ]/.test(text);

  const transliterate = async (fromHinglish = false) => {
    setTransliterating(true);
    try {
      const before = text;
      const r = await api.post<{ text?: string }>(
        "/api/indic/transliterate",
        { text: before, target: "devanagari" },
        { retries: 0 },
      );
      const out = r?.text;
      if (!out)
        throw new Error("The transliteration service returned no text.");
      onTextChange(out);
      if (fromHinglish) {
        // The Devanagari text is Hindi: show that now (the Hindi voice routing follows `detected`), and keep the
        // original so the conversion can be undone.
        setConverted({ before, after: out.slice(0, limit) });
        setDetected("hi");
        setHinglish(false);
      }
      toast.success("Converted to Devanagari.");
    } catch (e) {
      toast.error(errorMessage(e, "Could not transliterate. Try again."));
    } finally {
      setTransliterating(false);
    }
  };

  const undoConversion = () => {
    if (!converted) return;
    onTextChange(converted.before);
    setConverted(null);
  };

  const chooseEngine = (id: EngineId) => {
    setDone(false);
    setEngine(id);
    setManualEngine(true);
    setAutoRoutedFrom(null);
  };

  const chooseKind = (kind: "standard" | "mine") => {
    if (kind === voiceKind) return;
    setDone(false);
    if (kind === "mine") {
      chooseEngine("clone");
      return;
    }
    // Back to ready-made voices: Kokoro, and the Indian-language route follows the text again (auto routing).
    setEngine("kokoro");
    setManualEngine(false);
    setAutoRoutedFrom(null);
  };

  const needsClone = usesSavedVoice && !clonedId;
  const tooLong = engine === "indic" && text.length > limit;
  const needsIndic = engine === "indic" && (!indicAvailable || !indicVoice);
  const canGenerate =
    text.trim().length > 0 &&
    !needsClone &&
    !tooLong &&
    !needsIndic &&
    !(engine === "kokoro" && !kokoroVoice);

  // Why Generate is off, in plain words (nothing when it is on, or when the message is already shown beside the text).
  const disabledReason = canGenerate
    ? undefined
    : text.trim().length === 0
      ? "Type some text first."
      : needsClone
        ? cloned.length
          ? "Pick a voice."
          : "Clone a voice first on the Voices page."
        : needsIndic
          ? indicInfo && !indicAvailable
            ? "The Indian-language voices are not set up on this server."
            : "Pick a voice."
          : engine === "kokoro" && !kokoroVoice
            ? "Pick a voice."
            : undefined;

  /**
   * A saved voice speaks through POST /api/voices/generate/stream: each sentence plays as soon as it is rendered while
   * the rest is still being made, and the `done` event carries the whole clip (one receipt) for the waveform and the
   * download. Returns null when the stream fails (an old server, a network error, an `error` event): the caller then
   * makes the clip with the one-response route instead.
   */
  const streamClone = async (form: FormData, signal: AbortSignal) => {
    const queue = new SentenceQueue();
    const player = new AudioQueue();
    playerRef.current = player;
    const result: { done: StreamedGeneration | null } = { done: null };
    setStreamLabel(queue.label());
    try {
      await api.sse("/api/voices/generate/stream", form, {
        signal,
        onEvent: (event, raw) => {
          if (event === "sentence") {
            for (const clip of queue.push(raw as StreamedSentence))
              player.add(absoluteAudioUrl(clip.audio_url));
            setStreamLabel(queue.label());
          } else if (event === "done") {
            result.done = raw as StreamedGeneration;
            queue.finish();
            player.close(); // the clips already queued still play out
          } else if (event === "error") {
            throw new Error(
              (raw as { message?: string })?.message || "The stream failed.",
            );
          }
        },
      });
      const url = result.done?.audio_url;
      if (!url) throw new Error("The stream ended without the finished clip.");
      const blob = await fetchClipBlob(url, signal);
      return { blob, receiptId: result.done?.receipt_id ?? null };
    } catch (e) {
      player.stop();
      if (signal.aborted) throw e;
      return null;
    } finally {
      setStreamLabel("");
    }
  };

  const generate = async () => {
    if (!canGenerate) return;
    abortRef.current?.abort();
    playerRef.current?.stop();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setDone(false);
    setGenError(null);
    // The backend returns the audio in one response, so there is no real percentage: the button shows an honest message
    // and the seconds so far (see GenerateButton) rather than a bar that eases toward 90%.
    startGeneration();

    if (engine === "indic") {
      // Cold start: the worker is not running yet, so the first request waits while it loads. Say so, and clear the
      // note when the status endpoint reports the worker loaded (the request then moves on to making speech).
      const isLoaded = async () => {
        const st = await api.get<ModelStatus>("/api/models/status", {
          retries: 0,
        });
        return (
          st?.engines?.find((e) => e.id === "indic_speak")?.loaded === true
        );
      };
      isLoaded()
        .then((loaded) => {
          if (loaded || ctl.signal.aborted) return;
          setIndicLoading(true);
          loadPollRef.current = setInterval(() => {
            isLoaded()
              .then((now) => {
                if (now) {
                  setIndicLoading(false);
                  if (loadPollRef.current) clearInterval(loadPollRef.current);
                }
              })
              .catch(() => {});
          }, 2000);
        })
        .catch(() => {}); // the note is a courtesy; generation does not depend on it
    }

    try {
      let blob: Blob;
      let streamed: { blob: Blob; receiptId: string | null } | null = null;
      const lang =
        detected && INDIC_CODES.has(detected)
          ? detected
          : engine === "indic" || engine === "dhvaani"
            ? detected || "hi"
            : detected || "en";
      if (engine === "kokoro") {
        blob = await generateBlob(
          "/api/kokoro/generate",
          { voice: kokoroVoice, text, speed },
          ctl.signal,
        );
      } else if (engine === "indic") {
        blob = await generateBlob(
          "/api/indic/generate",
          { text, speaker: indicVoice, language: lang },
          ctl.signal,
        );
      } else {
        const f = new FormData();
        f.append("text", text);
        f.append("voice_id", clonedId);
        f.append("language", lang);
        f.append("mode", "premium");
        streamed = await streamClone(f, ctl.signal);
        blob = streamed
          ? streamed.blob
          : await generateBlob("/api/voices/generate", f, ctl.signal);
      }
      const url = URL.createObjectURL(blob);
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      prevUrl.current = url;
      setAudioBlob(blob);
      setAudioStreamed(streamed !== null);
      setReceiptId(streamed?.receiptId ?? null);
      setAudioEngine(engine);
      setAudioDhvaani(
        dhvaaniAvailable && usesSavedVoice && spokenByDhvaani(lang),
      );
      setAudioLang(hinglish ? langAttr("hinglish") : langAttr(detected));
      setCurrentAudio({
        url,
        text,
        voiceId:
          engine === "kokoro"
            ? kokoroVoice
            : engine === "indic"
              ? indicVoice
              : clonedId,
      });
      setDone(true);
      toast.success("Speech ready.");
    } catch (e) {
      // Stop (or a newer request taking over) is not a failure: no toast, no error row, the button just comes back.
      const stopped =
        ctl.signal.aborted ||
        (e as { code?: string })?.code === "aborted" ||
        (e as { name?: string })?.name === "AbortError";
      if (!stopped) {
        const message = errorMessage(e, "Generation failed. Try again.");
        setGenError(message);
        toast.error(message);
      }
    } finally {
      // A newer request owns the shared state now: leave it alone.
      if (abortRef.current === ctl) {
        if (loadPollRef.current) clearInterval(loadPollRef.current);
        setIndicLoading(false);
        finishGeneration();
      }
    }
  };

  /** Stop: drop the request in flight and silence any sentence already playing. */
  const stopGenerating = () => {
    abortRef.current?.abort();
    playerRef.current?.stop();
  };

  const state: GenerateState = isGenerating
    ? "working"
    : done
      ? "done"
      : "idle";
  const over = text.length >= limit * 0.9;
  // The language of the script as far as we can tell (Roman-letter Hindi is "hi-Latn"); no `lang` until it is known.
  const scriptLang = hinglish ? langAttr("hinglish") : langAttr(detected);

  const visibleSamples = samples.slice(0, 6);

  return (
    <div className="grid gap-5 lg:grid-cols-[18rem_minmax(0,1fr)] xl:grid-cols-[18rem_minmax(0,1fr)_22rem]">
      {/* Script workspace */}
      <Card
        variant="solid"
        padded={false}
        className="flex min-h-[32rem] min-w-0 flex-col lg:order-2"
      >
        <div className="flex flex-1 flex-col">
          <div className="flex items-center justify-between gap-2 border-b border-[var(--color-border)] px-4 py-3">
            <label
              htmlFor="script"
              className="text-sm font-medium text-[var(--color-text-primary)]"
            >
              Script
            </label>
            <div className="flex items-center gap-2">
              <AnimatePresence>
                {detected && (
                  <motion.span
                    key={detected}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Badge tone="primary">
                      <Languages className="h-3 w-3" aria-hidden="true" />
                      {languageName(detected)} detected
                    </Badge>
                  </motion.span>
                )}
              </AnimatePresence>
              <span
                className={`text-xs tabular-nums ${over ? "text-amber-400" : "text-[var(--color-text-secondary)]"}`}
                aria-live="polite"
              >
                {text.length}/{limit}
              </span>
            </div>
          </div>
          <div
            className={`${styles.field} ${typing ? styles.typing : ""} m-3 flex flex-1 flex-col`}
          >
            <div className={`${styles.inner} flex flex-1 flex-col`}>
              <textarea
                id="script"
                ref={taRef}
                value={text}
                onChange={(e) => onTextChange(e.target.value)}
                rows={10}
                lang={scriptLang}
                aria-describedby={trimmed ? "script-trimmed" : undefined}
                placeholder="Type or paste what you want spoken."
                className="block min-h-[18rem] w-full flex-1 resize-none rounded-[13px] bg-transparent p-4 text-base leading-relaxed text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)] focus:outline-none"
              />
            </div>
          </div>

          {trimmed && (
            <p
              id="script-trimmed"
              role="status"
              data-testid="script-trimmed"
              className="mx-4 mt-1 rounded-lg border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-xs text-[var(--color-text-primary)]"
            >
              Kept the first {trimmed.kept.toLocaleString("en-US")} of{" "}
              {trimmed.total.toLocaleString("en-US")} characters. The rest was
              left out.
            </p>
          )}

          <div className="mx-4 mt-1 flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowKeys((v) => !v)}
              aria-pressed={showKeys}
              leftIcon={<Keyboard className="h-3.5 w-3.5" />}
            >
              Devanagari keyboard
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void transliterate()}
              disabled={!looksLatin}
              loading={transliterating}
              title={
                looksLatin
                  ? "Convert Roman letters to Devanagari"
                  : "Type Romanized text, such as namaste, to convert it"
              }
              leftIcon={<Wand2 className="h-3.5 w-3.5" />}
            >
              {transliterating ? "Converting" : "Romanized to Devanagari"}
            </Button>
          </div>

          {hinglish && (
            <div
              role="status"
              data-testid="hinglish-suggestion"
              className="mx-4 mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-[#F0A23A]/40 bg-[#F0A23A]/10 px-3 py-2 text-xs text-[var(--color-text-primary)]"
            >
              <span>
                Looks like Hinglish - convert to Devanagari so the Hindi voice
                reads it right.
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void transliterate(true)}
                loading={transliterating}
                leftIcon={<Wand2 className="h-3.5 w-3.5" />}
              >
                {transliterating ? "Converting" : "Convert to Devanagari"}
              </Button>
            </div>
          )}
          {converted && text === converted.after && (
            <p className="mx-4 mt-2 text-xs text-[var(--color-text-secondary)]">
              Converted to Devanagari.{" "}
              <button
                type="button"
                onClick={undoConversion}
                className="underline underline-offset-2 hover:text-[var(--color-text-primary)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
              >
                Undo
              </button>
            </p>
          )}

          <AnimatePresence initial={false}>
            {showKeys && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
                className="mx-4 mt-2 overflow-hidden"
              >
                <DevanagariKeyboard
                  onInsert={insertAtCaret}
                  onBackspace={backspace}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {!text && (
            <div className="mx-4 mt-3">
              <p className="mb-1.5 text-xs text-[var(--color-text-secondary)]">
                Try a sample
              </p>
              <div className="flex flex-wrap gap-2">
                {visibleSamples.map((s, i) => (
                  <button
                    key={`${s.language}-${i}`}
                    type="button"
                    onClick={() => onTextChange(s.text)}
                    className="max-w-full truncate rounded-full border border-[var(--color-border)] px-3 py-1.5 text-xs text-[var(--color-text-primary)] hover:bg-[hsl(var(--au-text-primary)/0.06)] focus-visible:outline-2 focus-visible:outline-[var(--color-accent-cyan)]"
                    title={s.text}
                  >
                    <span className="mr-1.5 text-[var(--color-text-secondary)]">
                      {languageName(s.language)}
                    </span>
                    {s.text.slice(0, 22)}
                    {s.text.length > 22 ? "..." : ""}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-[var(--color-border)] px-4 py-3">
          <GenerateButton
            state={state}
            disabled={!canGenerate}
            disabledReason={disabledReason}
            workingHint={
              engine === "indic"
                ? "Making audio. This voice can take about a minute."
                : undefined
            }
            error={genError}
            onGenerate={generate}
            onStop={stopGenerating}
            onPlay={() => controlRef.current?.play()}
          />
          <div className="min-w-0 flex-1 basis-56 space-y-1">
            {streamLabel && isGenerating && (
              <p
                role="status"
                data-testid="stream-progress"
                className="text-xs text-[var(--color-text-primary)]"
              >
                {streamLabel}. The first sentence plays as soon as it is ready.
              </p>
            )}
            {tooLong && (
              <p role="alert" className="text-xs text-amber-400">
                Indic-Speak speaks up to {limit} characters at a time. Shorten
                the text to continue.
              </p>
            )}
            {indicLoading && isGenerating && (
              <p
                role="status"
                data-testid="indic-loading"
                className="text-xs text-[var(--color-text-primary)]"
              >
                Loading the Indian-language voices, about {INDIC_LOAD_SECONDS}{" "}
                s. They load when they have not been used for a while; the
                speech is made right after.
              </p>
            )}
            {engine === "indic" && indicAvailable && indicInfo && (
              <div className="space-y-1">
                <p className="text-xs text-[var(--color-text-secondary)]">
                  Indic-Speak runs on this machine. Expect about 35 seconds for
                  a short sentence on an Apple M4 (about 7x slower than real
                  time).
                </p>
                <IndicAttribution info={indicInfo} />
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Voice rail */}
      <section
        aria-label="Speech controls"
        className="min-w-0 self-start lg:order-1 xl:sticky xl:top-6"
      >
        <Card variant="solid" className="space-y-5">
          <p className="text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
            Voice
          </p>
          <fieldset>
            <legend className="mb-2 text-sm font-medium text-[var(--color-text-primary)]">
              Which voice?
            </legend>
            <div className="flex flex-col gap-2">
              {(
                [
                  {
                    id: "standard",
                    label: "Standard voices",
                    hint: "Ready-made voices in English, Hindi and many Indian languages.",
                  },
                  {
                    id: "mine",
                    label: "My voice",
                    hint: coverage
                      ? `A voice you cloned. English, plus ${coverageText(coverage)}.`
                      : "A voice you cloned, speaking the language it was recorded in.",
                  },
                ] as const
              ).map((k) => {
                const on = voiceKind === k.id;
                return (
                  <button
                    key={k.id}
                    type="button"
                    onClick={() => chooseKind(k.id)}
                    aria-pressed={on}
                    className={`flex min-h-[3.75rem] w-full items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)] ${
                      on
                        ? "border-[#F0A23A] bg-[#F0A23A]/15 text-[var(--color-text-primary)]"
                        : "border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[hsl(var(--au-text-primary)/0.05)]"
                    }`}
                  >
                    <span
                      aria-hidden="true"
                      className={`mt-1 grid h-4 w-4 shrink-0 place-items-center rounded-full border ${on ? "border-[#F0A23A]" : "border-[var(--color-border)]"}`}
                    >
                      {on && (
                        <span className="h-2 w-2 rounded-full bg-[#F0A23A]" />
                      )}
                    </span>
                    <span className="min-w-0">
                      <span className="block text-sm font-medium">
                        {k.label}
                      </span>
                      <span className="mt-0.5 block text-xs leading-snug opacity-80">
                        {k.hint}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
            {autoRoutedFrom && engine === "indic" && (
              <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                Using the Indian-language voices (Indic-Speak) because the text
                is {detected ? languageName(detected) : "Indian-language"}. To
                choose differently, open Advanced below.
              </p>
            )}
          </fieldset>

          {/* Voice picker */}
          <div>
            <label
              htmlFor="voice-pick"
              className="mb-2 block text-sm font-medium text-[var(--color-text-primary)]"
            >
              Voice
            </label>
            {engine === "kokoro" && (
              <select
                id="voice-pick"
                className={field}
                value={kokoroVoice}
                onChange={(e) => {
                  setKokoroVoice(e.target.value);
                  setKokoroTouched(true);
                  setDone(false);
                }}
                disabled={kokoroLoading || !kokoro.length}
              >
                {kokoroLoading && <option>Loading voices</option>}
                {!kokoroLoading && !kokoro.length && (
                  <option value="">Kokoro voices are unavailable</option>
                )}
                {kokoro.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name}
                    {v.language ? ` - ${v.language}` : ""}
                    {v.gender ? ` (${v.gender})` : ""}
                  </option>
                ))}
              </select>
            )}
            {engine === "kokoro" &&
              !kokoroTouched &&
              detected &&
              INDIC_CODES.has(detected) && (
                <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                  {`${languageName(detected)} text: using Kokoro's Hindi voice, the only Kokoro voice that can speak it.`}
                </p>
              )}
            {engine === "indic" && (
              <>
                <select
                  id="voice-pick"
                  className={field}
                  value={indicVoice}
                  onChange={(e) => {
                    setIndicVoice(e.target.value);
                    setIndicTouched(true);
                    setDone(false);
                  }}
                  disabled={!indicAvailable || !indicVoices.length}
                >
                  {!indicInfo && <option>Loading voices</option>}
                  {indicInfo && !indicAvailable && (
                    <option value="">
                      Indic-Speak is not set up on this server
                    </option>
                  )}
                  {indicAvailable && (
                    <>
                      {indicGroups.mine.length > 0 && (
                        <optgroup
                          label={`${indicGroups.mine[0].language} voices`}
                        >
                          {indicGroups.mine.map((v) => (
                            <option key={v.id} value={v.id}>
                              {v.name} ({v.gender})
                            </option>
                          ))}
                        </optgroup>
                      )}
                      <optgroup label="Other languages">
                        {indicGroups.others.map((v) => (
                          <option key={v.id} value={v.id}>
                            {v.name} - {v.language} ({v.gender})
                          </option>
                        ))}
                      </optgroup>
                    </>
                  )}
                </select>
                {indicInfo && !indicAvailable && (
                  <p
                    role="alert"
                    className="mt-2 text-xs text-[var(--color-text-secondary)]"
                  >
                    {indicInfo.reason}
                  </p>
                )}
                {indicAvailable && (
                  <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                    Preset voices only, no cloning. Any voice can speak any of
                    the 22 languages.
                  </p>
                )}
              </>
            )}
            {usesSavedVoice && (
              <>
                <select
                  id="voice-pick"
                  className={field}
                  value={clonedId}
                  onChange={(e) => {
                    setClonedId(e.target.value);
                    setDone(false);
                  }}
                  disabled={voicesLoading || !cloned.length}
                >
                  {voicesLoading && <option>Loading voices</option>}
                  {!voicesLoading && !cloned.length && (
                    <option value="">No saved voices yet</option>
                  )}
                  {cloned.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.name}
                    </option>
                  ))}
                </select>
                {voicesError && (
                  <p
                    role="alert"
                    className="mt-2 text-xs text-[var(--color-text-secondary)]"
                  >
                    Could not load your voices: {voicesError.message}
                  </p>
                )}
                {!voicesLoading && !cloned.length && !voicesError && (
                  <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                    Clone a voice first on the{" "}
                    <a
                      className="underline underline-offset-2"
                      href="/dashboard/voices"
                    >
                      Voices page
                    </a>
                    .
                  </p>
                )}
                {dhvaaniAvailable && (
                  <div className="mt-2 space-y-1">
                    <p className="text-xs text-[var(--color-text-secondary)]">
                      {detected === "hi" && hindiByChatterbox
                        ? "Hindi is spoken in your voice by Chatterbox (slower: about 20 to 40 seconds for a short sentence on an Apple M4)."
                        : detected && INDIC_CODES.has(detected)
                          ? `${languageName(detected)} is spoken in your voice by DhVaani (about 2x slower than real time on an Apple M4).`
                          : hindiByChatterbox
                            ? "Hindi is spoken in your voice by Chatterbox; other Indian languages by DhVaani (about 2x slower than real time on an Apple M4)."
                            : "Indian-language text is spoken in your voice by DhVaani (about 2x slower than real time on an Apple M4)."}
                    </p>
                    {!(detected === "hi" && hindiByChatterbox) && (
                      <DhvaaniCredit />
                    )}
                  </div>
                )}
              </>
            )}
          </div>

          <details
            className="group/adv rounded-xl border border-[var(--color-border)]"
            data-testid="engine-advanced"
          >
            <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-2 rounded-xl px-3 text-sm font-medium text-[var(--color-text-primary)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)] [&::-webkit-details-marker]:hidden">
              <span>Advanced: choose the engine yourself</span>
              <span className="text-xs font-normal text-[var(--color-text-secondary)]">
                {engineTiles.find((e) => e.id === engine)?.label}
              </span>
            </summary>
            <div className="space-y-2 px-3 pb-3">
              <p className="text-xs text-[var(--color-text-secondary)]">
                Normally BonClay picks the engine for you, including the
                Indian-language one when your text needs it.
              </p>
              <div className="grid grid-cols-1 gap-2">
                {engineTiles.map((e) => {
                  const on = engine === e.id;
                  return (
                    <button
                      key={e.id}
                      type="button"
                      onClick={() => chooseEngine(e.id)}
                      aria-pressed={on}
                      disabled={
                        e.id === "indic" && !!indicInfo && !indicAvailable
                      }
                      title={
                        e.id === "indic" && indicInfo && !indicAvailable
                          ? (indicInfo.reason ?? undefined)
                          : undefined
                      }
                      className={`min-h-[3.25rem] rounded-xl border px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-accent-cyan)] ${
                        on
                          ? "border-[#F0A23A] bg-[#F0A23A]/15 text-[var(--color-text-primary)]"
                          : "border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[hsl(var(--au-text-primary)/0.05)]"
                      }`}
                    >
                      <span className="block text-sm font-medium">
                        {e.label}
                      </span>
                      <span className="mt-0.5 block text-[11px] leading-snug opacity-80">
                        {e.id === "indic" && indicInfo && !indicAvailable
                          ? "Not set up on this server"
                          : e.hint}
                      </span>
                    </button>
                  );
                })}
              </div>
              {engine === "kokoro" && (
                <div>
                  <label
                    htmlFor="speed"
                    className="mb-2 flex items-center justify-between text-sm font-medium text-[var(--color-text-primary)]"
                  >
                    Speed
                    <span className="tabular-nums text-xs text-[var(--color-text-secondary)]">
                      {speed.toFixed(2)}x
                    </span>
                  </label>
                  <input
                    id="speed"
                    type="range"
                    min={0.5}
                    max={2}
                    step={0.05}
                    value={speed}
                    onChange={(e) => {
                      setSpeed(Number(e.target.value));
                      setDone(false);
                    }}
                    className="w-full accent-[#F0A23A]"
                  />
                </div>
              )}
            </div>
          </details>
        </Card>
      </section>

      {/* Output */}
      <section
        aria-label="Output"
        className="min-w-0 self-start lg:order-3 lg:col-span-2 xl:col-span-1 xl:sticky xl:top-6"
      >
        <Card variant="solid">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
            Output
          </p>
          <div>
            {currentAudio && audioBlob ? (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                className="space-y-3"
              >
                <WaveformPlayer
                  src={audioBlob}
                  height={110}
                  autoPlay={!audioStreamed}
                  controlRef={controlRef}
                  filename="bonclay-speech.wav"
                />
                <p
                  lang={audioLang}
                  data-testid="audio-text"
                  className="line-clamp-3 text-sm text-[var(--color-text-secondary)]"
                >
                  {currentAudio.text}
                </p>
                {audioEngine === "indic" && indicInfo && (
                  <IndicAttribution info={indicInfo} />
                )}
                {audioDhvaani && <DhvaaniCredit />}
                {receiptId && (
                  <p className="text-xs text-[var(--color-text-secondary)]">
                    Watermarked, with a signed{" "}
                    <a
                      className="underline underline-offset-2"
                      href={`${API_URL}/api/receipts/${encodeURIComponent(receiptId)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      receipt
                    </a>
                    .
                  </p>
                )}
              </motion.div>
            ) : (
              <EmptyState
                className="min-h-[16rem]"
                icon={<AudioLines aria-hidden="true" />}
                title="Your audio appears here"
                description="Write a script, choose a voice, and generate. You can scrub, change speed, and download the result."
              />
            )}
          </div>
        </Card>
      </section>
    </div>
  );
}
