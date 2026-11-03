"use client";

import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { AlertCircle, ArrowLeft, ArrowRight, Check, Loader2, Mic, ShieldCheck, Square, Upload, X } from "lucide-react";
import { toast } from "sonner";
import WaveformPlayer from "@/components/audio/WaveformPlayer";
import { Button, TaskProgressBar, useEstimatedProgress } from "@/components/ui";
import { Modal } from "./Modal";
import { ConsentNotice } from "./ConsentNotice";
import { RecordingScript } from "./RecordingScript";
import { VoiceThumb } from "./VoiceThumb";
import { readAudioInfo, toWav, type AudioInfo } from "./audioUtils";
import { cloneVoice, fetchConsentChallenge, rememberDuration, transcribe, type ConsentChallenge } from "./voiceApi";
import { ApiError } from "@/lib/api";
import {
  ALLOWED_EXTENSIONS,
  LANGUAGES,
  engineLabel,
  formatBytes,
  formatDuration,
  validateAudioFile,
} from "./validate";

const INDIC = new Set(["hi", "mr", "bn", "gu", "kn", "ml", "or", "pa", "ta", "te", "as"]);
const STEPS = ["Sample", "Transcript", "Review"] as const;

/** Sentence used only when the server has no challenge endpoint (an older backend). */
const consentPhraseFor = (date: Date) =>
  `I consent to creating an AI version of my voice for the BonClay demonstration on ${date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  })}.`;

interface Props {
  open: boolean;
  onClose: () => void;
  onCreated: (result: { id: string; engine: string; name: string; language: string }) => void;
}

interface Sample {
  blob: Blob;
  filename: string;
  info: AudioInfo | null;
}

export function CloneWizard({ open, onClose, onCreated }: Props) {
  const [step, setStep] = useState(0);
  const [sample, setSample] = useState<Sample | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const [transcript, setTranscript] = useState("");
  const [skipTranscript, setSkipTranscript] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [transcribeError, setTranscribeError] = useState<string | null>(null);
  const transcribedFor = useRef<Blob | null>(null);

  const [name, setName] = useState("");
  const [language, setLanguage] = useState("en");
  const [consentAudio, setConsentAudio] = useState<Blob | null>(null);
  const [consentRecording, setConsentRecording] = useState(false);
  const [consentError, setConsentError] = useState<string | null>(null);
  const consentRecorder = useRef<MediaRecorder | null>(null);
  const consentChunks = useRef<Blob[]>([]);
  const [challenge, setChallenge] = useState<ConsentChallenge | null>(null);
  const [challengeLoading, setChallengeLoading] = useState(false);
  const [challengeError, setChallengeError] = useState<string | null>(null);
  const [challengeExpired, setChallengeExpired] = useState(false);
  const [challengeName, setChallengeName] = useState(""); // the name the current sentence was issued for
  const [challengeTick, setChallengeTick] = useState(0); // bump to ask for a new sentence
  const challengeFor = useRef<{ name: string; tick: number } | null>(null);
  const [legacyPhrase] = useState(() => consentPhraseFor(new Date()));
  const consentPhrase = challenge?.text ?? legacyPhrase;
  const [saving, setSaving] = useState(false);
  const [progress, setProgress] = useState(0);
  const [saveError, setSaveError] = useState<string | null>(null);

  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const abort = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const stopStream = () => {
    recorder.current?.stream.getTracks().forEach((t) => t.stop());
    consentRecorder.current?.stream.getTracks().forEach((t) => t.stop());
    if (timer.current) clearInterval(timer.current);
  };

  const reset = useCallback(() => {
    setStep(0);
    setSample(null);
    setFileError(null);
    setTranscript("");
    setSkipTranscript(false);
    setTranscribeError(null);
    transcribedFor.current = null;
    setName("");
    setLanguage("en");
    setConsentAudio(null);
    setConsentRecording(false);
    setConsentError(null);
    setChallenge(null);
    setChallengeLoading(false);
    setChallengeError(null);
    setChallengeExpired(false);
    challengeFor.current = null;
    setChallengeName("");
    setSaving(false);
    setProgress(0);
    setSaveError(null);
    setRecording(false);
    setElapsed(0);
  }, []);

  useEffect(() => {
    if (!open) {
      stopStream();
      abort.current?.abort();
      reset();
    }
  }, [open, reset]);

  useEffect(() => () => {
    stopStream();
    abort.current?.abort();
  }, []);

  /* ---------- Step 1: sample ---------- */

  const acceptFile = async (file: File) => {
    const err = validateAudioFile(file);
    if (err) {
      setFileError(err);
      return;
    }
    setFileError(null);
    const info = await readAudioInfo(file);
    setSample({ blob: file, filename: file.name, info });
    if (!name) setName(file.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim().slice(0, 60));
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) acceptFile(f);
  };

  const startRecording = async () => {
    setFileError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setFileError("Recording is not supported in this browser. Upload a file instead.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      recorder.current = rec;
      chunks.current = [];
      rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
      rec.onstop = async () => {
        const raw = new Blob(chunks.current, { type: rec.mimeType || "audio/webm" });
        const wav = await toWav(raw);
        if (!wav) {
          setFileError("Could not process the recording. Upload a file instead.");
          return;
        }
        if (wav.size > 50 * 1024 * 1024) {
          setFileError("Recording is larger than 50 MB. Record a shorter sample.");
          return;
        }
        const info = await readAudioInfo(wav);
        setSample({ blob: wav, filename: "recording.wav", info });
      };
      rec.start();
      setRecording(true);
      setElapsed(0);
      timer.current = setInterval(() => setElapsed((s) => s + 1), 1000);
    } catch {
      setFileError("Microphone access was blocked. Allow it in your browser, or upload a file.");
    }
  };

  const stopRecording = () => {
    if (recorder.current && recorder.current.state !== "inactive") recorder.current.stop();
    stopStream();
    setRecording(false);
  };

  /* ---------- Step 2: transcript ---------- */

  const runTranscribe = useCallback(async () => {
    if (!sample) return;
    setTranscribing(true);
    setTranscribeError(null);
    try {
      const res = await transcribe(sample.blob, sample.filename);
      transcribedFor.current = sample.blob;
      setTranscript(res.text ?? "");
      if (res.language && LANGUAGES.some((l) => l.code === res.language)) setLanguage(res.language);
    } catch (e) {
      transcribedFor.current = sample.blob;
      setTranscribeError(e instanceof Error ? e.message : "Transcription failed.");
    } finally {
      setTranscribing(false);
    }
  }, [sample]);

  useEffect(() => {
    if (open && step === 1 && sample && !skipTranscript && transcribedFor.current !== sample.blob) {
      runTranscribe();
    }
  }, [open, step, sample, skipTranscript, runTranscribe]);

  /* ---------- Step 3: consent + save ---------- */

  // A fresh consent sentence (it contains the voice name) each time the Review step opens or the name changes.
  const trimmedName = name.trim();
  useEffect(() => {
    if (!open || step !== 2 || !trimmedName || saving) return;
    const last = challengeFor.current;
    if (last && last.name === trimmedName && last.tick === challengeTick) return;
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setChallengeLoading(true);
      setChallengeError(null);
      try {
        const c = await fetchConsentChallenge(trimmedName, ctrl.signal);
        if (ctrl.signal.aborted) return;
        challengeFor.current = { name: trimmedName, tick: challengeTick };
        setChallengeName(trimmedName);
        setChallenge(c);
        setChallengeExpired(false);
        setConsentAudio(null); // a recording of an earlier sentence no longer counts
        setConsentError(null);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        challengeFor.current = { name: trimmedName, tick: challengeTick };
        setChallengeName(trimmedName);
        setChallenge(null);
        // 404/405: an older server without challenges; the fixed sentence still works there.
        if (!(e instanceof ApiError && (e.status === 404 || e.status === 405))) {
          setChallengeError("Could not get a new consent sentence. Check your connection and try again.");
        }
      } finally {
        if (!ctrl.signal.aborted) setChallengeLoading(false);
      }
    }, 600);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [open, step, trimmedName, challengeTick, saving]);

  // The sentence is only valid for a few minutes: say so when it runs out.
  useEffect(() => {
    if (!challenge) return;
    const ms = new Date(challenge.expires_at).getTime() - Date.now();
    if (!isFinite(ms)) return;
    if (ms <= 0) {
      setChallengeExpired(true);
      return;
    }
    const t = setTimeout(() => setChallengeExpired(true), ms);
    return () => clearTimeout(t);
  }, [challenge]);

  const newSentence = () => {
    setChallengeName("");
    setChallengeTick((t) => t + 1);
  };
  const awaitingChallenge = !!trimmedName && challengeName !== trimmedName;

  const startConsentRecording = async () => {
    setConsentError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setConsentError("Recording is not supported in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      consentRecorder.current = rec;
      consentChunks.current = [];
      rec.ondataavailable = (e) => e.data.size && consentChunks.current.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const raw = new Blob(consentChunks.current, { type: rec.mimeType || "audio/webm" });
        const wav = await toWav(raw);
        if (!wav) {
          setConsentError("Could not process the recording. Please try again.");
          return;
        }
        setConsentAudio(wav);
      };
      rec.start();
      setConsentRecording(true);
    } catch {
      setConsentError("Microphone access was blocked. Allow it in your browser to record consent.");
    }
  };

  const stopConsentRecording = () => {
    if (consentRecorder.current && consentRecorder.current.state !== "inactive") consentRecorder.current.stop();
    setConsentRecording(false);
  };

  const save = async () => {
    if (!sample || !name.trim() || !consentAudio) return;
    const ctrl = new AbortController();
    abort.current = ctrl;
    setSaving(true);
    setSaveError(null);
    setProgress(0);
    try {
      const res = await cloneVoice(
        {
          file: sample.blob,
          filename: sample.filename,
          displayName: name.trim(),
          language,
          transcript: skipTranscript ? "" : transcript.trim(),
          consentAudio,
          consentText: consentPhrase,
          challengeId: challenge?.challenge_id,
        },
        setProgress,
        ctrl.signal
      );
      if (sample.info) rememberDuration(res.id, sample.info.duration);
      const score = res.consent?.verified && res.consent.score != null ? ` Consent verified (${Math.round(res.consent.score * 100)}% match).` : "";
      toast.success(`${name.trim()} is ready`, { description: `Your voice was cloned and added to the library.${score}` });
      onCreated({ id: res.id, engine: res.engine, name: name.trim(), language });
    } catch (e) {
      if (ctrl.signal.aborted) return;
      if (e instanceof ApiError && e.code === "CONSENT_CHALLENGE_INVALID") {
        setConsentAudio(null);
        setChallengeExpired(true);
        setConsentError("This consent sentence expired. Read the new one and record again.");
        toast.error("Consent sentence expired", { description: "Read the new sentence and re-record." });
        newSentence();
        return;
      }
      if (e instanceof ApiError && e.code === "CONSENT_SPEAKER_MISMATCH") {
        setConsentAudio(null);
        setConsentError(
          "The consent recording and the voice sample sound like different people. Make sure the same person records both the sample and the consent."
        );
        toast.error("Voices do not match", { description: "The same person must record both." });
        return;
      }
      if (e instanceof ApiError && (e.code === "CONSENT_MISMATCH" || e.code === "CONSENT_REQUIRED")) {
        const heard = (e.details as { heard?: string } | undefined)?.heard;
        setConsentAudio(null);
        setConsentError(heard ? `${e.message} We heard: "${heard}"` : e.message);
        toast.error("Consent not verified", { description: "Read the sentence again and re-record." });
        return;
      }
      const msg = e instanceof Error ? e.message : "Could not save the voice.";
      setSaveError(msg);
      toast.error("Voice was not saved", { description: msg });
    } finally {
      setSaving(false);
    }
  };

  const derivedEngine = INDIC.has(language) ? "indic_f5" : "qwen3_0_6b";
  const canNext0 = !!sample && !recording;
  const canSave =
    !!name.trim() && !!consentAudio && !consentRecording && !saving && !awaitingChallenge && !challengeExpired;
  const transcribeProgress = useEstimatedProgress(transcribing, 6000);
  const wizardPct = Math.round(((step + 1) / STEPS.length) * 100);

  return (
    <Modal
      open={open}
      onClose={saving ? () => {} : onClose}
      dismissible={!saving}
      size="lg"
      title="Clone a voice"
      description="Three short steps. A clean 15 to 60 second sample works best."
    >
      <div className="mb-4">
        <div className="mb-1.5 flex items-center justify-between text-xs text-[var(--color-text-secondary)]">
          <span>
            Step {step + 1} of {STEPS.length}: <strong className="font-semibold text-[var(--color-text-primary)]">{STEPS[step]}</strong>
          </span>
          <span className="font-mono font-semibold tabular-nums text-[var(--color-text-primary)]">{wizardPct}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-bg-elevated)]">
          <div
            className="h-full rounded-full bg-[var(--color-accent-primary)] transition-[width] duration-300"
            style={{ width: `${wizardPct}%` }}
          />
        </div>
      </div>
      <ol className="mb-6 flex items-center gap-2" aria-label="Progress">
        {STEPS.map((label, i) => (
          <li key={label} className="flex flex-1 items-center gap-2" aria-current={i === step ? "step" : undefined}>
            <span
              className={`grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-semibold transition-colors ${
                i < step
                  ? "bg-[var(--color-accent-success)] text-white"
                  : i === step
                  ? "bg-[var(--color-accent-primary)] text-[var(--color-on-accent)]"
                  : "bg-[var(--color-bg-elevated)] text-[var(--color-text-secondary)]"
              }`}
            >
              {i < step ? <Check className="h-4 w-4" /> : i + 1}
            </span>
            <span
              className={`hidden text-sm sm:inline ${i === step ? "font-medium text-[var(--color-text-primary)]" : "text-[var(--color-text-secondary)]"}`}
            >
              {label}
            </span>
            {i < STEPS.length - 1 && <span className="h-px flex-1 bg-[var(--color-border)]" aria-hidden="true" />}
          </li>
        ))}
      </ol>

      {step === 0 && (
        <div className="space-y-4">
          <p
            data-testid="consent-upfront"
            className="flex items-start gap-2 rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-3 text-sm text-[var(--color-text-secondary)]"
          >
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-accent-success)]" aria-hidden="true" />
            <span>
              At the end, the person whose voice this is reads one short sentence aloud, so no one&rsquo;s voice can be
              cloned without their say-so.
            </span>
          </p>
          {!sample ? (
            <>
              <RecordingScript recording={recording} />
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                className={`flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-5 py-8 text-center transition-colors ${
                  dragging
                    ? "border-[var(--color-accent-primary)] bg-[hsl(var(--au-accent)/0.08)]"
                    : "border-[var(--color-border)]"
                }`}
              >
                <Upload className="h-7 w-7 text-[var(--color-text-secondary)]" aria-hidden="true" />
                <div>
                  <p className="font-medium text-[var(--color-text-primary)]">Drop an audio file here</p>
                  <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
                    {ALLOWED_EXTENSIONS.join(", ")} up to 50 MB
                  </p>
                </div>
                <input
                  ref={fileInput}
                  type="file"
                  accept={ALLOWED_EXTENSIONS.join(",")}
                  className="sr-only"
                  tabIndex={-1}
                  aria-label="Choose audio file"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) acceptFile(f);
                    e.target.value = "";
                  }}
                />
                <div className="flex flex-wrap justify-center gap-2">
                  <Button data-autofocus onClick={() => fileInput.current?.click()} disabled={recording}>
                    <Upload className="h-4 w-4" />
                    Choose file
                  </Button>
                  {recording ? (
                    <Button className="!border-transparent !bg-[var(--color-accent-error)]" onClick={stopRecording}>
                      <Square className="h-4 w-4 fill-current" />
                      Stop {formatDuration(elapsed) || "0:00"}
                    </Button>
                  ) : (
                    <Button variant="outline" onClick={startRecording}>
                      <Mic className="h-4 w-4" />
                      Record
                    </Button>
                  )}
                </div>
                {recording && (
                  <p role="status" className="flex items-center gap-2 text-sm text-red-400">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                    Recording. Speak naturally, then stop.
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="space-y-3 rounded-2xl border border-[var(--color-border)] p-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate font-medium text-[var(--color-text-primary)]">{sample.filename}</p>
                  <p className="text-sm text-[var(--color-text-secondary)]">
                    {formatBytes(sample.blob.size)}
                    {sample.info ? ` - ${formatDuration(sample.info.duration)}` : ""}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setSample(null);
                    transcribedFor.current = null;
                  }}
                  aria-label="Remove sample"
                  className="grid h-11 w-11 shrink-0 place-items-center rounded-lg text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent)]"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <WaveformPlayer src={sample.blob} hideDownload height={56} aiGenerated={false} />
              {sample.info && sample.info.duration < 5 && (
                <p className="text-sm text-amber-300">Short samples clone poorly. Aim for at least 15 seconds.</p>
              )}
            </div>
          )}
          {fileError && (
            <p role="alert" className="flex items-start gap-2 text-sm text-red-400">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {fileError}
            </p>
          )}
        </div>
      )}

      {step === 1 && (
        <div className="space-y-4">
          <p className="text-sm text-[var(--color-text-secondary)]" data-testid="transcript-explainer">
            The words spoken in your sample help the clone sound natural, so we write them down for you to check.
          </p>
          <label className="flex min-h-11 cursor-pointer items-start gap-3 rounded-xl border border-[var(--color-border)] p-3">
            <input
              type="checkbox"
              checked={skipTranscript}
              onChange={(e) => setSkipTranscript(e.target.checked)}
              className="mt-1 h-4 w-4 accent-[var(--color-accent-primary)]"
            />
            <span>
              <span className="block text-sm font-medium text-[var(--color-text-primary)]">
                Skip the transcript
              </span>
              <span className="block text-sm text-[var(--color-text-secondary)]">
                Faster, but the clone only copies the tone of the voice and may sound less natural.
              </span>
            </span>
          </label>

          {!skipTranscript && (
            <div>
              <div className="mb-2 flex items-center justify-between">
                <label htmlFor="clone-transcript" className="text-sm font-medium text-[var(--color-text-primary)]">
                  What is said in the sample
                </label>
                {!transcribing && (
                  <button
                    type="button"
                    onClick={runTranscribe}
                    className="min-h-11 rounded-lg px-2 text-sm text-[var(--color-accent-secondary)] hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent)]"
                  >
                    Transcribe again
                  </button>
                )}
              </div>
              <div className="relative">
                <textarea
                  id="clone-transcript"
                  rows={6}
                  value={transcript}
                  disabled={transcribing}
                  onChange={(e) => setTranscript(e.target.value)}
                  placeholder={transcribing ? "Listening to your sample" : "Type what is said, or let BonClay transcribe it."}
                  className="w-full resize-y rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-bg))] p-3.5 text-base leading-relaxed text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)]/70 focus:border-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40 disabled:opacity-60"
                />
                {transcribing && (
                  <span className="absolute right-3 top-3 flex items-center gap-2 text-sm text-[var(--color-text-secondary)]" role="status">
                    <Loader2 className="h-4 w-4 animate-spin" /> Transcribing
                  </span>
                )}
              </div>
              <p className="mt-1.5 text-xs text-[var(--color-text-secondary)]">
                BonClay detects the language and writes this for you. Fix any word it misheard.
              </p>
              {transcribing && (
                <div className="mt-2.5">
                  <TaskProgressBar
                    value={transcribeProgress}
                    label="Transcribing audio sample with Whisper"
                    detail="Detecting spoken words and language..."
                    size="sm"
                  />
                </div>
              )}
              {transcribeError && (
                <p role="alert" className="mt-2 flex items-start gap-2 text-sm text-red-400">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  {transcribeError} You can type the transcript yourself or skip it.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {step === 2 && (
        <div className="space-y-4">
          <div className="h-16 overflow-hidden rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-bg))]">
            <VoiceThumb seed={name || "new"} peaks={sample?.info?.peaks} />
          </div>
          <div>
            <label htmlFor="clone-name" className="mb-2 block text-sm font-medium text-[var(--color-text-primary)]">
              Voice name
            </label>
            <input
              id="clone-name"
              data-autofocus
              value={name}
              maxLength={60}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Narrator, warm"
              className="h-11 w-full rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-bg))] px-3.5 text-base text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)]/70 focus:border-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40"
            />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label htmlFor="clone-lang" className="mb-2 block text-sm font-medium text-[var(--color-text-primary)]">
                Language
              </label>
              <select
                id="clone-lang"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="h-11 w-full rounded-xl border border-[var(--color-border)] bg-[hsl(var(--au-bg))] px-3 text-base text-[var(--color-text-primary)] focus:border-[var(--color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/40"
              >
                {LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <span className="mb-2 block text-sm font-medium text-[var(--color-text-primary)]">Engine</span>
              <div className="flex h-11 items-center rounded-xl border border-[var(--color-border)] bg-[var(--color-bg-elevated)] px-3.5 text-base text-[var(--color-text-primary)]">
                {engineLabel(derivedEngine)}
                <span className="ml-2 text-xs text-[var(--color-text-secondary)]">chosen for this language</span>
              </div>
            </div>
          </div>
<div className="space-y-3 rounded-xl border border-[var(--color-border)] p-3">
            <div>
              <p className="text-sm font-medium text-[var(--color-text-primary)]">Speaker consent</p>
              <p className="text-sm text-[var(--color-text-secondary)]">
                The person whose voice this is reads the sentence below aloud. BonClay transcribes it, checks it
                matches and checks it is the same voice as the sample before any voice is cloned.
              </p>
            </div>
            <ConsentNotice retentionDays={challenge?.retention_days} contact={challenge?.contact} />
            <div className="rounded-lg bg-[hsl(var(--au-bg))] p-3">
              {challengeLoading && !challenge ? (
                <p role="status" className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
                  <Loader2 className="h-4 w-4 animate-spin" /> Getting your consent sentence
                </p>
              ) : (
                <p
                  className={`font-mono text-sm leading-relaxed text-[var(--color-text-primary)] ${challengeExpired ? "opacity-50" : ""}`}
                >
                  &ldquo;{consentPhrase}&rdquo;
                </p>
              )}
              {challenge && (
                <p className="mt-2 text-xs text-[var(--color-text-secondary)]">
                  {challengeExpired ? "This sentence expired. " : "Valid for 15 minutes. "}
                  <button
                    type="button"
                    onClick={newSentence}
                    disabled={saving || challengeLoading || consentRecording}
                    className="underline underline-offset-2 hover:text-[var(--color-text-primary)] disabled:opacity-50"
                  >
                    {challengeExpired ? "Get a new sentence" : "Get a different sentence"}
                  </button>
                </p>
              )}
              {challengeError && (
                <p role="alert" className="mt-2 text-xs text-red-400">
                  {challengeError}{" "}
                  <button type="button" onClick={newSentence} className="underline underline-offset-2">
                    Try again
                  </button>
                </p>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {consentRecording ? (
                <Button className="!border-transparent !bg-[var(--color-accent-error)]" onClick={stopConsentRecording}>
                  <Square className="h-4 w-4 fill-current" />
                  Stop
                </Button>
              ) : (
                <Button
                  variant="outline"
                  onClick={startConsentRecording}
                  disabled={saving || awaitingChallenge || challengeExpired}
                >
                  <Mic className="h-4 w-4" />
                  {consentAudio ? "Re-record consent" : "Record consent"}
                </Button>
              )}
              {consentRecording && (
                <span role="status" className="flex items-center gap-2 text-sm text-red-400">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                  Recording, read the sentence now
                </span>
              )}
              {consentAudio && !consentRecording && (
                <span className="flex items-center gap-1.5 text-sm text-[var(--color-accent-success)]">
                  <Check className="h-4 w-4" /> Consent recorded
                </span>
              )}
            </div>
            {consentAudio && !consentRecording && <WaveformPlayer src={consentAudio} hideDownload height={40} aiGenerated={false} />}
            {consentError && (
              <p role="alert" className="flex items-start gap-2 text-sm text-red-400">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                {consentError}
              </p>
            )}
          </div>

          {saving && (
            <TaskProgressBar
              value={progress}
              label={progress < 100 ? "Uploading voice sample & consent" : "Building voice profile & verifying speaker consent"}
              detail={progress < 100 ? "Sending audio to local BonClay backend..." : "Checking voice match and computing speaker embedding..."}
              tone="accent"
            />
          )}
          {saveError && (
            <p role="alert" className="flex items-start gap-2 text-sm text-red-400">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              {saveError}
            </p>
          )}
        </div>
      )}

      <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-between">
        <Button
          variant="ghost"
          onClick={() => (step === 0 ? onClose() : setStep(step - 1))}
          disabled={saving}
        >
          {step === 0 ? (
            "Cancel"
          ) : (
            <>
              <ArrowLeft className="h-4 w-4" /> Back
            </>
          )}
        </Button>
        {step < 2 ? (
          <Button
           
            onClick={() => setStep(step + 1)}
            disabled={step === 0 ? !canNext0 : transcribing}
          >
            Continue <ArrowRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button onClick={save} disabled={!canSave}>
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            {saving ? "Cloning" : saveError ? "Try again" : "Clone voice"}
          </Button>
        )}
      </div>
    </Modal>
  );
}
