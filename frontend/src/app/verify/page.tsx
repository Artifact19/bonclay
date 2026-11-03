"use client";

import Link from "next/link";
import { useCallback, useRef, useState, type DragEvent } from "react";
import { ArrowUpRight, FileAudio, LoaderCircle, Upload } from "lucide-react";
import { Logo } from "@/components/Logo";
import { siteFont, siteSerif } from "@/components/site/siteFont";
import { VerifyResult } from "@/components/verify/VerifyResult";
import type { VerifyResponse } from "@/components/verify/types";
import { TaskProgressBar, useEstimatedProgress } from "@/components/ui/TaskProgressBar";
import { API_URL, ApiError, normalizeError } from "@/lib/apiBase";

const MAX_BYTES = 20 * 1024 * 1024;
const EXTENSIONS = [".wav", ".mp3", ".m4a", ".webm", ".flac", ".ogg"];

function sizeLabel(bytes: number): string {
  return bytes >= 1024 * 1024 ? `${(bytes / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Public: no sign-in, no Authorization header. The backend rate-limits this endpoint. */
async function verifyClip(file: File): Promise<VerifyResponse> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api/verify`, { method: "POST", body: form });
  } catch {
    throw new ApiError("Could not reach the BonClay server. Check that it is running and try again.", 0, "network_error");
  }
  if (!res.ok) {
    const err = await normalizeError(res);
    if (res.status === 429) throw new ApiError("Too many checks in a short time. Wait a minute and try again.", 429, err.code);
    throw err;
  }
  return (await res.json()) as VerifyResponse;
}

export default function VerifyPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const verifyPct = useEstimatedProgress(busy, 5, !!result);

  const check = useCallback(async (f: File) => {
    setResult(null);
    setError(null);
    const lower = f.name.toLowerCase();
    if (!EXTENSIONS.some((e) => lower.endsWith(e))) {
      setFile(f);
      setError("Upload a wav, mp3, m4a or webm clip.");
      return;
    }
    if (f.size > MAX_BYTES) {
      setFile(f);
      setError(`That file is ${sizeLabel(f.size)}. The limit is 20 MB.`);
      return;
    }
    setFile(f);
    setBusy(true);
    try {
      setResult(await verifyClip(f));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not check this clip.");
    } finally {
      setBusy(false);
    }
  }, []);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && !busy) void check(f);
  };

  return (
    <div className={`site ${siteFont.variable} ${siteSerif.variable} min-h-screen`}>
      <header className="s-container flex h-16 items-center gap-6">
        <Link href="/" aria-label="BonClay home" className="flex items-center">
          <Logo size={30} />
        </Link>
        <Link
          href="/dashboard"
          prefetch={true}
          className="group ml-auto inline-flex h-10 items-center gap-1.5 rounded-full bg-[var(--s-ink)] pl-4 pr-3.5 s-small font-medium text-white transition-transform duration-200 active:scale-[0.97]"
        >
          Open studio
          <ArrowUpRight className="h-4 w-4 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5" aria-hidden />
        </Link>
      </header>

      <section className="mx-auto max-w-[47.5rem] px-[var(--s-gutter)] pb-24 pt-10 md:pt-16">
        <h1 className="s-title">Verify a clip</h1>
        <p className="lede mt-4">
          Every clip BonClay makes carries an inaudible watermark and a signed receipt. Upload audio to see whether it came from here.
        </p>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`mt-8 rounded-[var(--s-r-card)] border-2 border-dashed px-6 py-12 text-center transition-colors ${
            dragging ? "border-[var(--s-ink)] bg-[var(--s-paper)]" : "border-[var(--s-bar)] bg-[var(--s-paper)]/60"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.webm,.flac,.ogg"
            className="sr-only"
            aria-label="Audio file to verify"
            tabIndex={-1}
            onChange={(e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (f) void check(f);
            }}
          />
          {busy ? (
            <div className="mx-auto flex max-w-sm flex-col items-center gap-4" role="status">
              <LoaderCircle className="h-8 w-8 animate-spin text-[var(--s-ink-2)]" aria-hidden />
              <p className="s-body text-[var(--s-ink-2)]">Checking {file?.name}...</p>
              <TaskProgressBar
                value={verifyPct}
                label="Scanning watermark & receipt"
                active
                testId="verify-progress"
              />
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <span className="grid h-12 w-12 place-items-center rounded-full bg-[var(--s-ground)]">
                {file ? <FileAudio className="h-6 w-6" aria-hidden /> : <Upload className="h-6 w-6" aria-hidden />}
              </span>
              <p className="s-body text-[var(--s-ink-2)]">
                {file ? (
                  <>
                    <span className="font-medium text-[var(--s-ink)]">{file.name}</span> ({sizeLabel(file.size)})
                  </>
                ) : (
                  "Drag an audio file here"
                )}
              </p>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="inline-flex h-11 items-center rounded-full bg-[var(--s-ink)] px-6 s-body font-medium text-white transition-transform duration-200 active:scale-[0.97]"
              >
                {file ? "Check another clip" : "Choose a file"}
              </button>
              <p className="s-small text-[var(--s-mute)]">wav, mp3, m4a or webm, up to 20 MB. The file is checked and not kept.</p>
            </div>
          )}
        </div>

        <div aria-live="polite" className="mt-6">
          {error && (
            <p role="alert" className="rounded-xl border border-[#E8B4AF] bg-[#FDF1F0] px-4 py-3 s-body text-[#8C1D16]">
              {error}
            </p>
          )}
          {result && !busy && <VerifyResult result={result} />}
        </div>

        <p className="mt-10 max-w-[64ch] s-small leading-relaxed text-[var(--s-mute)]">
          The watermark is read by AudioSeal, which flags audio that probably carries it, so a result is evidence, not proof. It usually survives compression and resampling, but heavy editing can remove it, and a clip with no watermark could still come from somewhere else.
        </p>
      </section>
    </div>
  );
}
