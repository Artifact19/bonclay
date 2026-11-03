"use client";

import { useEffect, useState } from "react";
import { Download, Loader2, ShieldCheck, Trash2 } from "lucide-react";
import { Button } from "@/components/ui";
import { Modal } from "./Modal";
import { downloadConsentRecord, fetchConsentRecord, type ConsentRecord, type VoiceView } from "./voiceApi";

function formatDateTime(iso: string | null): string {
  if (!iso) return "Not recorded";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function formatDay(iso: string | null): string {
  if (!iso) return "Not known";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { dateStyle: "medium" });
}

const pct = (x: number | null) => (x == null ? null : `${Math.round(x * 100)}%`);

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-0.5 py-2.5 sm:grid-cols-[11rem_1fr] sm:gap-4">
      <dt className="text-xs font-medium uppercase tracking-wide text-[var(--color-text-secondary)]">{label}</dt>
      <dd className="min-w-0 break-words text-sm text-[var(--color-text-primary)]">{children}</dd>
    </div>
  );
}

interface PanelProps {
  voice: VoiceView;
  /** One-click withdrawal: the parent confirms and deletes. Omit to hide the button. */
  onWithdraw?: (voice: VoiceView) => void;
}

/** The itemised consent record of one voice, with a JSON download and the withdraw action.
 *  Give it `key={voice.id}` so it starts fresh for another voice. */
export function ConsentRecordPanel({ voice, onWithdraw }: PanelProps) {
  const [record, setRecord] = useState<ConsentRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchConsentRecord(voice.id, ctrl.signal)
      .then(setRecord)
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setError(e instanceof Error ? e.message : "Could not load the consent record.");
      });
    return () => ctrl.abort();
  }, [voice.id]);

  if (error) {
    return (
      <p role="alert" className="text-sm text-red-400">
        {error}
      </p>
    );
  }
  if (!record) {
    return (
      <p role="status" className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading the consent record
      </p>
    );
  }

  const transcript = pct(record.transcript_match_score);
  return (
    <section aria-label="Consent record" data-testid="consent-record">
      {record.legacy && (
        <p className="mb-2 rounded-lg border border-[var(--color-border)] p-3 text-sm text-[var(--color-text-secondary)]">
          This voice was cloned before itemised records were kept, so some details (such as the sentence that was read)
          were never stored and show as not recorded.
        </p>
      )}
      <dl className="divide-y divide-[var(--color-border)]">
        <Row label="Voice">{record.voice_name ?? voice.name}</Row>
        <Row label="Purpose">{record.purpose}</Row>
        <Row label="What is stored">
          <ul className="list-disc space-y-0.5 pl-5">
            {record.data_stored.map((d) => (
              <li key={d}>{d}</li>
            ))}
          </ul>
        </Row>
        <Row label="Sentence read">
          {record.challenge_sentence ? <span className="font-mono text-[13px]">&ldquo;{record.challenge_sentence}&rdquo;</span> : "Not recorded"}
        </Row>
        <Row label="Code used">{record.challenge_code ?? "Not recorded"}</Row>
        <Row label="Transcript match">
          {transcript ? `${transcript}${record.transcript_verified ? " (verified)" : ""}` : "Not checked (speech recognition was unavailable)"}
        </Row>
        <Row label="Speaker match">
          {record.speaker_match_score != null
            ? `${pct(record.speaker_match_score)}${record.speaker_match_threshold != null ? ` (needed ${pct(record.speaker_match_threshold)})` : ""}`
            : "Not checked"}
        </Row>
        <Row label="Consent given">{formatDateTime(record.consented_at)}</Row>
        <Row label="Kept until">
          <span data-testid="kept-until">{formatDay(record.retain_until)}</span>{" "}
          <span className="text-[var(--color-text-secondary)]">({record.retention_days} days, or until you withdraw)</span>
        </Row>
        <Row label="Notice version">{record.notice_version ?? "Before notices were versioned"}</Row>
        <Row label="How to withdraw">{record.how_to_withdraw}</Row>
        <Row label="Contact">{record.contact}</Row>
      </dl>
      <div className="mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:justify-between">
        {onWithdraw ? (
          <Button
            className="!border-transparent !bg-[var(--color-accent-error)]"
            onClick={() => onWithdraw(voice)}
          >
            <Trash2 className="h-4 w-4" />
            Withdraw consent and delete
          </Button>
        ) : (
          <span />
        )}
        <Button variant="outline" onClick={() => downloadConsentRecord(record)}>
          <Download className="h-4 w-4" />
          Download as JSON
        </Button>
      </div>
      <p className="mt-3 text-xs text-[var(--color-text-secondary)]">
        This record shows what BonClay stored and when. It is not legal advice.
      </p>
    </section>
  );
}

interface DialogProps {
  voice: VoiceView | null;
  onClose: () => void;
  onWithdraw: (voice: VoiceView) => void;
}

export function ConsentRecordDialog({ voice, onClose, onWithdraw }: DialogProps) {
  return (
    <Modal
      open={!!voice}
      onClose={onClose}
      size="lg"
      title={voice ? `Consent record: ${voice.name}` : "Consent record"}
      description="What you agreed to when this voice was cloned."
    >
      {voice && (
        <>
          <p className="mb-3 flex items-center gap-1.5 text-sm text-[var(--color-accent-success)]">
            <ShieldCheck className="h-4 w-4" aria-hidden="true" /> Stored with the voice. Deleted with it.
          </p>
          <ConsentRecordPanel key={voice.id} voice={voice} onWithdraw={onWithdraw} />
        </>
      )}
    </Modal>
  );
}
