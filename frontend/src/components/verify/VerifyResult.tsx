import { CircleHelp, ShieldCheck, ShieldQuestion } from "lucide-react";
import type { PublicReceipt, Verdict, VerifyResponse } from "./types";

const VERDICTS: Record<Verdict, { title: string; body: string; tone: string; Icon: typeof ShieldCheck }> = {
  made_with_bonclay: {
    title: "Made with BonClay",
    body: "This clip carries BonClay's watermark and matches a receipt signed by this server.",
    tone: "bg-[var(--s-marigold)] text-[var(--s-marigold-ink)]",
    Icon: ShieldCheck,
  },
  watermark_only: {
    title: "Watermark found",
    body: "This clip carries an BonClay watermark, but no matching receipt was found on this server.",
    tone: "bg-[var(--s-marigold)] text-[var(--s-marigold-ink)]",
    Icon: ShieldQuestion,
  },
  not_detected: {
    title: "Not detected",
    body: "No BonClay watermark was found in this clip. That does not show the clip is genuine: it may come from another tool, or have been edited.",
    tone: "bg-[var(--s-line)] text-[var(--s-ink)]",
    Icon: CircleHelp,
  },
};

const KIND_LABEL: Record<PublicReceipt["kind"], string> = {
  generation: "Studio generation",
  agent_reply: "Agent reply",
};

const ENGINE_LABEL: Record<string, string> = {
  kokoro: "Kokoro",
  qwen3_0_6b: "Qwen3-TTS",
  qwen3_1_7b: "Qwen3-TTS",
  indic_f5: "IndicF5",
  indic_speak: "Indic-Speak",
  chatterbox_ml: "Chatterbox",
  dhvaani: "DhVaani (ARTPARK-IISc)",
  dialogue: "Dialogue",
};

export function formatCreated(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function consentLabel(r: PublicReceipt): { text: string; good: boolean } {
  if (r.voice_withdrawn_at && r.consent_id) {
    return { text: r.consent_verified === true ? "Consent verified when made, now withdrawn" : "Consent now withdrawn", good: false };
  }
  if (r.consent_verified === true) return { text: "Consent verified", good: true };
  if (r.consent_id) return { text: "Consent recorded, not verified", good: false };
  if (r.voice_id) return { text: "No consent record", good: false };
  return { text: "Preset voice (no cloning)", good: false };
}

/** The sentence shown for a clip whose voice has since been removed (null while the voice still exists). */
export function withdrawalNote(r: PublicReceipt): string | null {
  if (!r.voice_withdrawn_at) return null;
  const when = formatDay(r.voice_withdrawn_at);
  return r.voice_removed_reason === "retention_expired"
    ? `This voice was deleted on ${when} when its retention period ended.`
    : `The voice owner withdrew consent on ${when}.`;
}

function formatDay(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString(undefined, { dateStyle: "medium" });
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-3 sm:flex-row sm:items-baseline sm:gap-6">
      <dt className="w-40 shrink-0 text-[13px] text-[var(--s-mute)]">{label}</dt>
      <dd className="min-w-0 text-[15px] text-[var(--s-ink)]">{children}</dd>
    </div>
  );
}

function Receipt({ receipt, exact }: { receipt: PublicReceipt; exact: boolean }) {
  const consent = consentLabel(receipt);
  const withdrawn = withdrawalNote(receipt);
  return (
    <dl className="divide-y divide-[var(--s-line)]" data-testid="receipt">
      <Row label="Created">{formatCreated(receipt.created_at)}</Row>
      <Row label="Type">{KIND_LABEL[receipt.kind] ?? receipt.kind}</Row>
      <Row label="Engine">{(receipt.engine && ENGINE_LABEL[receipt.engine]) || receipt.engine || "Unknown"}</Row>
      <Row label="Voice">{receipt.voice_name || "Unknown"}</Row>
      <Row label="Consent">
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-[13px] font-medium ${
            consent.good ? "bg-[var(--s-ink)] text-white" : "border border-[var(--s-bar)] text-[var(--s-ink-2)]"
          }`}
        >
          {consent.text}
        </span>
      </Row>
      {withdrawn && (
        <Row label="Voice status">
          <span
            className="inline-flex items-center rounded-full border border-[#B3261E] px-2.5 py-0.5 text-[13px] font-medium text-[#B3261E]"
            data-testid="voice-withdrawn"
          >
            Voice withdrawn
          </span>
          <p className="mt-1.5 text-[14px] text-[var(--s-ink-2)]">
            {withdrawn} The receipt is kept as a record that the clip was made while consent stood.
          </p>
        </Row>
      )}
      <Row label="Match">{exact ? "Exact copy of the original file" : "Found through the watermark"}</Row>
      <Row label="Signature">
        <span className={receipt.signature_valid ? "" : "font-medium text-[#B3261E]"}>
          {receipt.signature_valid ? "Valid: the receipt has not been altered" : "Invalid: the receipt does not verify"}
        </span>
      </Row>
      <Row label="Receipt">
        <code className="break-all font-mono text-[13px] text-[var(--s-ink-2)]">{receipt.receipt_id}</code>
      </Row>
    </dl>
  );
}

export function VerifyResult({ result }: { result: VerifyResponse }) {
  const v = VERDICTS[result.verdict];
  const pct = Math.round(Math.max(0, Math.min(1, result.watermark.probability)) * 100);
  const candidates = result.match.candidates ?? [];
  return (
    <article className="overflow-hidden rounded-[20px] border border-[var(--s-line)] bg-[var(--s-paper)]" data-testid="verify-result">
      <header className={`flex items-start gap-4 px-6 py-7 sm:px-8 ${v.tone}`}>
        <v.Icon className="mt-1 h-9 w-9 shrink-0" aria-hidden />
        <div>
          <h2 className="display text-[clamp(1.9rem,4vw,2.75rem)]">{v.title}</h2>
          <p className="mt-2 max-w-[52ch] text-[15px] leading-relaxed opacity-90">{v.body}</p>
        </div>
      </header>

      <div className="px-6 py-6 sm:px-8">
        <div className="flex items-baseline justify-between gap-4">
          <p className="text-[14px] font-medium">Watermark confidence</p>
          <p className="text-[14px] tabular-nums text-[var(--s-ink-2)]">{pct}%</p>
        </div>
        <div
          className="mt-2 h-2 overflow-hidden rounded-full bg-[var(--s-line)]"
          role="meter"
          aria-label="Watermark confidence"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
        >
          <div className="h-full rounded-full bg-[var(--s-ink)]" style={{ width: `${pct}%` }} />
        </div>
        {result.watermark.reason && <p className="mt-2 text-[13px] text-[var(--s-mute)]">{result.watermark.reason}</p>}

        {result.match.receipt && (
          <div className="mt-6">
            <Receipt receipt={result.match.receipt} exact={result.match.exact} />
          </div>
        )}

        {!result.match.receipt && candidates.length > 1 && (
          <div className="mt-6" data-testid="candidates">
            <p className="text-[14px] font-medium">The watermark matches {candidates.length} possible generations</p>
            <p className="mt-1 text-[13px] text-[var(--s-mute)]">
              The watermark holds 16 bits, so unrelated clips can occasionally share it. Compare these with where you found the clip.
            </p>
            <ul className="mt-3 divide-y divide-[var(--s-line)] rounded-xl border border-[var(--s-line)]">
              {candidates.map((c) => (
                <li key={c.receipt_id} className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 text-[14px]">
                  <span>
                    {c.voice_name || "Unknown voice"}
                    {c.voice_withdrawn_at ? " (voice withdrawn)" : ""}
                  </span>
                  <span className="text-[var(--s-mute)]">{formatCreated(c.created_at)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </article>
  );
}
