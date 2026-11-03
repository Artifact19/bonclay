/** Response of the public POST /api/verify. */
export type Verdict = "made_with_bonclay" | "watermark_only" | "not_detected";

export interface PublicReceipt {
  receipt_id: string;
  kind: "generation" | "agent_reply";
  created_at: string;
  engine: string | null;
  voice_id: string | null;
  voice_name: string | null;
  consent_id: string | null;
  consent_verified: boolean | null;
  signature_valid: boolean;
  /** When the voice owner withdrew consent (or the voice was deleted after its retention period); null/absent while
   *  the voice still exists. Read from a withdrawal log: the signed receipt itself never changes. */
  voice_withdrawn_at?: string | null;
  /** "preset_voice" | "active" | "unverified" | "withdrawn" | "retention_ended" (see docs/RECEIPT_SPEC.md). */
  voice_consent_status?: string;
  /** "withdrawn" (the owner withdrew consent) or "retention_expired" (deleted when its retention period ended). */
  voice_removed_reason?: "withdrawn" | "retention_expired" | null;
}

export interface VerifyResponse {
  watermark: { detected: boolean; probability: number; reason?: string };
  match: {
    exact: boolean;
    receipt: PublicReceipt | null;
    signature_valid: boolean;
    candidates?: PublicReceipt[];
  };
  verdict: Verdict;
}
