// Verify BonClay receipts and the withdrawal log with no server. Node 18+ (node:crypto), no dependencies.
// See docs/RECEIPT_SPEC.md. Browser support for Ed25519 in WebCrypto varies; this file targets Node.
import { createHash, createPublicKey, verify } from "node:crypto";

export const SIGNED_FIELDS = ["version", "receipt_id", "kind", "generation_id", "turn_id", "created_at", "engine",
  "voice_id", "voice_name", "consent_id", "consent_verified", "text_sha256", "audio_sha256", "watermark_message"];
export const LOG_GENESIS = "0".repeat(64);

// JSON with sorted keys, no spaces, non-ASCII kept as is (matches Python json.dumps(sort_keys=True,
// separators=(",", ":"), ensure_ascii=False)).
function dumps(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(dumps).join(",") + "]";
  return "{" + Object.keys(value).sort().map((k) => JSON.stringify(k) + ":" + dumps(value[k])).join(",") + "}";
}

export function canonical(receipt, fields = SIGNED_FIELDS) {
  const payload = {};
  for (const k of fields) payload[k] = receipt[k] === undefined ? null : receipt[k];
  return Buffer.from(dumps(payload), "utf8");
}

function publicKey(b64) {
  // An Ed25519 public key in SPKI form is a fixed 12-byte prefix plus the 32 raw bytes.
  const raw = Buffer.from(b64, "base64");
  const spki = Buffer.concat([Buffer.from("302a300506032b6570032100", "hex"), raw]);
  return createPublicKey({ key: spki, format: "der", type: "spki" });
}

function check(publicB64, signatureB64, data) {
  if (typeof signatureB64 !== "string") return false;
  try { return verify(null, data, publicKey(publicB64), Buffer.from(signatureB64, "base64")); } catch { return false; }
}

export function verifyReceipt(receipt, publicB64, fields = SIGNED_FIELDS) {
  if (!receipt || receipt.version !== 1) return false;
  return check(publicB64, receipt.signature_ed25519, canonical(receipt, fields));
}

export function verifyAudioSha256(receipt, bytes) {
  return Boolean(receipt.audio_sha256) && createHash("sha256").update(bytes).digest("hex") === receipt.audio_sha256;
}

function rowHash(row, prev) {
  const body = { voice_id: row.voice_id ?? null, withdrawn_at: row.withdrawn_at ?? null, reason: row.reason ?? null, prev };
  return createHash("sha256").update(dumps(body), "utf8").digest("hex");
}

export function verifyWithdrawalLog(rows, publicB64) {
  let prev = LOG_GENESIS, signed = 0, legacy = 0, broken = null;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (!row.row_hash) { legacy++; continue; }
    const ok = row.prev === prev && row.sig && rowHash(row, String(row.prev)) === row.row_hash
      && check(publicB64, row.sig, Buffer.from(String(row.row_hash), "ascii"));
    if (!ok) { broken = i; break; }
    prev = row.row_hash; signed++;
  }
  return { ok: broken === null, rows: rows.length, signed_rows: signed, legacy_rows: legacy, broken_at: broken, head: prev };
}
