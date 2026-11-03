/**
 * Plain-language notice shown in the clone wizard before consent is recorded.
 *
 * Keep NOTICE_VERSION equal to NOTICE_VERSION in backend/services/consent_record.py (a backend test checks this).
 * Bump both when the wording changes: the version is stored in each voice's consent record.
 * Not legal advice: the operator should have the wording checked for the laws that apply to them.
 * TODO: add a Hindi version (and other Indian languages) of this notice.
 */
export const NOTICE_VERSION = "2026-10-15";

export interface NoticePoint {
  title: string;
  text: string;
}

export interface NoticeFacts {
  /** How long a voice is kept (server setting VOICE_RETENTION_DAYS). */
  retentionDays: number;
  /** Who to contact about a voice; falls back to the studio operator. */
  contact: string;
}

export const DEFAULT_RETENTION_DAYS = 365;
export const DEFAULT_CONTACT = "the person who runs this BonClay studio";

export function noticePoints({ retentionDays, contact }: NoticeFacts): NoticePoint[] {
  return [
    {
      title: "What we collect",
      text: "Your voice sample, a recording of you reading the consent sentence, that sentence as text, and a numeric fingerprint (embedding) of your voice.",
    },
    {
      title: "Why",
      text: "To make an AI version of your voice for use in BonClay, and to show that you agreed. We use it for nothing else.",
    },
    {
      title: "How long we keep it",
      text: `${retentionDays} days from today, or until you withdraw, whichever comes first.`,
    },
    {
      title: "How to withdraw",
      text: 'At any time, choose "Withdraw consent and delete" on the voice in the Voices page. That deletes the sample, the embedding and the consent recording. Clips already made are kept and marked as made with a voice whose owner withdrew consent.',
    },
    {
      title: "Contact",
      text: `Questions or requests: ${contact}.`,
    },
  ];
}
