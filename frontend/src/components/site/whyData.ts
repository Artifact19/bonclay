/** Content for the "Why we built BonClay" section. Every figure below was re-opened at its source on CHECKED_ON and the
 *  number, wording and date confirmed. Anything that could not be confirmed was dropped. Do not add a figure here without
 *  re-checking its source the same way, and keep the citation label in the "Source, Mon YYYY" form. */

import { REPO_PUBLIC, REPO_URL } from "./repo";

export const CHECKED_ON = "17 October 2026";

export interface Stat {
  id: string;
  /** The big text. When `countup` is true it is a plain number the page may animate. */
  numeral: string;
  countup: boolean;
  /** One plain sentence saying what the number means. */
  meaning: string;
  /** Visible citation text, "Source, Mon YYYY". */
  cite: string;
  href: string;
  /** Optional small caveat, shown under the citation. */
  note?: string;
}

export interface Beat {
  id: "language" | "trust" | "cost";
  title: string;
  story: string;
  stats: Stat[];
  /** What BonClay does about it. Only features that exist today. */
  answer: string;
  links?: { label: string; href: string }[];
}

export const BEATS: Beat[] = [
  {
    id: "language",
    title: "India is online in its own languages.",
    story: "Much voice AI is still built English-first. The people it is meant to serve mostly are not.",
    stats: [
      {
        id: "users",
        numeral: "886 million",
        countup: true,
        meaning: "Indians were active internet users in 2024, up 8% on the year before.",
        cite: "IAMAI x Kantar, via PTI, Jan 2025",
        href: "https://www.theweek.in/wire-updates/business/2025/01/16/dcm51-biz-iamai-internet-usage-report.html",
      },
      {
        id: "indic",
        numeral: "98%",
        countup: true,
        meaning: "of those users accessed content in Indic languages.",
        cite: "IAMAI x Kantar, via PTI, Jan 2025",
        href: "https://www.theweek.in/wire-updates/business/2025/01/16/dcm51-biz-iamai-internet-usage-report.html",
      },
    ],
    answer: "BonClay speaks in 22 Indian languages with ready voices, and can speak Hindi and other Indian languages in your own consented voice.",
    links: [{ label: "Hear the languages", href: "/#hear" }],
  },
  {
    id: "trust",
    title: "A voice can now be copied. Trust has to be built in.",
    story: "Cloned voices are used in scams, and India now has rules on labelling synthetic media and on handling personal data.",
    stats: [
      {
        id: "mcafee",
        numeral: "47%",
        countup: true,
        meaning: "of Indian adults said they had met an AI voice scam, or knew someone who had. The global figure was 25%. A 2023 survey of 7,054 adults in 7 countries.",
        cite: "McAfee, via Digital Terminal, May 2023",
        href: "https://digitalterminal.in/tech-companies/mcafee-reveals-47-of-indians-have-faced-ai-voice-scams",
      },
      {
        id: "pindrop",
        numeral: "1,300%",
        countup: true,
        meaning: "Pindrop reports deepfake fraud attempts rose by more than this in 2024, measured across 1.2 billion customer calls.",
        cite: "Pindrop 2025 report, via CIO Influence, Jun 2025",
        href: "https://cioinfluence.com/security/pindrops-2025-voice-intelligence-security-report-reveals-1300-surge-in-deepfake-fraud/",
      },
      {
        id: "itrules",
        numeral: "20 Feb 2026",
        countup: false,
        meaning: "India's amended IT Rules took effect. Synthetic audio and video shared on platforms must carry a prominent label and provenance markers.",
        cite: "IT Rules amendment, via Hogan Lovells, Feb 2026",
        href: "https://www.hlc.com/en/publications/india-introduces-mandatory-labelling-for-ai-and-3hour-takedown-for-illegal-content",
      },
      {
        id: "dpdp",
        numeral: "May 2027",
        countup: false,
        meaning: "The main consent and notice duties of the DPDP Rules begin, 18 months after the Rules were notified on 13 Nov 2025. Voice is personal data.",
        cite: "DPDP Rules 2025, via SS Rana, Nov 2025",
        href: "https://ssrana.in/articles/meity-notifies-final-digital-personal-data-protection-rules-2025/",
        note: "Law firms differ by a day on the exact date.",
      },
    ],
    answer: "BonClay asks for a spoken yes before it clones a voice, adds an inaudible watermark to its audio, and gives every clip a signed receipt that anyone can check.",
    links: [
      { label: "How consent works", href: "/how-it-works#consent" },
      { label: "Verify a clip", href: "/verify" },
    ],
  },
  {
    id: "cost",
    title: "Hosted voice agents are billed by the minute.",
    story: "That is fine for some teams. A clinic or a college office should also have the option to own the whole thing.",
    stats: [
      {
        id: "vapi",
        numeral: "$82 to $129",
        countup: false,
        meaning: "per 1,000 minutes, Vapi's own estimate: about $0.08 to $0.13 a minute with hosting, speech recognition, a language model and a voice. Telephony is extra.",
        cite: "Vapi pricing, Oct 2026",
        href: "https://vapi.ai/pricing",
        note: `Prices in $, as shown on the page on ${CHECKED_ON}.`,
      },
      {
        id: "retell",
        numeral: "$0.11",
        countup: true,
        meaning: "a minute, Retell's own estimator total for a sample agent. Its price list adds $0.015 a minute for telephony.",
        cite: "Retell pricing, Oct 2026",
        href: "https://www.retellai.com/pricing",
        note: `Prices in $, as shown on the page on ${CHECKED_ON}.`,
      },
    ],
    answer: `BonClay is free to run${REPO_PUBLIC ? ", with the source on GitHub" : ". The code is MIT-licensed and shared on request"}. It runs on one laptop, your data stays on your machine, and cloud models are optional. It adds no per-minute fee of its own. Phone calls still go through a phone provider.`,
    links: REPO_PUBLIC ? [{ label: "Source on GitHub", href: REPO_URL }] : [],
  },
];

export interface UseCase {
  id: string;
  title: string;
  line: string;
  href: string;
  cta: string;
}

/** Four use cases, each tied to something that exists today. */
export const USE_CASES: UseCase[] = [
  {
    id: "clinic",
    title: "Clinic reception",
    line: "Answers timings and fees, and books, moves or cancels appointments.",
    href: "/dashboard/studio/agents",
    cta: "Open Agents",
  },
  {
    id: "hod",
    title: "College HOD office",
    line: "Takes meeting requests, waits for your approval, then adds them to Google Calendar.",
    href: "/dashboard/approvals",
    cta: "See approvals",
  },
  {
    id: "helpdesk",
    title: "Helpdesk from your PDFs",
    line: "Answers from your own documents and says so when it does not know.",
    href: "/dashboard/knowledge",
    cta: "Add documents",
  },
  {
    id: "creators",
    title: "Creators",
    line: "Your own voice in Indian languages, only after a spoken consent.",
    href: "/#languages",
    cta: "Hear the languages",
  },
];
