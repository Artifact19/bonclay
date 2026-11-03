/** Content for "From your voice to a signed answer": one example question followed through eight steps.
 *
 *  Every timing is a median over 30 spoken questions (15 English, 15 Hindi), streaming, local models only, on an Apple M4
 *  with 16 GB. Source: docs/RESULTS.md, "Current numbers (16 October 2026)", "Speed, 30 spoken questions" (the table at
 *  lines 26 to 33), raw file backend/bench/results/latency_stream-20261016-053741.json. proofData.ts quotes the same table
 *  (MEASURED_SPEED); a test checks the two agree. The clock starts when the audio request is sent, not when the caller
 *  stops speaking.
 *
 *  The example is made up: a clinic agent, a caller who does not exist, a clinic PDF that does not exist. */

export const TIMING = {
  /** First audio, median. At the start of a call this is the spoken AI disclosure (RESULTS.md line 28). */
  firstAudio: 0.84,
  /** First audio, p95 (line 28). */
  firstAudioP95: 1.34,
  /** First sentence of the answer itself, median (line 29). */
  firstAnswer: 2.23,
  /** First sentence of the answer, p95 (line 29). */
  firstAnswerP95: 3.58,
  /** Full reply ready, stitched and watermarked, median (line 30). */
  fullReply: 3.02,
  /** Speech recognition, median (line 31; the raw file gives 730 ms). STATUS.md lists 0.83 s, which does not match. */
  asr: 0.73,
  /** Knowledge search, median (line 32). */
  search: 0.04,
  /** Voice synthesis per turn, median (line 33; the raw file gives 822 ms). STATUS.md lists 0.73 s, which does not match. */
  voice: 0.82,
} as const;

export const MEASURED_ON = "Medians over 30 spoken questions in English and Hindi, with local models on an Apple M4 laptop (16 GB).";
export const CLOCK_NOTE = "The clock starts when the audio request is sent, not when the caller stops speaking.";
export const OVERLAP_NOTE = "Stages overlap: speech starts while the reply is still being written.";

/** "0.84 s". */
export const sec = (n: number) => `${n.toFixed(2)} s`;

/** The made-up example, kept in one place so every artifact tells the same story. */
export const EXAMPLE = {
  greeting: "Namaste, main ek AI assistant hoon.",
  question: "Kal Dr. Mehta ki OPD kitne baje hai?",
  /** The reply, split into the words that stream in. */
  reply: "Dr. Mehta ki OPD kal subah 10 se 1 baje tak hai.",
  /** The document pages the search looks through; `hit` is the one that holds the passage. */
  pages: [
    { id: "p1", doc: "Fees and payments", page: 1, hit: false },
    { id: "p2", doc: "Clinic timings", page: 2, hit: true },
    { id: "p3", doc: "Our doctors", page: 3, hit: false },
    { id: "p4", doc: "Before your visit", page: 1, hit: false },
  ],
  passage: "OPD · Dr. Mehta · Mon-Sat 10:00-13:00",
  receipt: {
    id: "rcpt_8f3a9c21",
    signed: "signed (Ed25519)",
    engine: "Kokoro-82M",
    consent: "preset voice",
    check: "Check at /verify",
  },
} as const;

export const INPUTS = {
  browser: {
    label: "Browser",
    caption: "Hold to talk in the browser. Audio goes straight from the microphone.",
  },
  phone: {
    label: "Phone call",
    caption: "On a phone call the audio is 8 kHz, through Exotel (tested on a trial account).",
  },
} as const;
export type InputKind = keyof typeof INPUTS;

export interface Step {
  id: "disclose" | "speak" | "listen" | "find" | "write" | "talk" | "sign" | "answer";
  name: string;
  /** One line shown beside the stage. */
  line: string;
  /** The fuller explanation, always in the page. */
  body: string;
  trust: boolean;
  /** A measured time for this step, when there is one. */
  time?: string;
}

export const STEPS: Step[] = [
  {
    id: "disclose",
    name: "It says it is an AI",
    line: "Every call and chat starts with it.",
    body: "At the start of every call or chat the agent tells the caller it is an AI. In our test this spoken line is the first sound back, at 0.84 s, and it plays while the answer is still being written.",
    trust: true,
  },
  {
    id: "speak",
    name: "You speak",
    line: "In the browser, or on a phone call.",
    body: "Hold to talk in the browser, or take a phone call. Phone audio is 8 kHz, through Exotel (tested on a trial account).",
    trust: false,
  },
  {
    id: "listen",
    name: "It listens",
    line: "Speech becomes text.",
    body: "Whisper turns your speech into text, on Groq or on your Mac's GPU.",
    trust: false,
    time: sec(TIMING.asr),
  },
  {
    id: "find",
    name: "It finds the answer",
    line: "A search over your PDFs.",
    body: "The best-matching passages from your PDFs are found, even when you ask in Hindi.",
    trust: false,
    time: sec(TIMING.search),
  },
  {
    id: "write",
    name: "It writes the reply",
    line: "Only from what was found.",
    body: "A language model writes the reply from those passages only, or says it does not know. By default it is a Qwen3 model on Groq, with gpt-oss models as back-up. With no internet it is a small Llama 3.2 model on your Mac, which is what the times here used.",
    trust: false,
  },
  {
    id: "talk",
    name: "It speaks",
    line: "Text becomes a voice.",
    body: "Kokoro speaks replies for speed, Hindi included. A cloned voice speaks through Qwen3-TTS.",
    trust: false,
    time: sec(TIMING.voice),
  },
  {
    id: "sign",
    name: "Marked and signed",
    line: "A watermark and a receipt.",
    body: "Every spoken sentence gets the inaudible watermark, and a signed receipt is kept. Anyone can check a clip on the Verify page.",
    trust: true,
  },
  {
    id: "answer",
    name: "It answers",
    line: "The caller hears it.",
    body: `The caller hears the reply in their own language, Hindi and English mixed included. The first sentence of the answer is ready in ${sec(TIMING.firstAnswer)} (median; ${sec(TIMING.firstAnswerP95)} at p95).`,
    trust: false,
    time: `${sec(TIMING.firstAnswer)} to the first sentence`,
  },
];

/** The clock, in seconds, that the ruler runs along. */
export const RULER_MAX = 3.2;

export interface Marker {
  id: string;
  /** Clock seconds. */
  at: number;
  /** Short label on the ruler. */
  label: string;
  /** What the marker means, for the ruler's text description. */
  long: string;
  value: string;
  /** Trust moments are drawn in the marigold of the trust steps. */
  trust?: boolean;
}

/** Positions come straight from TIMING. */
export const MARKERS: Marker[] = [
  { id: "asr", at: TIMING.asr, label: "Heard", long: "speech heard", value: sec(TIMING.asr) },
  { id: "first-sound", at: TIMING.firstAudio, label: "First sound", long: "first sound, the spoken AI disclosure", value: sec(TIMING.firstAudio), trust: true },
  { id: "first-answer", at: TIMING.firstAnswer, label: "First answer", long: "first sentence of the answer", value: sec(TIMING.firstAnswer) },
  { id: "full", at: TIMING.fullReply, label: "Full reply", long: "full reply, watermarked and signed", value: sec(TIMING.fullReply), trust: true },
];

/** When each step's picture is on stage, in clock seconds. Steps 1 and 2 happen before the clock starts (0 s is the moment
 *  the request is sent). The windows do not overlap, even though the real work does; the ruler markers are the true times. */
export const WINDOWS: Record<Step["id"], [number, number]> = {
  disclose: [-1.5, -0.8],
  speak: [-0.8, 0],
  listen: [0, TIMING.asr],
  find: [TIMING.asr, 1.35],
  write: [1.35, TIMING.firstAnswer],
  talk: [TIMING.firstAnswer, 2.8],
  sign: [2.8, 3.4],
  answer: [3.4, 4.2],
};
