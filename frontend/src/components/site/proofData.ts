/** Content for the landing page's proof sections. Every figure below is copied from docs/RESULTS.md ("Current numbers",
 *  16 October 2026, with the run files and commits named there); if a number changes there, change it here.
 *  The external price and latency figures are quoted from the vendors' own pages as read on 17 October 2026
 *  (listed in RESULTS.md, "Hosted-platform reference points"). Nothing on this page is invented. */

export const RESULTS_URL = "https://github.com/Artifact19/bonclay/blob/main/docs/RESULTS.md";

export const INDIC_ATTRIBUTION = "Built with Indic-Speak from Bodhan AI / AI4Bharat.";

export type ClipEngine = "kokoro" | "indic_speak";

export interface LanguageClip {
  code: string;
  name: string;
  native: string;
  /** BCP 47 tag for the sentence, so screen readers and fonts pick the right script. */
  lang: string;
  text: string;
  src: string;
  engine: ClipEngine;
  voice: string;
  seconds: number;
}

/** One sentence about booking an appointment, rendered once with PRESET voices only (never a cloned voice). */
export const LANGUAGE_CLIPS: LanguageClip[] = [
  { code: "en", name: "English", native: "English", lang: "en", text: "Hello, I would like to book an appointment for tomorrow at ten in the morning.", src: "/samples/languages/en.m4a", engine: "kokoro", voice: "Heart", seconds: 3.9 },
  { code: "hi", name: "Hindi", native: "हिन्दी", lang: "hi", text: "नमस्ते, मैं कल सुबह दस बजे के लिए अपॉइंटमेंट बुक करना चाहती हूँ।", src: "/samples/languages/hi.m4a", engine: "kokoro", voice: "Alpha", seconds: 5.2 },
  { code: "mr", name: "Marathi", native: "मराठी", lang: "mr", text: "नमस्कार, मला उद्या सकाळी दहा वाजता अपॉइंटमेंट बुक करायची आहे.", src: "/samples/languages/mr.m4a", engine: "indic_speak", voice: "Anagha", seconds: 5.7 },
  { code: "ta", name: "Tamil", native: "தமிழ்", lang: "ta", text: "வணக்கம், நாளை காலை பத்து மணிக்கு ஒரு சந்திப்பை முன்பதிவு செய்ய விரும்புகிறேன்.", src: "/samples/languages/ta.m4a", engine: "indic_speak", voice: "Anitha", seconds: 6.1 },
  { code: "bn", name: "Bengali", native: "বাংলা", lang: "bn", text: "নমস্কার, আমি আগামীকাল সকাল দশটায় একটি অ্যাপয়েন্টমেন্ট বুক করতে চাই।", src: "/samples/languages/bn.m4a", engine: "indic_speak", voice: "Ishita", seconds: 5.5 },
  { code: "te", name: "Telugu", native: "తెలుగు", lang: "te", text: "నమస్కారం, నేను రేపు ఉదయం పది గంటలకు అపాయింట్‌మెంట్ బుక్ చేసుకోవాలనుకుంటున్నాను.", src: "/samples/languages/te.m4a", engine: "indic_speak", voice: "Sravani", seconds: 5.9 },
  { code: "fr", name: "French", native: "Français", lang: "fr", text: "Bonjour, je voudrais prendre rendez-vous demain à dix heures du matin.", src: "/samples/languages/fr.m4a", engine: "kokoro", voice: "Siwis", seconds: 3.7 },
  { code: "es", name: "Spanish", native: "Español", lang: "es", text: "Hola, quisiera reservar una cita para mañana a las diez de la mañana.", src: "/samples/languages/es.m4a", engine: "kokoro", voice: "Dora", seconds: 3.9 },
];

export const ENGINE_CREDIT: Record<ClipEngine, string> = {
  kokoro: "Kokoro-82M, preset voice",
  indic_speak: "Indic-Speak, preset voice",
};

export interface Stat {
  value: string;
  label: string;
  note: string;
}

/** Speed: 30 spoken questions (15 English, 15 Hindi), streaming, local models only, 16 Oct 2026 (docs/RESULTS.md,
 *  "Speed, 30 spoken questions"); Groq 15 Oct 2026 (TRIAL_REPORT.md section 2); clone time 15 Oct 2026. */
export const MEASURED_SPEED: Stat[] = [
  { value: "2.23 s", label: "to the first sentence of the answer", note: "median; 3.58 s at p95" },
  { value: "0.84 s", label: "to first audio", note: "median. It is the spoken AI disclosure, which plays while the answer is written" },
  { value: "3.02 s", label: "to the full reply, all on the laptop", note: "median; 5.03 s at p95, no cloud keys" },
  { value: "2.81 s", label: "to the full reply with Groq", note: "median, under a 3 s target; the 10 turns Groq answered, 15 Oct" },
  { value: "2.4 s", label: "to clone a voice", note: "warm median, consent check included, 10 runs, 15 Oct" },
];

export const MEASURED_SPEED_CONDITIONS = "Apple M4, 16 GB, 30 spoken questions (15 English, 15 Hindi), local models only, 16 Oct 2026.";

/** Trust: retrieval (TRIAL_REPORT.md section 3), watermark (sections 4 and 8), soak (RESULTS.md), test counts (RESULTS.md "Test counts"). */
export const MEASURED_TRUST: Stat[] = [
  { value: "97.8%", label: "knowledge search finds the right passage in its top 3", note: "45 questions in English, Hindi and Hinglish; 88.9% before the current search" },
  { value: "70 / 70", label: "watermark checks passed on common channels", note: "MP3, Opus, WhatsApp-style, 8 kHz phone line, trimming; English and Hindi. Speed or pitch changes and heavy noise can remove it" },
  { value: "733 / 733", label: "requests answered in a 45-minute soak", note: "0 failures, 0 crashes, 15 Oct" },
  { value: "1,390 + 404", label: "automated tests, backend and frontend", note: "all passing on 17 Oct 2026 (pytest and Jest)" },
];

export const MEASURED_TRUST_CONDITIONS = "Knowledge search: 45 questions. Watermark: 20 clips, 21 attacks. Soak: Apple M4, 16 GB, 15 Oct 2026.";

/** Everything in the strip, for anything that wants one list. */
export const MEASURED: Stat[] = [...MEASURED_SPEED, ...MEASURED_TRUST];

export const MEASURED_CONTEXT =
  "Measured on one Apple M4 laptop with 16 GB of memory. Speed numbers use local models and no cloud keys unless a line says Groq.";
export const MEASURED_CAVEAT =
  "Hindi is slower than English: first sentence of the answer 3.07 s against 1.89 s. The Groq figure comes from 10 turns; the other 20 fell back to the local model after the free tier ran out. These times start when the audio request is sent, not when the caller stops speaking. Read-back error (7.6% English word error, 7.1% Hindi character error) measures how a speech recogniser hears the voice, not how natural it sounds.";

export interface Cite {
  label: string;
  href: string;
}

export interface Cell {
  text: string;
  /** Linked citations for figures that are not ours, each with its month and year. */
  cites?: Cite[];
  /** True when we did not check this, so the cell is shown quietly instead of as a claim. */
  unchecked?: boolean;
}

export interface CompareRow {
  topic: string;
  bonclay: Cell;
  groq: Cell;
  hosted: Cell;
}

export const VAPI_PRICING: Cite = { label: "Vapi pricing, Oct 2026", href: "https://vapi.ai/pricing" };
export const RETELL_PRICING: Cite = { label: "Retell AI pricing, Oct 2026", href: "https://www.retellai.com/pricing" };
export const VAPI_LANGUAGES: Cite = { label: "Vapi multilingual docs, Oct 2026", href: "https://docs.vapi.ai/customization/multilingual" };
export const TWILIO_LATENCY: Cite = {
  label: "Twilio, Core Latency in AI Voice Agents, Nov 2025",
  href: "https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents",
};

export const COMPARISON_COLUMNS = ["BonClay (local)", "BonClay (+ Groq)", "Typical hosted voice-agent platform"] as const;

export const COMPARISON: CompareRow[] = [
  {
    topic: "Price per minute",
    bonclay: { text: "No per-minute fee. You pay for your own computer's power; we have not measured that cost." },
    groq: { text: "Groq charges by the token. We have used only its free tier, which runs out after a day of heavy testing; a paid tier is untested." },
    hosted: {
      text: "Vapi: $0.05 a minute for hosting alone, with speech, model, voice and phone costs added. Retell AI: $0.11 a minute in its default estimator (voice infrastructure $0.055, model $0.04, voice $0.015, phone $0).",
      cites: [VAPI_PRICING, RETELL_PRICING],
    },
  },
  {
    topic: "Where audio and data live",
    bonclay: { text: "On your laptop: audio, documents, voices and receipts. No cloud keys needed." },
    groq: { text: "The caller's audio, and the question with the document passages it uses, go to Groq. Voice synthesis, watermark and receipts stay on your laptop." },
    hosted: {
      text: "On the platform's servers and those of the speech, model and voice vendors it routes to. Vapi's page names Deepgram, OpenAI and ElevenLabs.",
      cites: [VAPI_PRICING],
    },
  },
  {
    topic: "Works offline",
    bonclay: { text: "Yes, once the models are downloaded. The measurements here used no cloud keys." },
    groq: { text: "No, it needs the internet. When Groq's limits run out it falls back to the slower local model." },
    hosted: { text: "No. It is a web service." },
  },
  {
    topic: "Consent-checked cloning",
    bonclay: { text: "Yes. The speaker says a fresh sentence and code aloud; it is transcribed and a speaker check compares it with the sample. A consent record is kept and withdrawal deletes the voice." },
    groq: { text: "Yes, the same checks. Cloning runs on your laptop." },
    hosted: { text: "Not checked.", unchecked: true },
  },
  {
    topic: "Watermark and public verify",
    bonclay: { text: "Yes. Every clip and reply gets an inaudible watermark and a signed receipt, and anyone can check a clip on the Verify page. Detected 70 of 70 times after MP3, WhatsApp-style and phone-line audio; speed or pitch changes can remove it." },
    groq: { text: "Yes, the same." },
    hosted: { text: "Not checked.", unchecked: true },
  },
  {
    topic: "Indian languages",
    bonclay: { text: "Speech in 22 Indian languages with preset voices, and your own consented voice in Indian languages with DhVaani. Live agent replies are in English, Hindi and Hinglish." },
    groq: { text: "The same voices. Groq's Qwen model answered Hindi, Marathi and Hinglish correctly in our checks." },
    hosted: {
      text: "Depends on the speech and voice vendor you plug in. Vapi's docs give language counts per provider, not a list of Indian languages.",
      cites: [VAPI_LANGUAGES],
    },
  },
  {
    topic: "First response time",
    bonclay: { text: "First sentence of the answer 2.23 s median (3.58 s at p95); full reply 3.02 s. 30 questions, 16 Oct 2026." },
    groq: { text: "First sentence 2.52 s; full reply 2.81 s. The 10 turns Groq answered, 15 Oct 2026." },
    hosted: {
      text: "Twilio's guide sets a 1.1 s (1,115 ms) mouth-to-ear target for a cascaded voice agent. That is a published target, not a measured product, and its clock starts when the caller stops speaking, so it is not like-for-like.",
      cites: [TWILIO_LATENCY],
    },
  },
];

export const COMPARISON_FOOTNOTE =
  "Hosted-platform figures are quoted from Vapi's and Retell AI's own public pages and Twilio's guide as read on 17 October 2026; the pricing pages show no publication date, so the month is when we read them. Prices change, so follow the links. \"Not checked\" means we did not review those vendors' policies.";

export const NOT_YET: { topic: string; body: string }[] = [
  { topic: "WhatsApp is not live-tested", body: "Voice notes work against a simulated WhatsApp service. It needs a Meta test number before we can say it works on a real phone." },
  { topic: "Phone calls: trial account only", body: "Outbound calls through Exotel were tested live once on a trial account. Receiving calls on your own number is untested." },
  { topic: "Windows is untested", body: "A Windows set-up script exists, but nobody has run BonClay on a Windows machine with a graphics card yet." },
  { topic: "No paid language-model tier yet", body: "The fast answers use Groq's free tier, which runs out after a day of heavy testing. Replies then come from the slower local model." },
  { topic: "Tight on a 16 GB Mac", body: "With every model loaded, a 16 GB Mac can swap 3 to 5 GB to disk. The biggest Indian-language voice alone needs about 7 GB." },
  { topic: "Indian-language voices are not live", body: "Indian-language preset and cloned voices take several seconds per sentence on a laptop, so they suit Studio renders. Live calls use English and Hindi preset voices." },
];
