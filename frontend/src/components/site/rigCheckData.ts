/** Content for the landing page's "Rig Check" section. Every figure carries a `source`:
 *   measured  - taken on a real machine by this project (Apple M4, 16 GB). Numbers are copied from docs/TRIAL_REPORT.md and
 *               docs/RESULTS.md, and disk sizes were read from that machine's Hugging Face cache and backend/.venv* folders.
 *   estimated - worked out from model sizes or the optimisation research (docs/OPTIMIZATION_RESEARCH.md), not run on that hardware.
 *   untested  - nobody has run BonClay on this hardware yet; the card says what we expect and why.
 *  If a number changes in the docs, change it here. Nothing on this card is invented to look better. */

export type Source = "measured" | "estimated" | "untested";

export const README_SETUP_URL = "https://github.com/Artifact19/bonclay/blob/main/README.md#-getting-started";
export const OPTIMIZATION_URL = "https://github.com/Artifact19/bonclay/blob/main/docs/OPTIMIZATION_RESEARCH.md";
export const TRIAL_REPORT_URL = "https://github.com/Artifact19/bonclay/blob/main/docs/TRIAL_REPORT.md";

export const SOURCE_LABEL: Record<Source, string> = { measured: "Measured", estimated: "Estimated", untested: "Untested" };
export const SOURCE_HINT: Record<Source, string> = {
  measured: "Timed or sized on a real machine by this project.",
  estimated: "Worked out from model sizes or research notes, not run on this hardware.",
  untested: "No one has run BonClay on this hardware yet.",
};

/* ------------------------------------------------------------------ capabilities */

export type CapId = "live" | "clone" | "hindiClone" | "indic" | "offline" | "handsFree";
/** yes = works well; partial = works with limits (slow, swaps, needs extra setup); no = not practical. */
export type Level = "yes" | "partial" | "no";

export const CAPABILITIES: { id: CapId; label: string; short: string }[] = [
  { id: "live", label: "Live voice agent", short: "Live agent" },
  { id: "clone", label: "Cloning in your voice", short: "Cloning" },
  { id: "hindiClone", label: "Your voice in Indian languages", short: "Indian-language clone" },
  { id: "indic", label: "22 Indian-language preset voices", short: "22 Indian voices" },
  { id: "offline", label: "Offline thinking (no internet)", short: "Offline thinking" },
  { id: "handsFree", label: "Hands-free conversation", short: "Hands-free" },
];

export const LEVEL_SYMBOL: Record<Level, string> = { yes: "✓", partial: "~", no: "✗" };
export const LEVEL_WORD: Record<Level, string> = { yes: "Works", partial: "Works with limits", no: "Not practical" };
export const LEGEND: { level: Level; text: string }[] = [
  { level: "yes", text: "works well" },
  { level: "partial", text: "works with limits (slow, swaps memory, or needs extra setup)" },
  { level: "no", text: "not practical on this machine" },
];

/* ------------------------------------------------------------------ result shape */

export type Tier = "Full studio" | "Runs well" | "Runs with Groq" | "Basic";

export interface Figure {
  label: string;
  value: string;
  source: Source;
  note?: string;
}

export interface CapCell {
  level: Level;
  /** One short reason, shown on the card (not in the table). */
  note?: string;
}

export interface Profile {
  tier: Tier;
  /** One sentence under the badge. */
  summary: string;
  caps: Record<CapId, CapCell>;
  /** What to switch off, or send to the cloud, on this machine. */
  advice: string[];
  speed: Figure[];
}

export interface Machine {
  id: string;
  group: "Apple Silicon Mac" | "Windows / Linux with an NVIDIA GPU" | "Any laptop";
  label: string;
  hint: string;
  /** How well this machine is known. The worst case drives the banner on the card. */
  evidence: Source;
  evidenceNote: string;
  /** Short name for the comparison table's row header. */
  tableName: string;
  local: Profile;
  groq: Profile;
  /** Disk and memory figures that do not depend on the Groq toggle. */
  footprint: Figure[];
}

const c = (level: Level, note?: string): CapCell => ({ level, note });

/* ------------------------------------------------------------------ shared figures */

/** Sizes read from the M4's ~/.cache/huggingface/hub and backend/.venv* folders (du, read-only). */
const MAC_CORE_DISK: Figure = {
  label: "Core install on disk",
  value: "about 15 GB",
  source: "measured",
  note: "10.6 GB of models (embedding model 4.3 GB, Qwen3-TTS 2.3 GB, local language model 1.7 GB, Whisper 2.0 GB, Kokoro 0.3 GB) plus 4.1 GB of Python environments.",
};
const INDIC_DISK: Figure = { label: "Add Indic-Speak (--indic)", value: "about 7.5 GB", source: "measured", note: "6.6 GB model plus a 0.9 GB environment, read on the M4; the same files on any machine. Needs about 8 GB of free memory to run." };
const DHVAANI_DISK: Figure = { label: "Add DhVaani (--dhvaani)", value: "about 1.3 GB", source: "measured", note: "0.5 GB model and vocoder plus a 0.8 GB environment, read on the M4; the same files on any machine. Needs an HF_TOKEN whose account has been granted access by ARTPARK-IISc." };
const CHATTERBOX_DISK: Figure = { label: "Add Chatterbox (--chatterbox)", value: "about 4.3 GB", source: "measured", note: "3.0 GB model plus a 1.3 GB environment, read on the M4; the same files on any machine." };
const PC_CORE_DISK: Figure = {
  label: "Core install on disk",
  value: "about 10 to 14 GB",
  source: "estimated",
  note: "Same models as the Mac except the Apple-only language model and the Apple-only speed-ups, plus NVIDIA's GPU software if you have an NVIDIA GPU. Not measured on a PC.",
};

/* ------------------------------------------------------------------ machines */

const MAC16_SPEED_LOCAL: Figure[] = [
  { label: "Full spoken reply (median)", value: "3.1 s", source: "measured", note: "English 2.1 s, Hindi 4.1 s; 30 spoken questions, local models only" },
  { label: "First audio (median)", value: "1.0 s", source: "measured" },
  { label: "Voice clone built", value: "2.4 s", source: "measured", note: "warm, consent check included" },
  { label: "Indian-language preset voice", value: "about 35 s per sentence", source: "measured", note: "Indic-Speak is about 7x slower than real time: for Studio renders, not live calls" },
  { label: "Your voice in Indian languages", value: "about 5 to 7 s per sentence", source: "measured", note: "DhVaani on the Apple GPU, 1.2 to 1.5x real time on a quiet M4 and up to 3x under load: for Studio renders, not live calls. Hindi alone can fall back to Chatterbox (30 to 40 s per sentence on the CPU)" },
  { label: "Memory with every model loaded", value: "10 to 12 GB", source: "measured", note: "16 GB works but swaps under heavy use" },
];

export const MACHINES: Machine[] = [
  {
    id: "mac8",
    group: "Apple Silicon Mac",
    label: "Apple Silicon Mac, 8 GB",
    tableName: "Mac 8 GB",
    hint: "M1 to M4 base models",
    evidence: "estimated",
    evidenceNote: "Not run on an 8 GB Mac. This comes from the 10 to 12 GB the full set of models needs when loaded together.",
    footprint: [MAC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Basic",
      summary: "It starts, but every model together needs 10 to 12 GB, so local thinking plus every voice swaps heavily and replies drag.",
      caps: {
        live: c("partial", "works, but slow once memory swaps"),
        clone: c("partial", "fits only if nothing else is loaded"),
        hindiClone: c("partial", "not run on 8 GB; the model is small but loads next to everything else, render it"),
        indic: c("no", "needs about 8 GB of free memory on its own"),
        offline: c("partial", "the local model runs, but crowds out the rest"),
        handsFree: c("no", "waiting on swapped-out models is too slow"),
      },
      advice: [
        "Turn on a free Groq key: it moves speech recognition and thinking off your machine and frees about 2 GB.",
        "Skip --indic. Hindi and Marathi still speak through Kokoro's Hindi voice.",
        "Use the lighter knowledge-search model (see backend/.env.example) to save about 4 GB of disk.",
      ],
      speed: [
        { label: "Full spoken reply", value: "expect well over the M4's 3.1 s once swapping starts", source: "estimated", note: "no timing taken on 8 GB" },
        { label: "Memory with every model loaded", value: "10 to 12 GB, more than the machine has", source: "measured", note: "measured on the 16 GB M4" },
      ],
    },
    groq: {
      tier: "Runs with Groq",
      summary: "With Groq doing the listening and thinking, only the voices stay on your Mac, so a live agent is realistic.",
      caps: {
        live: c("yes", "Groq listens and thinks, Kokoro speaks locally"),
        clone: c("partial", "one voice model at a time"),
        hindiClone: c("partial", "not run on 8 GB; the model is small but loads next to everything else, render it"),
        indic: c("no", "needs about 8 GB of free memory on its own"),
        offline: c("partial", "needs the local model installed as a fallback"),
        handsFree: c("yes", "should hold up with cloud listening"),
      },
      advice: [
        "Keep Groq on for live calls. Switch to Studio-style rendering for cloning and your voice in Indian languages.",
        "Skip --indic on this machine, and install --chatterbox only if you will wait for it.",
        "Close the browser tabs you do not need before a live demo.",
      ],
      speed: [
        { label: "Full spoken reply", value: "expect close to the M4's 2.8 s with Groq", source: "estimated", note: "Groq does the heavy part, which is the same on any Mac; not timed on 8 GB" },
      ],
    },
  },
  {
    id: "mac16",
    group: "Apple Silicon Mac",
    label: "Apple Silicon Mac, 16 GB",
    tableName: "Mac 16 GB",
    hint: "Where every number was measured (Apple M4)",
    evidence: "measured",
    evidenceNote: "Measured on an Apple M4 with 16 GB, local models only unless the line says Groq.",
    footprint: [MAC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Runs well",
      summary: "The whole studio runs offline with no keys. A spoken reply takes about 3 seconds, and the heavy voices are for renders.",
      caps: {
        live: c("yes", "full spoken reply 3.1 s"),
        clone: c("yes", "voice built in 2.4 s"),
        hindiClone: c("yes", "5 to 7 s per sentence on the Apple GPU, measured; renders, not live calls"),
        indic: c("partial", "35 s per sentence, and it swaps with the rest"),
        offline: c("yes", "Llama 3.2 3B on the Apple GPU"),
        handsFree: c("yes", "first audio in about 1 s"),
      },
      advice: [
        "Run ./bonclay warm before a demo so the first reply is about 3 s instead of about 18 s.",
        "Use Indic-Speak and DhVaani (your voice in Indian languages) for Studio renders. Keep live calls on Kokoro voices.",
        "Avoid loading Indic-Speak while a live call runs: with every model loaded memory hits 10 to 12 GB and the Mac swaps.",
      ],
      speed: MAC16_SPEED_LOCAL,
    },
    groq: {
      tier: "Runs well",
      summary: "Groq takes the listening and thinking, which frees memory and shaves the reply time. Voices still run on your Mac.",
      caps: {
        live: c("yes", "full spoken reply 2.8 s"),
        clone: c("yes", "voice built in 2.4 s"),
        hindiClone: c("yes", "5 to 7 s per sentence on the Apple GPU, measured; renders, not live calls"),
        indic: c("partial", "35 s per sentence"),
        offline: c("yes", "local model stays installed as the fallback"),
        handsFree: c("yes", "first audio about 0.5 s on Groq turns"),
      },
      advice: [
        "Groq's free tier has daily and per-minute limits. In the trial 20 of 30 turns fell back to the local model; that is why the local model stays installed.",
        "Use Indic-Speak and DhVaani for Studio renders, not live calls.",
        "Skip the local language model if you never go offline, to save about 1.7 GB of disk.",
      ],
      speed: [
        { label: "Full spoken reply (Groq answering)", value: "2.8 s", source: "measured", note: "on the 10 of 30 turns Groq answered; first audio 0.5 s" },
        { label: "Voice clone built", value: "2.4 s", source: "measured" },
        { label: "Indian-language preset voice", value: "about 35 s per sentence", source: "measured", note: "Studio renders, not live calls" },
        { label: "Your voice in Indian languages", value: "about 5 to 7 s per sentence", source: "measured", note: "DhVaani on the Apple GPU; Chatterbox, the Hindi fallback, takes 30 to 40 s on the CPU" },
        { label: "Memory", value: "lower than the 10 to 12 GB all-local figure", source: "estimated", note: "the local language model need not be loaded" },
      ],
    },
  },
  {
    id: "mac24",
    group: "Apple Silicon Mac",
    label: "Apple Silicon Mac, 24 GB or more",
    tableName: "Mac 24 GB+",
    hint: "Room for every model at once",
    evidence: "estimated",
    evidenceNote: "Not run on a 24 GB Mac. Same engines as the measured 16 GB machine with more room, so the figures there are a floor.",
    footprint: [MAC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Full studio",
      summary: "Everything fits without swapping, so the heavy voices and a live call can run side by side.",
      caps: {
        live: c("yes"),
        clone: c("yes"),
        hindiClone: c("yes", "about 5 to 7 s per sentence on the M4 GPU, with more room to spare"),
        indic: c("yes", "fits next to the other models; about 35 s per sentence on the M4"),
        offline: c("yes", "the better Gemma 4 E4B model fits too"),
        handsFree: c("yes"),
      },
      advice: [
        "Switch the offline model to Gemma 4 E4B (LOCAL_LLM_MODEL in backend/.env): much better at Hindi, Marathi and Tamil, about 2.5 GB more.",
        "The slow engines are slow because of compute, not memory, so Indic-Speak and DhVaani are still renders.",
      ],
      speed: [
        { label: "Full spoken reply", value: "3.1 s or better; swap stalls go away", source: "estimated", note: "M4 16 GB figure used as the floor" },
        { label: "Indian-language preset voice", value: "about 35 s per sentence or better", source: "estimated" },
      ],
    },
    groq: {
      tier: "Full studio",
      summary: "Groq for fast thinking, the whole voice stack local, and no memory pressure.",
      caps: {
        live: c("yes"),
        clone: c("yes"),
        hindiClone: c("yes", "about 5 to 7 s per sentence on the M4 GPU, with more room to spare"),
        indic: c("yes"),
        offline: c("yes", "local model kept as the fallback"),
        handsFree: c("yes"),
      },
      advice: [
        "Nothing needs switching off. Keep the local model as the fallback for when the free Groq limits run out.",
      ],
      speed: [
        { label: "Full spoken reply", value: "2.8 s or better", source: "estimated", note: "M4 16 GB Groq figure used as the floor" },
      ],
    },
  },
  {
    id: "nv4",
    group: "Windows / Linux with an NVIDIA GPU",
    label: "NVIDIA GPU, 4 GB VRAM (for example RTX 3050 Ti)",
    tableName: "NVIDIA 4 GB",
    hint: "Laptop GPUs such as the RTX 3050 Ti",
    evidence: "untested",
    evidenceNote: "Untested on real hardware. The Windows setup script (bonclay.ps1) exists but has not been run on a 3050 Ti; this card comes from model sizes.",
    footprint: [PC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Basic",
      summary: "Voices fit on the GPU one job at a time, but a local language model would need the whole 4 GB. Without Groq you must bring your own (Ollama or LM Studio).",
      caps: {
        live: c("partial", "needs Ollama or LM Studio you set up yourself"),
        clone: c("partial", "needs the GPU to itself (about 2 GB)"),
        hindiClone: c("partial", "not tried; the small model should fit next to cloning, estimated"),
        indic: c("no", "the 7 GB model does not fit in 4 GB of VRAM"),
        offline: c("partial", "a 3B model takes most of the GPU"),
        handsFree: c("no", "model swapping on a small GPU is too slow"),
      },
      advice: [
        "Use a free Groq key for thinking. A local language model would need the whole GPU.",
        "Set MAX_GPU_MEMORY_GB=3 in backend/.env so models unload before the GPU runs out.",
        "Skip --indic. Hindi and Marathi speak through Kokoro's Hindi voice.",
      ],
      speed: [
        { label: "Kokoro plus Whisper on the GPU", value: "about 1.6 to 1.9 GB together", source: "estimated" },
        { label: "Voice cloning on the GPU", value: "about 2 GB, GPU alone", source: "estimated" },
        { label: "Full spoken reply", value: "not timed", source: "untested" },
      ],
    },
    groq: {
      tier: "Runs with Groq",
      summary: "Groq does the thinking, so the GPU only holds Kokoro and Whisper together, and steps aside when you clone a voice.",
      caps: {
        live: c("yes", "Groq thinks; Kokoro and Whisper share the GPU"),
        clone: c("partial", "needs the GPU alone (about 2 GB), so other models unload first"),
        hindiClone: c("partial", "not tried; the small model should fit next to cloning, estimated"),
        indic: c("no", "the 7 GB model does not fit in 4 GB of VRAM"),
        offline: c("partial", "only if you also run Ollama or LM Studio"),
        handsFree: c("partial", "possible, but not tried"),
      },
      advice: [
        "Keep Groq on: it is what makes this machine usable for live calls.",
        "Clone voices as a separate step, then go back to live calls. Do not run both at once.",
        "Skip --indic, and treat Windows as untested: run bonclay.ps1 doctor and tell us what breaks.",
      ],
      speed: [
        { label: "Kokoro plus Whisper on the GPU", value: "about 1.6 to 1.9 GB together", source: "estimated" },
        { label: "Voice cloning on the GPU", value: "about 2 GB, GPU alone", source: "estimated" },
        { label: "Full spoken reply", value: "not timed", source: "untested" },
      ],
    },
  },
  {
    id: "nv8",
    group: "Windows / Linux with an NVIDIA GPU",
    label: "NVIDIA GPU, 6 to 8 GB VRAM",
    tableName: "NVIDIA 6-8 GB",
    hint: "RTX 3060, 4060, 3070 class",
    evidence: "untested",
    evidenceNote: "Untested on real hardware. Figures are inferred from the 4 GB estimates and model sizes.",
    footprint: [PC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Runs well",
      summary: "Speech models and a small local language model can share the GPU, with cloning loading when you ask for it.",
      caps: {
        live: c("partial", "needs Ollama or LM Studio you set up yourself"),
        clone: c("yes", "about 2 GB, fits"),
        hindiClone: c("yes", "not tried; should be at least as fast as the Mac, estimated"),
        indic: c("partial", "7 GB model is tight; likely needs the GPU to itself"),
        offline: c("yes", "a 3B model next to Kokoro and Whisper"),
        handsFree: c("partial", "plausible, not tried"),
      },
      advice: [
        "Point LLM_BASE_URL at Ollama or LM Studio for a local 3B model, or use Groq.",
        "Run Indic-Speak alone for renders rather than next to a live call.",
      ],
      speed: [
        { label: "Kokoro plus Whisper on the GPU", value: "about 1.6 to 1.9 GB together", source: "estimated" },
        { label: "Full spoken reply", value: "not timed", source: "untested" },
      ],
    },
    groq: {
      tier: "Runs well",
      summary: "Groq thinking leaves plenty of VRAM for voices and cloning.",
      caps: {
        live: c("yes"),
        clone: c("yes", "about 2 GB, fits next to the rest"),
        hindiClone: c("yes", "not tried; should be at least as fast as the Mac, estimated"),
        indic: c("partial", "tight; run it alone"),
        offline: c("partial", "only with Ollama or LM Studio installed"),
        handsFree: c("yes"),
      },
      advice: [
        "Keep Groq on for live calls and use the spare VRAM for cloning.",
        "Run Indic-Speak alone for renders.",
      ],
      speed: [
        { label: "Full spoken reply", value: "not timed", source: "untested" },
      ],
    },
  },
  {
    id: "nv12",
    group: "Windows / Linux with an NVIDIA GPU",
    label: "NVIDIA GPU, 12 GB VRAM or more",
    tableName: "NVIDIA 12 GB+",
    hint: "RTX 3080 Ti, 4070 Ti and up",
    evidence: "untested",
    evidenceNote: "Untested on real hardware. The models are small enough that this should be comfortable, but nobody has run it.",
    footprint: [PC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Full studio",
      summary: "Every model fits in VRAM at once, including a local language model and Indic-Speak.",
      caps: {
        live: c("yes", "with Ollama or LM Studio you set up"),
        clone: c("yes"),
        hindiClone: c("yes", "not tried; should be at least as fast as the Mac, estimated"),
        indic: c("yes", "7.5 GB model fits"),
        offline: c("yes"),
        handsFree: c("yes"),
      },
      advice: [
        "Install a local model in Ollama or LM Studio and set LLM_BASE_URL: BonClay does not bundle one outside Apple Silicon.",
        "Treat everything as unconfirmed until you have run bonclay.ps1 doctor (Windows) or ./bonclay doctor (Linux) on your own machine.",
      ],
      speed: [
        { label: "Full spoken reply", value: "not timed", source: "untested" },
        { label: "Indian-language preset voice", value: "should be much faster than the Mac's 35 s", source: "untested" },
      ],
    },
    groq: {
      tier: "Full studio",
      summary: "Cloud thinking for speed, the whole voice stack on your GPU, and room to spare.",
      caps: {
        live: c("yes"),
        clone: c("yes"),
        hindiClone: c("yes", "not tried; should be at least as fast as the Mac, estimated"),
        indic: c("yes"),
        offline: c("yes", "with Ollama or LM Studio installed"),
        handsFree: c("yes"),
      },
      advice: ["Nothing needs switching off. Report your timings so this card can say measured instead of untested."],
      speed: [{ label: "Full spoken reply", value: "not timed", source: "untested" }],
    },
  },
  {
    id: "cpu",
    group: "Any laptop",
    label: "Any laptop, no GPU (CPU only)",
    tableName: "CPU only",
    hint: "Intel or AMD without a dedicated GPU",
    evidence: "estimated",
    evidenceNote: "Not run on a CPU-only laptop. Kokoro is small and runs on the CPU; the slow parts are cloning and anything large.",
    footprint: [PC_CORE_DISK, INDIC_DISK, CHATTERBOX_DISK, DHVAANI_DISK],
    local: {
      tier: "Basic",
      summary: "There is no bundled local language model outside Apple Silicon, so without Groq (or Ollama) there is nothing to think with.",
      caps: {
        live: c("no", "needs a Groq key or Ollama / LM Studio"),
        clone: c("partial", "works on the CPU but slow"),
        hindiClone: c("partial", "works on the CPU but several times slower than real time, estimated"),
        indic: c("no", "7 GB model on a CPU is not practical"),
        offline: c("no", "no local language model is bundled"),
        handsFree: c("no"),
      },
      advice: [
        "Add a free Groq key. It is the difference between a talking agent and none.",
        "Use Kokoro preset voices (fine on the CPU) and render clones as a background job.",
      ],
      speed: [
        { label: "Kokoro preset voices", value: "fine on the CPU", source: "estimated" },
        { label: "Voice cloning", value: "slow, minutes rather than seconds", source: "estimated" },
      ],
    },
    groq: {
      tier: "Runs with Groq",
      summary: "Groq listens and thinks and Kokoro speaks on the CPU, which is enough for a live agent. Cloning stays slow.",
      caps: {
        live: c("yes", "Groq thinks, Kokoro speaks on the CPU"),
        clone: c("partial", "works but slow"),
        hindiClone: c("partial", "works on the CPU but several times slower than real time, estimated"),
        indic: c("no", "7 GB model on a CPU is not practical"),
        offline: c("no", "no local language model is bundled"),
        handsFree: c("partial", "depends on your CPU"),
      },
      advice: [
        "Use preset voices for live calls. Clone as a background render.",
        "Skip --indic and --chatterbox unless you are happy to wait.",
        "8 GB of RAM should be enough here, because the large local models are not loaded (not tried).",
      ],
      speed: [
        { label: "Kokoro preset voices", value: "fine on the CPU", source: "estimated" },
        { label: "Full spoken reply", value: "not timed", source: "untested" },
      ],
    },
  },
];

export const MACHINE_GROUPS: Machine["group"][] = ["Apple Silicon Mac", "Windows / Linux with an NVIDIA GPU", "Any laptop"];

export const MINIMUM_LINE = "Minimum: any 8 GB laptop with a free Groq key. Recommended: Apple Silicon 16 GB or an NVIDIA GPU with 6 GB+.";
export const RIG_SECTION_CAVEAT =
  "Only the Apple M4 16 GB figures were timed. Everything else is marked estimated or untested, and we would rather say so than guess at a number.";

/** One line about the other ways to reach the agent. Phone and meetings run on the live agent, so they follow its row. */
export const CHANNELS_NOTE =
  "Phone calls (through Exotel, tested on a trial account) and meeting booking with Google Calendar (tested live) use the same live agent, so they work wherever \"Live voice agent\" works. Phone calls also need an internet address that Exotel can reach, and they sound best with a preset voice because cloned voices pause between sentences. WhatsApp voice notes are built but not yet tested live.";

/** Plain-language meanings for the words on this card. */
export const GLOSSARY: { term: string; text: string }[] = [
  { term: "Groq", text: "A free online service that listens to the caller and thinks of the answer, so your computer only has to speak it. Needs internet and a free key." },
  { term: "Local models", text: "The AI runs on your own computer, with no internet. Nothing leaves the machine, but it needs more memory." },
  { term: "Kokoro", text: "The small ready-made voice that talks during live calls and chats. Fast, and runs even without a graphics card." },
  { term: "Indic-Speak", text: "Ready-made voices in 22 Indian languages. Big and slow, so it is used to prepare audio in advance, not for live calls." },
  { term: "DhVaani", text: "Clones your own voice in Indian languages, from a sample you agreed to give. Made by ARTPARK-IISc. A few seconds per sentence, so it is for finished audio, not live calls." },
  { term: "Render", text: "Making a finished audio file you can play later, instead of talking live." },
  { term: "Swapping", text: "When a computer runs out of memory and borrows disk space. It still works, but everything gets slow." },
  { term: "VRAM", text: "The memory on a graphics card. AI voices and models have to fit in it to run fast." },
  { term: "Offline thinking", text: "The agent can work out answers with no internet by using an AI model stored on your machine." },
  { term: "Measured, estimated, untested", text: "Measured means we timed it on a real machine. Estimated means we worked it out from model sizes. Untested means nobody has run it there yet." },
];
