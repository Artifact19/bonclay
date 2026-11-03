"use client";

import { useState } from "react";

/** Short reading scripts for the voice sample: about 20 to 25 seconds, with a greeting, questions, an exclamation, an
 *  apology and a warm ending so the clone hears the speaker's range. Anyone can read them (no name in the text). */
export const RECORDING_SCRIPTS = [
  {
    code: "en",
    label: "English",
    lang: "en",
    text:
      "Good morning! Why did the quick brown fox jump over that lazy dog? Please call me at nine thirty on Thursday, " +
      "or Friday if that's easier. Wow, that's great news! Are you sure? Honestly, I'm sorry, I can't help with that. " +
      "Thank you so much, and have a lovely day, everyone.",
  },
  {
    code: "hi",
    label: "हिन्दी",
    lang: "hi",
    text:
      "नमस्ते! आज मौसम बहुत अच्छा है। क्या आप मुझे गुरुवार सुबह नौ बजे फ़ोन कर सकते हैं? अगर शुक्रवार आसान हो तो वह भी ठीक है। " +
      "वाह, यह तो बहुत अच्छी खबर है! क्या आप सच में पक्का कह रहे हैं? माफ़ कीजिए, इसमें मैं आपकी मदद नहीं कर सकता। " +
      "आपका बहुत धन्यवाद, आपका दिन शुभ हो।",
  },
  {
    code: "hinglish",
    label: "Hinglish",
    lang: "hi-Latn",
    text:
      "Hello, good morning! Aap kaise hain? Kya aap mujhe Thursday ko subah nau baje call kar sakte hain? " +
      "Agar Friday easy ho toh woh bhi theek hai. Wow, yeh toh bahut badhiya news hai! Kya aap sach mein sure hain? " +
      "Sorry, is baare mein main aapki help nahi kar sakta. Bahut bahut shukriya, aapka din accha jaye.",
  },
  {
    code: "mr",
    label: "मराठी",
    lang: "mr",
    text:
      "नमस्कार! आज हवामान खूप छान आहे. तुम्ही मला गुरुवारी सकाळी नऊ वाजता फोन करू शकता का? जर शुक्रवार सोयीचा असेल तर तेही चालेल. " +
      "अरे वा, ही तर खूप चांगली बातमी आहे! खरंच का? तुम्हाला नक्की खात्री आहे का? माफ करा, यात मी तुम्हाला मदत करू शकत नाही. " +
      "तुमचे खूप आभार, तुमचा दिवस चांगला जावो.",
  },
] as const;

/** The text to read aloud while recording the voice sample. Large while recording so it can be read from an arm's length. */
export function RecordingScript({ recording }: { recording: boolean }) {
  const [code, setCode] = useState<(typeof RECORDING_SCRIPTS)[number]["code"]>("en");
  const script = RECORDING_SCRIPTS.find((s) => s.code === code) ?? RECORDING_SCRIPTS[0];

  return (
    <section
      aria-label="Script to read for your voice sample"
      className="space-y-3 rounded-2xl border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium text-[var(--color-text-primary)]">
          {recording ? "Read this aloud now" : "Read this when you record"}
        </p>
        <div role="tablist" aria-label="Script language" className="flex gap-1">
          {RECORDING_SCRIPTS.map((s) => (
            <button
              key={s.code}
              type="button"
              role="tab"
              aria-selected={s.code === code}
              onClick={() => setCode(s.code)}
              className={`rounded-lg px-3 py-1.5 text-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-accent)] ${
                s.code === code
                  ? "bg-[var(--color-accent-primary)] text-white"
                  : "text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-surface)]"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
      <p
        lang={script.lang}
        data-testid="recording-script"
        className={`leading-relaxed text-[var(--color-text-primary)] ${recording ? "text-xl sm:text-2xl" : "text-base"}`}
      >
        {script.text}
      </p>
      <p className="text-xs text-[var(--color-text-secondary)]">
        About 20 seconds at a natural pace. Smile on the greeting, go up on the questions, sound pleased at the good news.
        Quiet room, phone a hand&apos;s width away, one clean take.
      </p>
    </section>
  );
}
