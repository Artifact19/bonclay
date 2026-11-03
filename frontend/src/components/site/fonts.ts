import { useEffect, useState, type RefObject } from "react";
import {
  Anek_Bangla,
  Anek_Devanagari,
  Anek_Gujarati,
  Anek_Gurmukhi,
  Anek_Kannada,
  Anek_Malayalam,
  Anek_Odia,
  Anek_Tamil,
  Anek_Telugu,
} from "next/font/google";

// Self-hosted Anek (Ek Type, SIL OFL 1.1) per script, the same superfamily as the Latin face in ./siteFont, so every
// language card renders in one designed face at one weight, instead of whatever the visitor's OS happens to ship (or tofu
// boxes). Anek has no Urdu (Arabic script) cut, so Urdu would need Noto Naskh/Nastaliq; no card uses it today.
// next/font needs literal options per call.
// Weight 400 only and preload: false: these faces are not preloaded. A face is only downloaded once text in its script
// is laid out with its class, and the sections that use them apply the class through useScriptFontsNear below, so
// the files arrive as the reader nears those sections, not on page load (font-weight 500 text falls back to 400).
const devanagari = Anek_Devanagari({ weight: "400", subsets: ["devanagari"], display: "swap", preload: false });
const bengali = Anek_Bangla({ weight: "400", subsets: ["bengali"], display: "swap", preload: false });
const gujarati = Anek_Gujarati({ weight: "400", subsets: ["gujarati"], display: "swap", preload: false });
const gurmukhi = Anek_Gurmukhi({ weight: "400", subsets: ["gurmukhi"], display: "swap", preload: false });
const tamil = Anek_Tamil({ weight: "400", subsets: ["tamil"], display: "swap", preload: false });
const telugu = Anek_Telugu({ weight: "400", subsets: ["telugu"], display: "swap", preload: false });
const kannada = Anek_Kannada({ weight: "400", subsets: ["kannada"], display: "swap", preload: false });
const malayalam = Anek_Malayalam({ weight: "400", subsets: ["malayalam"], display: "swap", preload: false });
const oriya = Anek_Odia({ weight: "400", subsets: ["oriya"], display: "swap", preload: false });

/** True once the element behind `ref` comes within `rootMargin` of the screen. The default, 1.5 viewports either side, is
 *  for long sections the reader scrolls toward; pass "0px" for something that should load only when it is on screen.
 *  Components that show script text apply a script's class only after this turns true, so until then the text uses the
 *  page font and the system's fallback for those glyphs. Server render and first client render both see false, so there
 *  is no hydration mismatch. It never turns back to false. Without IntersectionObserver it stays false, the same as
 *  without JS: the fallback glyphs are used and nothing is downloaded. */
export function useScriptFontsNear(ref: RefObject<Element | null>, rootMargin = "150% 0px"): boolean {
  const [near, setNear] = useState(false);
  useEffect(() => {
    if (near) return;
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        setNear(true);
        io.disconnect();
      },
      { rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref, rootMargin, near]);
  return near;
}

/** Language code -> font class for that language's script. */
export const SCRIPT_FONT: Record<string, string> = {
  hi: devanagari.className,
  mr: devanagari.className,
  bn: bengali.className,
  as: bengali.className,
  gu: gujarati.className,
  pa: gurmukhi.className,
  ta: tamil.className,
  te: telugu.className,
  kn: kannada.className,
  ml: malayalam.className,
  or: oriya.className,
};
