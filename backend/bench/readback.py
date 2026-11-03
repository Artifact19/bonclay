#!/usr/bin/env python3
"""Intelligibility by read-back: speak a known sentence, transcribe the audio with the API's own ASR, score it.

  English  -> word error rate (WER, jiwer): lowercase, punctuation removed, numbers compared as written.
  Hindi    -> character error rate (CER, jiwer): Devanagari letters and marks only, punctuation and spaces removed.

Real-time factor (RTF) = generation seconds / audio seconds. Lower is faster; below 1.0 is faster than real time.
Generation seconds are the client wall time of the generate request (includes the server saving the file).

Engines: kokoro-en (af_heart), kokoro-hi (hf_alpha), clone-en (needs --voice-id), indic-hi (Indic-Speak, slow:
about 8x real time on an Apple M4 and a multi-GB model to load on first use).

    backend/.venv/bin/python backend/bench/readback.py --engines kokoro-en,kokoro-hi
    backend/.venv/bin/python backend/bench/readback.py --quick
"""
import argparse
import re
import sys
import unicodedata

import jiwer

import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # services.numwords for norm_hi

from scoring import letters  # noqa: E402
from common import (LIMITS, Api, ApiError, RateGate, add_common_args, err_text, fmt, kokoro_voices, new_result,
                    pick_voice, print_header, require_api, stats, wav_info, write_result)

SENTENCES_EN = [
    "Doctor Meera Kulkarni will see you on Monday at nine thirty in the morning.",
    "The train from Pune to Mumbai leaves at six fifteen and arrives before noon.",
    "Please bring your old reports and arrive ten minutes before your appointment.",
    "My sister Anjali bought twelve oranges and three loaves of bread yesterday.",
    "The consultation fee is four hundred rupees for the first visit.",
    "Tomorrow the weather in Bangalore will be warm with a light evening breeze.",
    "Rohan lives on the fourth floor of a quiet building near the lake.",
    "We have received your order and it will be delivered within five working days.",
    "The quarterly meeting moved to Thursday afternoon, so please update your calendar.",
    "Sunita planted two mango trees and seven rose bushes behind the house.",
    "Our new office opens in January, and everyone is invited to the celebration.",
    "The library closes at eight in the evening on every weekday except Friday.",
    "Imran drove two hundred kilometres to visit his grandmother in Nashik.",
    "Please keep your seat belt fastened until the aircraft has come to a complete stop.",
    "The recipe needs three cups of flour, one cup of sugar and two eggs.",
    "Thank you for calling Sunrise Clinic, your call is important to us.",
    "A small blue bicycle was leaning against the old wall near the market.",
    "Priya finished the marathon in four hours and twenty minutes, which surprised everyone.",
    "The museum is open from ten in the morning until five in the afternoon.",
    "Could you please confirm your name, your age and your preferred day?",
]
SENTENCES_HI = [
    "डॉक्टर मीरा कुलकर्णी सोमवार को सुबह नौ बजे मरीजों को देखती हैं।",
    "पुणे से मुंबई जाने वाली ट्रेन सुबह छह बजे निकलती है।",
    "कृपया अपनी पुरानी रिपोर्ट साथ लाएं और अपॉइंटमेंट से दस मिनट पहले पहुंचें।",
    "मेरी बहन अंजलि कल बाजार से बारह संतरे और तीन डबल रोटी लाई।",
    "पहली बार दिखाने की फीस चार सौ रुपये है।",
    "कल बेंगलुरु में मौसम गर्म रहेगा और शाम को हल्की हवा चलेगी।",
    "रोहन झील के पास एक शांत इमारत की चौथी मंजिल पर रहता है।",
    "आपका ऑर्डर मिल गया है और पांच कामकाजी दिनों में पहुंच जाएगा।",
    "तिमाही बैठक गुरुवार दोपहर को होगी, इसलिए अपना कैलेंडर बदल लें।",
    "सुनीता ने घर के पीछे आम के दो पेड़ और गुलाब के सात पौधे लगाए।",
    "हमारा नया दफ्तर जनवरी में खुलेगा और सभी को जश्न में आना है।",
    "पुस्तकालय शुक्रवार को छोड़कर हर दिन शाम आठ बजे बंद हो जाता है।",
    "इमरान अपनी दादी से मिलने नासिक तक दो सौ किलोमीटर गाड़ी चलाकर गया।",
    "जब तक विमान पूरी तरह रुक न जाए, कृपया अपनी सीट बेल्ट बांधे रखें।",
    "इस व्यंजन के लिए तीन कप आटा, एक कप चीनी और दो अंडे चाहिए।",
    "सनराइज क्लिनिक में फोन करने के लिए धन्यवाद, आपकी कॉल हमारे लिए महत्वपूर्ण है।",
    "एक छोटी नीली साइकिल बाजार के पास पुरानी दीवार से टिकी हुई थी।",
    "प्रिया ने मैराथन चार घंटे बीस मिनट में पूरी की, जिससे सब हैरान रह गए।",
    "संग्रहालय सुबह दस बजे से शाम पांच बजे तक खुला रहता है।",
    "कृपया अपना नाम, अपनी उम्र और मिलने का पसंदीदा दिन बताइए।",
]

ENGINES = {
    "kokoro-en": {"lang": "en", "metric": "WER", "voice": "af_heart"},
    "kokoro-hi": {"lang": "hi", "metric": "CER", "voice": "hf_alpha"},
    "clone-en": {"lang": "en", "metric": "WER", "voice": "(cloned voice)"},
    "indic-hi": {"lang": "hi", "metric": "CER", "voice": "Indic-Speak Hindi preset"},
}


def norm_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)  # hyphens and punctuation become spaces; numbers stay as written
    s = s.replace("'", "")
    return re.sub(r"\s+", " ", s).strip()




def norm_hi(s: str) -> str:
    """Devanagari letters and vowel signs only (scoring.letters with script="devanagari"), with digits first written
    as Hindi number words (services/numwords): Whisper writes "400 रुपये" for a spoken "चार सौ रुपये", and without this
    a correctly spoken number scored as four wrong characters (22 Oct 2026: 5 of 10 Hindi sentences were hit)."""
    try:
        from services import numwords  # noqa: PLC0415
        s = numwords.normalize(s, "hi")
    except ImportError:
        pass
    return letters(s, script="devanagari")


def score(lang: str, ref: str, hyp: str):
    if lang == "en":
        r, h = norm_en(ref), norm_en(hyp)
        if not h:
            return 1.0, r, h
        return float(jiwer.wer(r, h)), r, h
    r, h = norm_hi(ref), norm_hi(hyp)
    if not h:
        return 1.0, r, h
    return float(jiwer.cer(r, h)), r, h


def generate(api: Api, engine: str, text: str, ctx: dict, gates: dict):
    """Returns (wav bytes, generation seconds). Raises ApiError."""
    if engine in ("kokoro-en", "kokoro-hi"):
        r, secs, _ = api.post("/api/kokoro/generate", json={"voice": ctx["voice"], "text": text}, timeout=180)
    elif engine == "clone-en":
        gates["voices_generate"].wait()
        r, secs, _ = api.post("/api/voices/generate", data={"text": text, "voice_id": ctx["voice_id"], "language": "en"},
                              timeout=600)
    else:  # indic-hi
        gates["indic_generate"].wait()
        r, secs, _ = api.post("/api/indic/generate", json={"text": text, "speaker": ctx["speaker"], "language": "hi"},
                              timeout=1800)
    if r.status_code != 200:
        raise ApiError(r.status_code, err_text(r))
    return r.content, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--engines", default="kokoro-en,kokoro-hi", help="comma list of: " + ", ".join(ENGINES))
    ap.add_argument("--voice-id", help="cloned voice id (required for clone-en)")
    ap.add_argument("--indic-speaker", default=None, help="Indic-Speak voice (default: the first Hindi voice)")
    ap.add_argument("-n", "--sentences", type=int, default=None, help="sentences per engine (default 20; quick 3)")
    a = ap.parse_args()
    n = min(a.sentences if a.sentences is not None else (3 if a.quick else 20), 20)
    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    for e in engines:
        if e not in ENGINES:
            sys.exit(f"Unknown engine {e!r}; choose from {', '.join(ENGINES)}")
    if "clone-en" in engines and not a.voice_id:
        sys.exit("clone-en needs --voice-id (an existing cloned voice)")
    if "indic-hi" in engines:
        print("WARNING: indic-hi is slow (about 8x real time on an Apple M4; the first call loads a multi-GB model "
              "and can take minutes). Rate limit: 6 requests/minute.")

    api = Api(a.api)
    health = require_api(api)
    settings = {"quick": a.quick, "engines": engines, "sentences_per_engine": n, "voice_id": a.voice_id,
                "rtf_definition": "generation seconds / audio seconds (client wall time of the generate request)",
                "wer": "jiwer, lowercase, punctuation removed, numbers compared as written (reference spells numbers "
                       "out; a transcript with digits counts as errors)",
                "cer": "jiwer, Devanagari letters and marks only; digits in the transcript become Hindi number words; punctuation and whitespace removed"}
    res = new_result("readback", api, health, settings)
    print_header("Read-back intelligibility", res)

    avail = kokoro_voices(api)
    ctx_by_engine = {}
    for e in engines:
        if e == "kokoro-en":
            v = pick_voice(avail, ["af_heart"]) or sys.exit("Kokoro voice af_heart is not available")
            ctx_by_engine[e] = {"voice": v}
        elif e == "kokoro-hi":
            v = pick_voice(avail, ["hf_alpha"]) or sys.exit("Kokoro Hindi voice hf_alpha is not available")
            ctx_by_engine[e] = {"voice": v}
        elif e == "clone-en":
            ctx_by_engine[e] = {"voice_id": a.voice_id}
        else:
            r, _, _ = api.get("/api/indic/voices")
            j = r.json() if r.status_code == 200 else {}
            if not j.get("available"):
                print(f"skipping indic-hi: engine unavailable ({j.get('reason')})")
                engines.remove(e)
                continue
            hi = [v["id"] for v in j.get("voices", []) if v.get("language_code") == "hi"]
            sp = a.indic_speaker or (hi[0] if hi else None) or sys.exit("No Indic-Speak Hindi voice found")
            ctx_by_engine[e] = {"speaker": sp}
            ENGINES[e]["voice"] = sp
    settings["engines"] = engines
    settings["voices"] = {e: ctx_by_engine[e] for e in engines}

    gates = {k: RateGate(LIMITS[k]) for k in ("transcribe", "voices_generate", "indic_generate")}
    try:
        for e in engines:
            lang, metric = ENGINES[e]["lang"], ENGINES[e]["metric"]
            sentences = (SENTENCES_EN if lang == "en" else SENTENCES_HI)[:n]
            print(f"\n-- {e} ({metric}), {len(sentences)} sentences")
            for i, text in enumerate(sentences):
                item = {"engine": e, "index": i, "language": lang, "text": text, "metric": metric}
                try:
                    wav, gen_s = generate(api, e, text, ctx_by_engine[e], gates)
                    info = wav_info(wav)
                    item.update({"gen_s": round(gen_s, 3), "audio_s": round(info["seconds"], 3),
                                 "rtf": round(gen_s / info["seconds"], 3) if info["seconds"] else None})
                    gates["transcribe"].wait()
                    r, asr_s, _ = api.post("/api/transcribe", files={"audio_file": ("readback.wav", wav, "audio/wav")},
                                           data={"language": lang}, timeout=300)
                    if r.status_code != 200:
                        raise ApiError(r.status_code, "transcribe: " + err_text(r))
                    j = r.json()
                    sc, ref_n, hyp_n = score(lang, text, j.get("text", ""))
                    item.update({"transcript": j.get("text"), "asr_model": j.get("model"),
                                 "asr_language": j.get("language"), "asr_s": round(asr_s, 3), "error_rate": round(sc, 4),
                                 "ref_normalized": ref_n, "hyp_normalized": hyp_n})
                except Exception as ex:  # noqa: BLE001
                    item["failure"] = f"{type(ex).__name__}: {ex}"
                res["items"].append(item)
                print(f"  {i + 1:2d}. {metric} {fmt(item.get('error_rate'), 3)}  rtf {fmt(item.get('rtf'), 2)}  "
                      f"{item.get('failure', '')}", flush=True)
    finally:
        summarize(res)
        print_summary(res)
        print(f"\nraw results: {write_result(res)}")


def summarize(res: dict) -> None:
    out = {}
    for e in res["settings"]["engines"]:
        rows = [i for i in res["items"] if i["engine"] == e]
        ok = [r for r in rows if "error_rate" in r]
        rates = [r["error_rate"] for r in ok]
        rtf = [r["rtf"] for r in ok if r.get("rtf") is not None]
        models = sorted({r.get("asr_model") for r in ok if r.get("asr_model")})
        out[e] = {"metric": ENGINES[e]["metric"], "attempted": len(rows), "scored": len(ok),
                  "failures": len(rows) - len(ok), "mean_error": stats(rates)["mean"],
                  "median_error": stats(rates)["p50"], "mean_rtf": stats(rtf)["mean"],
                  "median_rtf": stats(rtf)["p50"], "asr_models": models,
                  "worst": [{"text": r["text"], "heard": r.get("transcript"), "error": r["error_rate"]}
                            for r in sorted(ok, key=lambda r: -r["error_rate"])[:3]]}
    res["summary"] = out


def print_summary(res: dict) -> None:
    print("\nsummary (RTF = generation seconds / audio seconds)")
    for e, s in res["summary"].items():
        print(f"  {e:10s} {s['metric']} mean {fmt(s['mean_error'], 3)} median {fmt(s['median_error'], 3)} | "
              f"RTF mean {fmt(s['mean_rtf'], 2)} | scored {s['scored']}/{s['attempted']} failures {s['failures']} | "
              f"ASR: {', '.join(s['asr_models']) or 'unknown'}")


if __name__ == "__main__":
    main()
