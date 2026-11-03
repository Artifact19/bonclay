#!/usr/bin/env python3
"""AudioSeal robustness trial: synthesize Kokoro preset clips, watermark with the app's own functions, attack, detect.

    backend/.venv/bin/python backend/bench/watermark_attacks.py gen      # make 20 clips (needs Kokoro), clean + marked
    backend/.venv/bin/python backend/bench/watermark_attacks.py attack   # attack all, write attacked wavs
    backend/.venv/bin/python backend/bench/watermark_attacks.py detect   # run provenance.detect on everything
    backend/.venv/bin/python backend/bench/watermark_attacks.py mark     # re-watermark the existing clean clips (no Kokoro);
                                                                         # WM_LEGACY=1 uses the pre-scaling embed path
    backend/.venv/bin/python backend/bench/watermark_attacks.py quality  # watermark SNR + log-spectral distance (+ PESQ if installed)
    ... encodec step: see attack_encodec() (uses transformers EncodecModel, facebook/encodec_24khz, 6 kbps)

Only Kokoro preset voices are used. Work dir: env WM_WORK (default backend/bench/results/wm_work, gitignored).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import zlib

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)
WORK = os.environ.get("WM_WORK", os.path.join(HERE, "results", "wm_work"))
SR = 24000
FFMPEG = "ffmpeg"

import torch  # noqa: E402
torch.set_num_threads(3)
from services import provenance  # noqa: E402

EN = ["Doctor Meera Kulkarni will see you on Monday at nine thirty in the morning, so please bring your old reports.",
      "The train from Pune to Mumbai leaves at six fifteen and arrives before noon, and the seats are very comfortable.",
      "My sister Anjali bought twelve oranges and three loaves of bread yesterday, and then she walked home slowly.",
      "The consultation fee is four hundred rupees for the first visit, and follow up visits cost less than that.",
      "Tomorrow the weather in Bangalore will be warm with a light evening breeze, perfect for a walk by the lake.",
      "Rohan lives on the fourth floor of a quiet building near the lake, and he loves watching the birds at dawn.",
      "We have received your order and it will be delivered within five working days to the address you gave us.",
      "The quarterly meeting moved to Thursday afternoon, so please update your calendar and tell your whole team.",
      "Sunita planted two mango trees and seven rose bushes behind the house, and the garden now smells wonderful.",
      "Thank you for calling Sunrise Clinic, your call is important to us, and one of our staff will help you shortly."]
HI = ["डॉक्टर मीरा कुलकर्णी सोमवार को सुबह नौ बजे मरीजों को देखती हैं, कृपया अपनी पुरानी रिपोर्ट साथ लाएं।",
      "पुणे से मुंबई जाने वाली ट्रेन सुबह छह बजे निकलती है और दोपहर से पहले पहुंच जाती है।",
      "मेरी बहन अंजलि कल बाजार से बारह संतरे और तीन डबल रोटी लाई, फिर वह धीरे धीरे घर गई।",
      "पहली बार दिखाने की फीस चार सौ रुपये है, और दोबारा आने पर फीस इससे कम लगती है।",
      "कल बेंगलुरु में मौसम गर्म रहेगा और शाम को हल्की हवा चलेगी, झील के पास टहलना अच्छा रहेगा।",
      "रोहन झील के पास एक शांत इमारत की चौथी मंजिल पर रहता है और सुबह पक्षियों को देखना उसे पसंद है।",
      "आपका ऑर्डर मिल गया है और पांच कामकाजी दिनों में आपके बताए हुए पते पर पहुंच जाएगा।",
      "तिमाही बैठक गुरुवार दोपहर को होगी, इसलिए अपना कैलेंडर बदल लें और अपनी पूरी टीम को बता दें।",
      "सुनीता ने घर के पीछे आम के दो पेड़ और गुलाब के सात पौधे लगाए, अब बगीचा बहुत सुंदर लगता है।",
      "सनराइज क्लिनिक में फोन करने के लिए धन्यवाद, आपकी कॉल हमारे लिए महत्वपूर्ण है, कृपया थोड़ी प्रतीक्षा करें।"]


def legacy_watermark(x, sr, message):
    """The embed path before per-clip scaling (raw AudioSeal output added as is), kept to reproduce the 'before' numbers."""
    x16 = provenance._resample(x.astype(np.float32), sr, 16000)
    step = provenance.CHUNK_SECONDS * 16000
    wm16 = np.concatenate([provenance._embed16(x16[i:i + step], message) for i in range(0, x16.size, step)])
    wm = provenance._resample(wm16, 16000, sr)
    wm = np.pad(wm, (0, max(0, x.size - wm.size)))
    return np.clip(x + wm[:x.size], -1.0, 1.0).astype(np.float32)


def mark():
    """Re-watermark the clean clips in WORK with the current code (or the legacy path with WM_LEGACY=1)."""
    meta = json.load(open(os.path.join(WORK, "meta.json")))
    os.makedirs(os.path.join(WORK, "marked"), exist_ok=True)
    for m in meta:
        x, sr = sf.read(clip_path("clean", m["name"]), dtype="float32")
        if os.environ.get("WM_LEGACY"):
            y = legacy_watermark(x, sr, m["message"])
        else:
            y, info = provenance.watermark_ex(x, sr, m["message"])
            assert info["applied"], info
        sf.write(clip_path("marked", m["name"]), y, sr, subtype="FLOAT")
    print("marked", len(meta), "legacy" if os.environ.get("WM_LEGACY") else "current")


def active_rms(x):
    return provenance._speech_rms(x.astype(np.float32))


def log_spectral_distance(a, b, sr, n_fft=512):
    """Mean over frames of the RMS difference (dB) between the log power spectra of two signals."""
    from scipy.signal import stft
    _, _, A = stft(a, sr, nperseg=n_fft)
    _, _, B = stft(b, sr, nperseg=n_fft)
    la, lb = 10 * np.log10(np.abs(A) ** 2 + 1e-10), 10 * np.log10(np.abs(B) ** 2 + 1e-10)
    return float(np.mean(np.sqrt(np.mean((la - lb) ** 2, axis=0))))


def quality():
    """Imperceptibility of the marked clips in WORK versus their clean originals, per clip, as JSON rows."""
    meta = json.load(open(os.path.join(WORK, "meta.json")))
    try:
        from pesq import pesq
    except Exception:  # noqa: BLE001
        pesq = None
    rows = []
    for m in meta:
        x, sr = sf.read(clip_path("clean", m["name"]), dtype="float32")
        y, _ = sf.read(clip_path("marked", m["name"]), dtype="float32")
        w = y - x
        row = {"name": m["name"], "lang": m["lang"],
               "snr_db": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(w ** 2)))),
               "snr_active_db": float(20 * np.log10(active_rms(x) / np.sqrt(np.mean(w ** 2)))),
               "lsd_db": log_spectral_distance(x, y, sr)}
        if pesq is not None:
            x16, y16 = provenance._resample(x, sr, 16000), provenance._resample(y, sr, 16000)
            row["pesq_wb"] = float(pesq(16000, x16, y16, "wb"))
        rows.append(row)
    out = os.path.join(WORK, "quality.json")
    json.dump(rows, open(out, "w"), indent=1)
    for lang in ("en", "hi"):
        r = [q for q in rows if q["lang"] == lang]
        print(lang, {k: round(float(np.mean([q[k] for q in r])), 2) for k in r[0] if k not in ("name", "lang")})


def clip_path(kind, name):
    return os.path.join(WORK, kind, name + ".wav")


def gen():
    from services.espeak import configure_espeak
    configure_espeak()
    from kokoro import KPipeline
    os.makedirs(os.path.join(WORK, "clean"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "marked"), exist_ok=True)
    meta = []
    for lang, code, voice, texts in (("en", "a", "af_heart", EN), ("hi", "h", "hf_alpha", HI)):
        pipe = KPipeline(lang_code=code)
        for i, t in enumerate(texts):
            parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in pipe(t, voice=voice) if a is not None]
            x = np.concatenate(parts).astype(np.float32)
            name = f"{lang}{i:02d}"
            msg = provenance.message_for(f"wm-trial-{name}")
            marked, info = provenance.watermark_ex(x, SR, msg)
            assert info["applied"], info
            sf.write(clip_path("clean", name), x, SR, subtype="FLOAT")
            sf.write(clip_path("marked", name), marked, SR, subtype="FLOAT")
            meta.append({"name": name, "lang": lang, "voice": voice, "seconds": round(len(x) / SR, 2), "message": msg,
                         "snr_db": round(float(10 * np.log10(np.sum(x ** 2) / max(np.sum((marked - x) ** 2), 1e-12))), 1)})
            print(meta[-1], flush=True)
    json.dump(meta, open(os.path.join(WORK, "meta.json"), "w"), indent=1)


# ---------------------------------------------------------------- attacks (float32 mono 24 kHz in, same out)
def ff(x, in_args_out, ext):
    """Round trip through ffmpeg: write wav, run `ffmpeg -i in <args> out.<ext>`, then decode back to 24 kHz mono float."""
    with tempfile.TemporaryDirectory() as d:
        a, b, c = os.path.join(d, "a.wav"), os.path.join(d, "b." + ext), os.path.join(d, "c.wav")
        sf.write(a, x, SR, subtype="FLOAT")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", a] + in_args_out + [b], check=True)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", b, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", c], check=True)
        y, sr = sf.read(c, dtype="float32")
        assert sr == SR
        return y


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X)); f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return (y / np.std(y)).astype(np.float32)


def add_noise(x, snr_db, kind, seed):
    rng = np.random.default_rng(seed)
    nz = pink(len(x), rng) if kind == "pink" else rng.standard_normal(len(x)).astype(np.float32)
    p = np.mean(x ** 2)
    nz *= np.sqrt(p / (10 ** (snr_db / 10)))
    return (x + nz).astype(np.float32)


def music_like(n, seed):
    """Chord progression of harmonic tones with slow tremolo plus a little pink noise (a stand-in for background music)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    out = np.zeros(n)
    chords = [(220, 277.2, 329.6), (196, 246.9, 293.7), (174.6, 220, 261.6), (196, 246.9, 329.6)]
    seg = int(SR * 2.0)
    for k in range(0, n, seg):
        notes = chords[(k // seg) % len(chords)]
        tt = t[k:k + seg]
        env = np.minimum(1, np.minimum(np.arange(len(tt)), len(tt) - np.arange(len(tt))) / (0.1 * SR))
        for f0 in notes:
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)):
                out[k:k + seg] += amp * np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 6.28)) * env
    out *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    out += 0.1 * pink(n, rng) * np.std(out)
    return (out / np.std(out)).astype(np.float32)


def mix_music(x, snr_db, seed):
    m = music_like(len(x), seed)
    m *= np.sqrt(np.mean(x ** 2) / (10 ** (snr_db / 10)))
    return (x + m).astype(np.float32)


def pitch(x, semis):
    r = 2 ** (semis / 12)  # change pitch, keep duration: resample up/down then time-stretch back
    return ff(x, ["-af", f"asetrate={int(SR * r)},aresample={SR},atempo={1 / r:.6f}", "-ar", str(SR), "-c:a", "pcm_f32le"], "wav")


ATTACKS = {  # name -> function(x, seed)
    "none": lambda x, s: x,
    "mp3_64k": lambda x, s: ff(x, ["-c:a", "libmp3lame", "-b:a", "64k"], "mp3"),
    "mp3_32k": lambda x, s: ff(x, ["-c:a", "libmp3lame", "-b:a", "32k"], "mp3"),
    "opus_16k": lambda x, s: ff(x, ["-c:a", "libopus", "-b:a", "16k", "-application", "voip"], "ogg"),
    "mulaw_8k": lambda x, s: ff(x, ["-ar", "8000", "-c:a", "pcm_mulaw"], "wav"),
    "speed_+5%": lambda x, s: ff(x, ["-af", "atempo=1.05", "-c:a", "pcm_f32le"], "wav"),
    "speed_-5%": lambda x, s: ff(x, ["-af", "atempo=0.95", "-c:a", "pcm_f32le"], "wav"),
    "varispeed_+5%": lambda x, s: ff(x, ["-af", f"asetrate={int(SR * 1.05)},aresample={SR}", "-c:a", "pcm_f32le"], "wav"),
    "varispeed_-5%": lambda x, s: ff(x, ["-af", f"asetrate={int(SR * 0.95)},aresample={SR}", "-c:a", "pcm_f32le"], "wav"),
    "pitch_+1st": lambda x, s: pitch(x, 1),
    "pitch_-1st": lambda x, s: pitch(x, -1),
    "lowpass_3.4k": lambda x, s: ff(x, ["-af", "lowpass=f=3400:poles=2,lowpass=f=3400:poles=2", "-c:a", "pcm_f32le"], "wav"),
    "pink_20dB": lambda x, s: add_noise(x, 20, "pink", s),
    "pink_10dB": lambda x, s: add_noise(x, 10, "pink", s),
    "white_20dB": lambda x, s: add_noise(x, 20, "white", s),
    "white_10dB": lambda x, s: add_noise(x, 10, "white", s),
    "trim_0.5s": lambda x, s: x[int(0.5 * SR):],
    "music_bg_10dB": lambda x, s: mix_music(x, 10, s),
    "music_bg_20dB": lambda x, s: mix_music(x, 20, s),
    "whatsapp_chain": lambda x, s: ff(ff(x, ["-ar", "16000", "-c:a", "libopus", "-b:a", "16k", "-application", "voip"], "ogg"),
                                     ["-c:a", "libmp3lame", "-b:a", "64k"], "mp3"),
}


def attack_encodec(x, model_cache={}):
    from transformers import EncodecModel, AutoProcessor
    if "m" not in model_cache:
        model_cache["m"] = EncodecModel.from_pretrained("facebook/encodec_24khz").eval()
        model_cache["p"] = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    m, p = model_cache["m"], model_cache["p"]
    inp = p(raw_audio=x, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode():
        enc = m.encode(inp["input_values"], inp["padding_mask"], bandwidth=6.0)
        dec = m.decode(enc.audio_codes, enc.audio_scales, inp["padding_mask"])[0]
    return dec[0, 0].numpy().astype(np.float32)[:len(x)]


def attack_all():
    meta = json.load(open(os.path.join(WORK, "meta.json")))
    for kind in ("marked", "clean"):
        for m in meta:
            x, _ = sf.read(clip_path(kind, m["name"]), dtype="float32")
            for an, fn in dict(ATTACKS, **({"encodec_6k": attack_encodec} if os.environ.get("WM_ENCODEC") else {})).items():
                if os.environ.get("WM_ONLY") and an != os.environ["WM_ONLY"]:
                    continue
                out = os.path.join(WORK, "attacked", kind, an.replace("%", "pct"))
                os.makedirs(out, exist_ok=True)
                y = fn(x, zlib.crc32((m["name"] + an).encode()) % 100000) if an != "encodec_6k" else fn(x)
                sf.write(os.path.join(out, m["name"] + ".wav"), y, SR, subtype="FLOAT")
        print("attacked", kind, flush=True)


def detect_all():
    meta = {m["name"]: m for m in json.load(open(os.path.join(WORK, "meta.json")))}
    thr = provenance.detect_threshold()
    rows = []
    for kind in ("marked", "clean"):
        base = os.path.join(WORK, "attacked", kind)
        for an in sorted(os.listdir(base)):
            for fn in sorted(os.listdir(os.path.join(base, an))):
                name = fn[:-4]
                x, sr = sf.read(os.path.join(base, an, fn), dtype="float32")
                r = provenance.detect(x, sr)
                x16 = provenance._resample(x, sr, 16000)
                prob, raw_msg = provenance._detect16(x16)   # raw frame-level message even when below threshold
                want = meta[name]["message"]
                bitacc = None if raw_msg is None else 1 - bin(raw_msg ^ want).count("1") / 16
                rows.append({"kind": kind, "attack": an, "clip": name, "lang": meta[name]["lang"], "detected": r["detected"],
                             "prob": r["probability"], "msg_app": r["message"], "msg_exact": r["message"] == want,
                             "raw_msg_bitacc": bitacc, "thr": thr})
    json.dump(rows, open(os.path.join(WORK, "detect_rows.json"), "w"))
    print("rows", len(rows))


if __name__ == "__main__":
    t = time.time()
    {"gen": gen, "attack": attack_all, "detect": detect_all, "mark": mark, "quality": quality}[sys.argv[1]]()
    print("done in %.0fs" % (time.time() - t))
