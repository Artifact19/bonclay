#!/usr/bin/env python3
"""Trial: can DhVaani-0.5 (ARTPARK-IISc, ZipVoice-based) clone a voice into Indian languages beyond Hindi?

Three stages, each run with the python that has the right packages (nothing here touches the running API server;
the references are synthetic Kokoro voices, never a real person's):

  refs        (backend/.venv)            make the two synthetic reference clips with Kokoro
  gen-dhvaani (DhVaani scratch venv)     5 sentences x 5 languages x 2 references on the Apple GPU (MPS)
  gen-chatterbox (backend/.venv)         the same Hindi sentences through services/chatterbox_worker.py, the way
                                         services/chatterbox_clone.py starts it (own port, CPU, stopped afterwards)
  score       (backend/.venv)            read-back character error rate (mlx-whisper large-v3-turbo, in the
                                         backend/.venv-mlx python) and ECAPA speaker similarity (services/speaker.py)
  report                                 markdown tables from the three result files

    WORK=/path/to/work/dir
    backend/.venv/bin/python backend/bench/indic_clone_trial.py refs --work $WORK
    $WORK/venv/bin/python -I backend/bench/indic_clone_trial.py gen-dhvaani --work $WORK --model-dir $WORK/dl
    backend/.venv/bin/python backend/bench/indic_clone_trial.py gen-chatterbox --work $WORK
    backend/.venv/bin/python backend/bench/indic_clone_trial.py score --work $WORK
    backend/.venv/bin/python backend/bench/indic_clone_trial.py report --work $WORK

Reference text must be given to DhVaani (ZipVoice conditions on prompt audio + its transcript); Chatterbox needs
only the audio.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)

REFS = {
    "en": ("a", "am_michael",
           "Good morning, you have reached the front desk, please tell me your name and the day you would like to visit."),
    "hi": ("h", "hf_alpha", "नमस्ते, आप रिसेप्शन पर पहुंचे हैं, कृपया अपना नाम और मिलने का दिन बताइए।"),
}

# Simple clinic / appointment sentences. "nums" are the number words each sentence contains (Whisper writes them as
# digits; the digits in a transcript are replaced by these before scoring so a correct reading is not penalised).
SENTENCES = {
    "hi": [
        ("डॉक्टर साहब कल सुबह दस बजे आपको क्लिनिक में देखेंगे।", ["दस"]),
        ("कृपया अपनी पुरानी रिपोर्ट और दवाइयों की पर्ची साथ लेकर आइए।", []),
        ("आपकी अपॉइंटमेंट बुधवार शाम पांच बजे के लिए पक्की कर दी गई है।", ["पांच"]),
        ("पहली बार दिखाने की फीस चार सौ रुपये है और भुगतान काउंटर पर होगा।", ["चार सौ"]),
        ("अगर आप समय पर नहीं पहुंच सकते तो कृपया हमें पहले फोन करके बता दीजिए।", []),
    ],
    "mr": [
        ("डॉक्टर उद्या सकाळी दहा वाजता तुम्हाला क्लिनिकमध्ये तपासतील.", ["दहा"]),
        ("कृपया तुमचे जुने रिपोर्ट आणि औषधांची चिठ्ठी सोबत घेऊन या.", []),
        ("तुमची भेट बुधवारी संध्याकाळी पाच वाजता निश्चित करण्यात आली आहे.", ["पाच"]),
        ("पहिल्या भेटीची फी चारशे रुपये आहे आणि पैसे काउंटरवर भरावे लागतील.", ["चारशे"]),
        ("जर तुम्ही वेळेवर पोहोचू शकत नसाल तर कृपया आम्हाला आधी फोन करून सांगा.", []),
    ],
    "ta": [
        ("மருத்துவர் நாளை காலை பத்து மணிக்கு உங்களை கிளினிக்கில் பார்ப்பார்.", ["பத்து"]),
        ("தயவுசெய்து உங்கள் பழைய அறிக்கைகளையும் மருந்துச் சீட்டையும் கொண்டு வாருங்கள்.", []),
        ("உங்கள் சந்திப்பு புதன்கிழமை மாலை ஐந்து மணிக்கு உறுதி செய்யப்பட்டுள்ளது.", ["ஐந்து"]),
        ("முதல் முறை பார்ப்பதற்கான கட்டணம் நானூறு ரூபாய், பணத்தை கவுண்டரில் செலுத்த வேண்டும்.", ["நானூறு"]),
        ("நீங்கள் நேரத்திற்கு வர முடியாவிட்டால் தயவுசெய்து முன்கூட்டியே எங்களுக்கு தொலைபேசியில் தெரிவியுங்கள்.", []),
    ],
    "bn": [
        ("ডাক্তারবাবু আগামীকাল সকাল দশটায় আপনাকে ক্লিনিকে এসে দেখা করতে বলেছেন।", ["দশ"]),
        ("দয়া করে আপনার পুরোনো রিপোর্ট আর ওষুধের প্রেসক্রিপশন সঙ্গে নিয়ে আসবেন।", []),
        ("আপনার অ্যাপয়েন্টমেন্ট বুধবার বিকেল পাঁচটার জন্য নিশ্চিত করা হয়েছে।", ["পাঁচ"]),
        ("প্রথমবার দেখানোর ফি চারশো টাকা এবং টাকা কাউন্টারে জমা দিতে হবে।", ["চারশো"]),
        ("আপনি যদি সময়মতো পৌঁছাতে না পারেন তাহলে দয়া করে আগে আমাদের ফোন করে জানাবেন।", []),
    ],
    "te": [
        ("డాక్టర్ గారు రేపు ఉదయం పది గంటలకు మిమ్మల్ని క్లినిక్ వద్ద చూస్తారు.", ["పది"]),
        ("దయచేసి మీ పాత రిపోర్టులు మరియు మందుల చీటీ తీసుకుని రండి.", []),
        ("మీ అపాయింట్మెంట్ బుధవారం సాయంత్రం ఐదు గంటలకు ఖరారు చేయబడింది.", ["ఐదు"]),
        ("మొదటిసారి చూపించుకునే ఫీజు నాలుగు వందల రూపాయలు, డబ్బులు కౌంటర్ వద్ద చెల్లించాలి.", ["నాలుగు వందల"]),
        ("మీరు సమయానికి రాలేకపోతే దయచేసి ముందుగా మాకు ఫోన్ చేసి చెప్పండి.", []),
    ],
}
LANGS = list(SENTENCES)
MLX_PY = os.path.join(BACKEND, ".venv-mlx", "bin", "python")
CB_PY = os.path.join(BACKEND, ".venv-chatterbox", "bin", "python")
ASR_MODEL = "mlx-community/whisper-large-v3-turbo"


def paths(work):
    return {"refs": os.path.join(work, "refs"), "wavs": os.path.join(work, "wavs"),
            "gen_d": os.path.join(work, "results_dhvaani.json"), "gen_c": os.path.join(work, "results_chatterbox.json"),
            "score": os.path.join(work, "results_score.json")}


def wav_path(work, engine, ref, lang, i):
    return os.path.join(work, "wavs", engine, ref, f"{lang}_{i + 1}.wav")


# --------------------------------------------------------------------------------------------- refs

def cmd_refs(a):
    sys.path.insert(0, BACKEND)
    from services.espeak import configure_espeak
    configure_espeak()
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline
    os.makedirs(paths(a.work)["refs"], exist_ok=True)
    meta = {}
    for k, (code, voice, text) in REFS.items():
        pipe = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
        parts = [x.numpy() if hasattr(x, "numpy") else np.asarray(x) for _, _, x in pipe(text, voice=voice, speed=1.0)
                 if x is not None]
        wav = np.concatenate(parts)
        sf.write(os.path.join(paths(a.work)["refs"], f"ref_{k}.wav"), wav, 24000)
        meta[k] = {"voice": voice, "text": text, "seconds": round(len(wav) / 24000, 2)}
    json.dump(meta, open(os.path.join(paths(a.work)["refs"], "refs.json"), "w"), ensure_ascii=False, indent=1)
    print(meta)


# --------------------------------------------------------------------------------------------- DhVaani

def cmd_gen_dhvaani(a):
    import soundfile as sf
    sys.path.insert(0, a.model_dir)
    import torch
    import dhvaani
    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    tts = dhvaani.DhVaani(model_dir=a.model_dir, device=device)
    load_s = time.time() - t0
    refs = json.load(open(os.path.join(paths(a.work)["refs"], "refs.json")))
    out = {"engine": "dhvaani", "device": device, "num_step": a.num_step, "load_seconds": round(load_s, 1),
           "torch": torch.__version__, "loadavg_start": os.getloadavg(), "items": []}
    tmp = os.path.join(a.work, "warmup.wav")
    t0 = time.time()
    tts.synthesize(SENTENCES["hi"][0][0], os.path.join(paths(a.work)["refs"], "ref_en.wav"), refs["en"]["text"], tmp,
                   num_step=a.num_step)
    out["warmup_seconds"] = round(time.time() - t0, 1)
    for ref in ("en", "hi"):
        ref_wav = os.path.join(paths(a.work)["refs"], f"ref_{ref}.wav")
        for lang in LANGS:
            for i, (text, _nums) in enumerate(SENTENCES[lang]):
                dst = wav_path(a.work, "dhvaani", ref, lang, i)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                item = {"ref": ref, "lang": lang, "idx": i, "text": text, "wav": dst, "error": None,
                        "loadavg": round(os.getloadavg()[0], 2)}
                t0 = time.time()
                try:
                    tts.synthesize(text, ref_wav, refs[ref]["text"], dst, num_step=a.num_step)
                    item["gen_seconds"] = round(time.time() - t0, 2)
                    audio, sr = sf.read(dst, dtype="float32")
                    item["audio_seconds"] = round(len(audio) / sr, 2)
                    item["rtf"] = round(item["gen_seconds"] / max(item["audio_seconds"], 1e-6), 3)
                except Exception as e:  # noqa: BLE001
                    item["gen_seconds"] = round(time.time() - t0, 2)
                    item["error"] = f"{type(e).__name__}: {e}"[:300]
                out["items"].append(item)
                print(ref, lang, i + 1, item.get("gen_seconds"), item.get("audio_seconds"), item["error"], flush=True)
                json.dump(out, open(paths(a.work)["gen_d"], "w"), ensure_ascii=False, indent=1)
    out["loadavg_end"] = os.getloadavg()
    json.dump(out, open(paths(a.work)["gen_d"], "w"), ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------------------------- Chatterbox

def cmd_gen_chatterbox(a):
    import socket
    import urllib.request
    if not os.path.isfile(CB_PY):
        sys.exit("backend/.venv-chatterbox does not exist")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.Popen([CB_PY, os.path.join(BACKEND, "services", "chatterbox_worker.py"), "--port", str(port),
                             "--model", "ResembleAI/chatterbox"], env=env)
    out = {"engine": "chatterbox", "device": "cpu", "loadavg_start": os.getloadavg(), "items": []}
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                sys.exit(f"chatterbox worker exited {proc.returncode}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if json.load(r).get("ok"):
                        break
            except OSError:
                time.sleep(1)
        out["load_seconds"] = round(time.time() - t0, 1)
        import io
        import soundfile as sf
        for ref in ("en", "hi"):
            ref_wav = os.path.join(paths(a.work)["refs"], f"ref_{ref}.wav")
            for i, (text, _n) in enumerate(SENTENCES["hi"]):
                dst = wav_path(a.work, "chatterbox", ref, "hi", i)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                item = {"ref": ref, "lang": "hi", "idx": i, "text": text, "wav": dst, "error": None,
                        "loadavg": round(os.getloadavg()[0], 2)}
                req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", headers={"Content-Type": "application/json"},
                                             data=json.dumps({"text": text, "language": "hi", "ref_audio": ref_wav}).encode())
                t0 = time.time()
                try:
                    with urllib.request.urlopen(req, timeout=1800) as r:
                        data = r.read()
                    item["gen_seconds"] = round(time.time() - t0, 2)
                    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
                    sf.write(dst, audio, sr)
                    item["audio_seconds"] = round(len(audio) / sr, 2)
                    item["rtf"] = round(item["gen_seconds"] / max(item["audio_seconds"], 1e-6), 3)
                except Exception as e:  # noqa: BLE001
                    item["gen_seconds"] = round(time.time() - t0, 2)
                    item["error"] = f"{type(e).__name__}: {e}"[:300]
                out["items"].append(item)
                print("chatterbox", ref, i + 1, item.get("gen_seconds"), item.get("audio_seconds"), item["error"], flush=True)
                json.dump(out, open(paths(a.work)["gen_c"], "w"), ensure_ascii=False, indent=1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    out["loadavg_end"] = os.getloadavg()
    json.dump(out, open(paths(a.work)["gen_c"], "w"), ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------------------------- scoring

_ASR_SCRIPT = r"""
import json, sys
import mlx_whisper
jobs = json.load(open(sys.argv[1]))
res = {}
for j in jobs:
    r = mlx_whisper.transcribe(j["wav"], path_or_hf_repo=sys.argv[3], language=j["lang"], fp16=True)
    res[j["wav"]] = (r.get("text") or "").strip()
    print(j["wav"], res[j["wav"]], flush=True)
json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False)
"""
sys.path.insert(0, HERE)  # python -I does not add the script folder
from scoring import cer, letters as _letters, normalise_hyp  # noqa: E402,F401  (the one CER definition)


def load_audio_16k(path):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def cmd_score(a):
    sys.path.insert(0, BACKEND)
    import numpy as np
    from services import speaker
    P = paths(a.work)
    items = []
    for key, name in (("gen_d", "dhvaani"), ("gen_c", "chatterbox")):
        if os.path.exists(P[key]):
            for it in json.load(open(P[key]))["items"]:
                if not it["error"]:
                    items.append({**it, "engine": name})
    # 1. transcribe everything in one mlx-whisper process (model loaded once)
    jobs = [{"wav": it["wav"], "lang": it["lang"]} for it in items]
    jf, rf = os.path.join(a.work, "asr_jobs.json"), os.path.join(a.work, "asr_out.json")
    json.dump(jobs, open(jf, "w"))
    t0 = time.time()
    subprocess.run([MLX_PY, "-c", _ASR_SCRIPT, jf, rf, ASR_MODEL], check=True)
    asr_seconds = time.time() - t0
    hyps = json.load(open(rf))
    # 2. ECAPA similarity to the reference
    ref_emb = {}
    for k in REFS:
        ref_emb[k] = speaker.embed(load_audio_16k(os.path.join(P["refs"], f"ref_{k}.wav")))
    if any(v is None for v in ref_emb.values()):
        sys.exit("ECAPA model unavailable")

    def cos(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    out = {"asr_model": ASR_MODEL, "asr_seconds": round(asr_seconds, 1), "speaker_threshold": speaker.threshold(),
           "control_ref_en_vs_ref_hi": round(cos(ref_emb["en"], ref_emb["hi"]), 3), "items": []}
    for it in items:
        nums = SENTENCES[it["lang"]][it["idx"]][1]
        hyp = hyps[it["wav"]]
        emb = speaker.embed(load_audio_16k(it["wav"]))
        out["items"].append({"engine": it["engine"], "ref": it["ref"], "lang": it["lang"], "idx": it["idx"],
                             "hyp": hyp, "cer": cer(it["text"], normalise_hyp(hyp, nums)),
                             "cer_raw": cer(it["text"], hyp),
                             "sim": cos(emb, ref_emb[it["ref"]]),
                             "sim_other_ref": cos(emb, ref_emb["hi" if it["ref"] == "en" else "en"])})
    json.dump(out, open(P["score"], "w"), ensure_ascii=False, indent=1)
    print("scored", len(out["items"]))


# --------------------------------------------------------------------------------------------- report

FAIL_CER = 0.5


def _mean(v):
    v = [x for x in v if x is not None]
    return statistics.mean(v) if v else float("nan")


def _med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else float("nan")


def cmd_report(a):
    P = paths(a.work)
    gens = {}
    for key, name in (("gen_d", "dhvaani"), ("gen_c", "chatterbox")):
        if os.path.exists(P[key]):
            gens[name] = json.load(open(P[key]))
    sc = json.load(open(P["score"]))
    sidx = {(s["engine"], s["ref"], s["lang"], s["idx"]): s for s in sc["items"]}
    rows = ["| engine | ref | lang | n | RTF mean | gen s mean | CER mean | CER median | ECAPA mean | failures |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, g in gens.items():
        for ref in ("en", "hi"):
            for lang in LANGS:
                its = [i for i in g["items"] if i["ref"] == ref and i["lang"] == lang]
                if not its:
                    continue
                ss = [sidx.get((name, ref, lang, i["idx"])) for i in its]
                cers = [s["cer"] for s in ss if s]
                fails = sum(1 for i, s in zip(its, ss) if i["error"] or not s or (s["cer"] or 0) > FAIL_CER)
                rows.append(f"| {name} | {ref} | {lang} | {len(its)} | {_mean([i.get('rtf') for i in its]):.2f} | "
                            f"{_mean([i.get('gen_seconds') for i in its]):.1f} | {_mean(cers):.2f} | {_med(cers):.2f} | "
                            f"{_mean([s['sim'] for s in ss if s]):.2f} | {fails}/{len(its)} |")
    print("\n".join(rows))
    print(f"\nECAPA control (ref_en vs ref_hi, two different voices): {sc['control_ref_en_vs_ref_hi']}; "
          f"consent threshold {sc['speaker_threshold']}")
    for name, g in gens.items():
        print(name, "device", g["device"], "load", g.get("load_seconds"), "warmup", g.get("warmup_seconds"),
              "loadavg", g["loadavg_start"], "->", g.get("loadavg_end"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["refs", "gen-dhvaani", "gen-chatterbox", "score", "report"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", help="local snapshot of ARTPARK-IISc/DhVaani-0.5 (gen-dhvaani)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-step", type=int, default=16)
    a = ap.parse_args()
    {"refs": cmd_refs, "gen-dhvaani": cmd_gen_dhvaani, "gen-chatterbox": cmd_gen_chatterbox, "score": cmd_score,
     "report": cmd_report}[a.stage](a)


if __name__ == "__main__":
    main()
