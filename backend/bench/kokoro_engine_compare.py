#!/usr/bin/env python3
"""Kokoro engines on the app's own code path: PyTorch KPipeline vs the MLX worker, measured INTERLEAVED.

Both engines are loaded once through the same function the API uses (`backend._kokoro_generate`, which takes either a
PyTorch pipeline or the MLX handle), warmed up, then for each round and each sentence each engine speaks the same
sentence in turn (engine order alternates every round) so a load spike hits both alike. Per sentence the median over
rounds is taken; the table reports p50 / mean of those per-sentence medians, and the audio length (same text, same
voice, so the lengths should match closely). Memory is process RSS from `ps`.

    backend/.venv/bin/python backend/bench/kokoro_engine_compare.py [--rounds 3] [--out file.json]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)
os.environ.setdefault("KOKORO_ENGINE", "torch")  # importing backend must not start the worker by itself

EN = [
    "Hello, thanks for calling Sunrise Family Clinic, how can I help you today?",
    "The clinic opens at nine in the morning on Monday and closes at six in the evening.",
    "Dr. Meera Kulkarni sees patients on Tuesday, Thursday and Saturday.",
    "A dentist consultation costs five hundred rupees, and the first visit includes a check-up.",
    "Yes, you can pay by UPI, by card, or in cash at the front desk.",
    "I can book you an appointment for tomorrow at eleven. Shall I go ahead?",
    "Please arrive ten minutes early and bring any previous reports with you.",
    "Blood test reports are usually ready within twenty-four hours.",
    "If you have chest pain, please call the emergency number right away.",
    "Is there anything else I can help you with today?",
]
HI = [
    "नमस्ते, सनराइज फैमिली क्लिनिक में आपका स्वागत है, मैं आपकी कैसे मदद कर सकती हूँ?",
    "क्लिनिक सोमवार को सुबह नौ बजे खुलता है और शाम छह बजे बंद होता है।",
    "डॉक्टर मीरा कुलकर्णी मंगलवार, गुरुवार और शनिवार को मरीज़ देखती हैं।",
    "दांतों के डॉक्टर की फीस पाँच सौ रुपये है, और पहली विज़िट में जांच शामिल है।",
    "जी हाँ, आप यूपीआई, कार्ड या फ्रंट डेस्क पर नकद से भुगतान कर सकते हैं।",
    "मैं आपके लिए कल सुबह ग्यारह बजे की अपॉइंटमेंट बुक कर सकती हूँ। क्या मैं बुक कर दूँ?",
    "कृपया दस मिनट पहले आ जाइए और अपनी पुरानी रिपोर्ट साथ लाइए।",
    "खून की जांच की रिपोर्ट आमतौर पर चौबीस घंटे में तैयार हो जाती है।",
    "अगर सीने में दर्द हो, तो कृपया तुरंत आपातकालीन नंबर पर फ़ोन करें।",
    "क्या आज मैं आपकी और कोई मदद कर सकती हूँ?",
]


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return round(int(out) / 1024, 1) if out else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()

    import backend
    from services import mlx_kokoro

    me = os.getpid()
    base_rss = rss_mb(me)
    torch_pipe = backend.AdvancedModelManager()._instantiate_model_sync("kokoro")  # KOKORO_ENGINE=torch: PyTorch
    handle = mlx_kokoro.MlxKokoro()
    engines = {"torch": torch_pipe, "mlx": handle}
    print(f"loaded: process RSS {base_rss} -> {rss_mb(me)} MB, MLX worker RSS {rss_mb(handle.proc.pid)} MB", flush=True)

    for lang in ("en", "hi"):  # warm-up: builds the language pipelines, compiles kernels
        for name, eng in engines.items():
            for _ in range(2):
                backend._kokoro_generate(eng, EN[0] if lang == "en" else HI[0], lang)

    results = {name: {"en": {}, "hi": {}} for name in engines}
    audio_s = {name: {"en": {}, "hi": {}} for name in engines}
    peak_worker = rss_mb(handle.proc.pid)
    for rnd in range(a.rounds):
        order = list(engines) if rnd % 2 == 0 else list(reversed(list(engines)))
        for lang, sentences in (("en", EN), ("hi", HI)):
            for i, text in enumerate(sentences):
                for name in order:
                    t0 = time.perf_counter()
                    wav = backend._kokoro_generate(engines[name], text, lang)
                    dt = time.perf_counter() - t0
                    results[name][lang].setdefault(i, []).append(dt)
                    audio_s[name][lang][i] = len(wav) / 24000
                peak_worker = max(peak_worker, rss_mb(handle.proc.pid))
    assert handle.last_engine == "mlx", f"MLX handle fell back to PyTorch: {handle.last_error}"

    summary = {}
    print(f"\n{'':10}{'engine':8}{'p50 s':>8}{'mean s':>8}{'min':>8}{'max':>8}{'audio s':>9}")
    for lang in ("en", "hi"):
        for name in engines:
            per_sentence = [statistics.median(results[name][lang][i]) for i in sorted(results[name][lang])]
            summary.setdefault(lang, {})[name] = {
                "p50_s": round(statistics.median(per_sentence), 3), "mean_s": round(statistics.mean(per_sentence), 3),
                "min_s": round(min(per_sentence), 3), "max_s": round(max(per_sentence), 3),
                "audio_s_mean": round(statistics.mean(audio_s[name][lang].values()), 2), "per_sentence_s": [round(x, 3) for x in per_sentence]}
            s = summary[lang][name]
            print(f"{lang:10}{name:8}{s['p50_s']:>8}{s['mean_s']:>8}{s['min_s']:>8}{s['max_s']:>8}{s['audio_s_mean']:>9}")
        t, m = summary[lang]["torch"]["p50_s"], summary[lang]["mlx"]["p50_s"]
        summary[lang]["mlx_faster_pct"] = round(100 * (t - m) / t, 1)
        print(f"{'':10}MLX is {summary[lang]['mlx_faster_pct']}% faster at p50")
    mem = {"main_process_rss_mb_before": base_rss, "main_process_rss_mb_after": rss_mb(me),
           "mlx_worker_rss_mb": rss_mb(handle.proc.pid), "mlx_worker_rss_peak_mb": peak_worker}
    print(f"memory: {mem}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rounds": a.rounds, "summary": summary, "memory": mem}, f, indent=2, ensure_ascii=False)
    handle.close()


if __name__ == "__main__":
    main()
