#!/usr/bin/env python3
"""Runs Pocket TTS multilingual (community fine-tune of Kyutai Pocket TTS) for bench/indic_benchmark.py.

Runs in its own virtualenv (backend/.venv-pocket: `pip install pocket-tts`), reads a jobs file and writes wavs plus a
results file; nothing here imports the backend. Weights are not bundled: download config.yaml, model.safetensors and
tokenizer.model from huggingface.co/Thanushttk/pocket-tts-multilingual into one folder (--model-dir).

    .venv-pocket/bin/python bench/pocket_worker.py JOBS.json OUT.json --model-dir DIR --ref REF.wav
JOBS.json: [{"id", "text", "wav"}]. OUT.json: {id: {"gen_seconds", "audio_seconds"} or {"error"}}.
"""
import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ref", required=True)
    a = ap.parse_args()
    import scipy.io.wavfile
    from pocket_tts import TTSModel
    jobs = json.load(open(a.jobs))
    ref = os.path.abspath(a.ref)
    os.chdir(a.model_dir)  # the config's weight and tokenizer paths are relative
    model = TTSModel.load_model(config="config.yaml")
    voice = model.get_state_for_audio_prompt(ref)
    res = {}

    def one(job):
        t0 = time.time()
        audio = model.generate_audio(voice, job["text"]).detach().cpu().numpy().squeeze()
        os.makedirs(os.path.dirname(job["wav"]), exist_ok=True)
        scipy.io.wavfile.write(job["wav"], model.sample_rate, audio)
        return {"gen_seconds": round(time.time() - t0, 2), "audio_seconds": round(len(audio) / model.sample_rate, 2)}

    try:
        one({"text": jobs[0]["text"], "wav": jobs[0]["wav"] + ".warmup.wav"})  # warm-up
        os.remove(jobs[0]["wav"] + ".warmup.wav")
    except Exception as e:  # noqa: BLE001
        print("warm-up failed:", e, flush=True)
    for job in jobs:
        try:
            res[job["id"]] = one(job)
        except Exception as e:  # noqa: BLE001
            res[job["id"]] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print("  pocket", job["id"], res[job["id"]], flush=True)
        json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
