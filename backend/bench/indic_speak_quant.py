#!/usr/bin/env python3
"""Indic-Speak with a quantized language model (MLX 4-bit / 8-bit), for bench/indic_benchmark.py.

Indic-Speak = Llama-style language model (3.2B parameters, bf16, 6.1 GB) that predicts audio tokens + a small SNAC/Vocos
decoder. Only the language model is large, so only it is swapped:

  gen     (backend/.venv-mlx)    MLX model (converted with `mlx_lm convert -q`) generates audio-token ids. Same prompt
                                 format, sampling (temperature 0.6, top-p 0.9, top-k 50), stop tokens, token cap and
                                 runaway split-and-retry as services/indic_speak.py.
  decode  (backend/.venv-indic)  token ids -> SNAC codes -> Vocos -> 24 kHz wav, with the model's own inference.py pieces;
                                 the 6 GB language model is NOT loaded here.

    .venv-mlx/bin/python bench/indic_speak_quant.py gen    JOBS.json IDS.json --mlx-model DIR --snapshot SNAP
    .venv-indic/bin/python bench/indic_speak_quant.py decode IDS.json OUT.json --snapshot SNAP
JOBS.json: [{"id", "text", "speaker", "wav"}]; OUT.json: {id: {"lm_seconds", "decode_seconds", "audio_seconds",
"tokens", "peak_gb"} or {"error"}}. Quality is whatever the read-back says: quantization can change the sound.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PAUSE_S, SR = 0.25, 24000


def _prompt(tok, text, speaker):
    """Copy of inference.build_prompt (that module imports torch, which the MLX environment does not have)."""
    tid = tok.convert_tokens_to_ids
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    meta = [tid("<|speaker>")] + enc(speaker.strip()) + [tid("<speaker|>")] + enc("\n") if speaker.strip() else []
    body = [tok.bos_token_id] + meta + enc(text) + [tid("<|eot_id|>")]
    return [tid("<|start_of_human|>")] + body + [tid("<|end_of_human|>"), tid("<|start_of_ai|>"), tid("<|start_of_speech|>")]


def cmd_gen(a):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from transformers import AutoTokenizer
    from services import indic_speak as isp
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    model, _ = load(a.mlx_model)
    base = tok.convert_tokens_to_ids("<|snac_0|>")
    stops = {tok.convert_tokens_to_ids("<|end_of_speech|>"), tok.eos_token_id}
    sampler = make_sampler(temp=0.6, top_p=0.9, top_k=50)

    def speak(text, speaker, depth=0):
        cap = isp.token_cap(text)
        ids, t0 = [], time.time()
        for t, _ in generate_step(mx.array(_prompt(tok, text, speaker)), model, max_tokens=cap, sampler=sampler):
            t = int(t)
            if t in stops:
                break
            ids.append(t)
        runaway = len(ids) >= cap
        sec = time.time() - t0
        if not runaway:
            return [ids], sec
        if depth >= isp.MAX_SPLIT_DEPTH or len(text) < 30:
            raise RuntimeError("could not speak this text cleanly")
        left, right = isp._halve(text)
        (l, s1), (r, s2) = speak(left, speaker, depth + 1), speak(right, speaker, depth + 1)
        return l + r, sec + s1 + s2

    jobs, out = json.load(open(a.jobs)), {}
    try:
        speak("नमस्ते, आज मौसम अच्छा है।", jobs[0]["speaker"])  # warm-up; sampling can occasionally run on
    except RuntimeError:
        pass
    mx.reset_peak_memory()
    for job in jobs:
        try:
            chunks, lm_s = [], 0.0
            for c in isp.split_for_speech(job["text"]):
                ids_list, s = speak(c, job["speaker"])
                chunks += ids_list
                lm_s += s
            out[job["id"]] = {"chunks": chunks, "lm_seconds": round(lm_s, 2), "tokens": sum(map(len, chunks)),
                              "peak_gb": round(mx.get_peak_memory() / 2 ** 30, 2), "base": base}
        except Exception as e:  # noqa: BLE001
            out[job["id"]] = {"error": f"{type(e).__name__}: {e}"[:200]}
        print("  gen", job["id"], {k: v for k, v in out[job["id"]].items() if k != "chunks"}, flush=True)
        json.dump(out, open(a.out, "w"))


def cmd_decode(a):
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, a.snapshot)
    from inference import _load_vocos, ids_to_codes
    from snac import SNAC
    dev = "cpu"
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(dev).eval()
    vocos = _load_vocos(os.path.join(a.snapshot, "vocos", "best.pt"), dev)
    jobs = {j["id"]: j for j in json.load(open(a.jobs_for_paths))}
    gen, res = json.load(open(a.ids)), {}
    pause = np.zeros(int(SR * PAUSE_S), dtype=np.float32)
    for jid, g in gen.items():
        if "error" in g:
            res[jid] = {"error": g["error"]}
            continue
        t0 = time.time()
        parts = []
        with torch.no_grad():
            for ids in g["chunks"]:
                codes = ids_to_codes(ids, g["base"], dev)
                wav = vocos(snac.quantizer.from_codes(codes).float())[0, 0].clamp(-1, 1).float().cpu().numpy()
                if parts:
                    parts.append(pause)
                parts.append(wav)
        audio = np.concatenate(parts)
        os.makedirs(os.path.dirname(jobs[jid]["wav"]), exist_ok=True)
        sf.write(jobs[jid]["wav"], audio, SR)
        res[jid] = {"lm_seconds": g["lm_seconds"], "decode_seconds": round(time.time() - t0, 2),
                    "audio_seconds": round(len(audio) / SR, 2), "tokens": g["tokens"], "peak_gb": g["peak_gb"]}
        print("  decode", jid, res[jid], flush=True)
    json.dump(res, open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("jobs"), g.add_argument("out")
    g.add_argument("--mlx-model", required=True), g.add_argument("--snapshot", required=True)
    d = sub.add_parser("decode")
    d.add_argument("ids"), d.add_argument("out")
    d.add_argument("--snapshot", required=True), d.add_argument("--jobs-for-paths", dest="jobs_for_paths", required=True)
    a = ap.parse_args()
    {"gen": cmd_gen, "decode": cmd_decode}[a.cmd](a)


if __name__ == "__main__":
    main()
