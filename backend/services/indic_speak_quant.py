"""Indic-Speak with a quantized language model (MLX 4-bit): same interface as the model's own `TTS`, about 2 GB instead of 7.

Indic-Speak is a Llama-style language model (3.2B parameters, bf16) that predicts audio tokens, plus a small SNAC quantizer
and Vocos decoder. Only the language model is large. This class runs it with MLX (Apple GPU) from a 4-bit conversion and keeps
the model's own prompt format, token-to-audio decoding and sampling. Measured 2026-10-20 on 81 benchmark sentences
(docs/INDIC_MODEL_SELECTION.md): +0.010 read-back CER against bf16, about 3.7x faster, 2.05 GB peak.

Runs inside the worker (services/indic_speak_worker.py --mlx-model DIR) in a virtualenv that has torch, snac, transformers 5
and mlx-lm (scripts/setup_indic_speak_q4.sh). Not imported by the API server.

    python services/indic_speak_quant.py convert --snapshot SNAP --out DIR [--bits 4]   one-time conversion
"""
import argparse
import json
import os
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SR = 24_000


def fix_rope_config(model_dir: str) -> dict:
    """mlx-lm reads `rope_theta` / `rope_scaling`; this model's transformers-5 config only has `rope_parameters`. Without the
    fix mlx-lm silently uses rope_theta 10000 with no scaling and the model never stops talking. Idempotent."""
    path = Path(model_dir) / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    rp = cfg.get("rope_parameters")
    if rp and "rope_scaling" not in cfg:
        cfg["rope_theta"] = rp["rope_theta"]
        if rp.get("rope_type") == "llama3":
            cfg["rope_scaling"] = {"rope_type": "llama3", "type": "llama3", "factor": rp["factor"],
                                   "low_freq_factor": rp["low_freq_factor"], "high_freq_factor": rp["high_freq_factor"],
                                   "original_max_position_embeddings": rp["original_max_position_embeddings"]}
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def convert(snapshot: str, out: str, bits: int = 4, group_size: int = 64) -> None:
    """Quantize the language model with mlx-lm and apply `fix_rope_config`. The decoder is not converted (it stays torch)."""
    out_path = Path(out)
    if out_path.exists() and (out_path / "model.safetensors").exists():
        fix_rope_config(out)
        return
    subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", snapshot, "--mlx-path", out, "-q",
                    "--q-bits", str(bits), "--q-group-size", str(group_size)], check=True)
    fix_rope_config(out)


class QuantTTS:
    """Drop-in for inference.TTS: `tts(text, speaker=, style=, temperature=, top_p=, top_k=, max_new_tokens=, seed=)` -> float32 wav."""

    sample_rate = SR

    def __init__(self, snapshot: str, mlx_model: str, device: str = "cpu"):
        # MLX's GPU stream belongs to the thread that created it, and the worker's HTTP server answers each request on a new
        # thread ("There is no Stream(gpu, 1) in current thread"): load and generate on one dedicated thread.
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._ex.submit(self._load, snapshot, mlx_model, device).result()

    def _load(self, snapshot: str, mlx_model: str, device: str) -> None:
        import torch
        from mlx_lm import load
        from snac import SNAC
        from transformers import AutoTokenizer
        sys.path.insert(0, snapshot)
        import inference  # the model's own code from the snapshot (prompt format, token layout, Vocos loader)
        self._inf = inference
        self._torch = torch
        self.device = device  # the decoder is small; CPU keeps the Apple GPU free for the language model
        self.tok = AutoTokenizer.from_pretrained(snapshot)
        self.lm, _ = load(mlx_model)
        self.snac = SNAC.from_pretrained(inference.SNAC_REPO).to(device).eval()
        self.vocos = inference._load_vocos(os.path.join(snapshot, "vocos", "best.pt"), device)
        self._base = self.tok.convert_tokens_to_ids("<|snac_0|>")
        self._stops = {self.tok.convert_tokens_to_ids("<|end_of_speech|>"), self.tok.eos_token_id}

    def __call__(self, text: str, speaker: str = "", style: str = "", **kw) -> np.ndarray:
        with warnings.catch_warnings(record=True) as caught:  # warnings are per thread: carry the truncation warning back
            warnings.simplefilter("always")
            wav = self._ex.submit(self._generate, text, speaker, style, **kw).result()
        for w in caught:
            warnings.warn(w.message, w.category, stacklevel=2)
        return wav

    def _generate(self, text: str, speaker: str = "", style: str = "", *, temperature: float = 0.6, top_p: float = 0.9,
                  top_k: int = 50, max_new_tokens: int = 2520, seed=None, stock: bool = False) -> np.ndarray:
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        if seed is not None:
            mx.random.seed(int(seed))
        prompt = self._inf.build_prompt(self.tok, text, speaker, style)
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        ids = []
        for tok, _ in generate_step(mx.array(prompt), self.lm, max_tokens=max_new_tokens, sampler=sampler):
            tok = int(tok)
            if tok in self._stops:
                break
            ids.append(tok)
        if len(ids) >= max_new_tokens:  # same message and category as inference.TTS: the worker reports it as truncation
            warnings.warn(f"generation hit max_new_tokens ({max_new_tokens}) without emitting <|end_of_speech|>",
                          RuntimeWarning, stacklevel=2)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        torch = self._torch
        with torch.no_grad():
            codes = self._inf.ids_to_codes(ids, self._base, self.device)
            z_q = self.snac.quantizer.from_codes(codes)
            dec = self.snac.decoder if stock else self.vocos
            wav = dec(z_q.float() if not stock else z_q)
        return wav[0, 0].clamp(-1, 1).float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--snapshot", required=True), c.add_argument("--out", required=True)
    c.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    convert(a.snapshot, a.out, a.bits)
    print(f"[indic-speak-quant] ready: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
