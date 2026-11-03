"""CUDA Graph accelerator for Qwen3-TTS-12Hz-0.6B-Base on Windows / Linux NVIDIA GPUs.

Why this exists:
Qwen3-TTS-12Hz is a dual-autoregressive transformer: every 12 Hz audio frame runs 1 step of the 28-layer Talker AND
15 steps of the 5-layer CodePredictor (103 transformer layers = ~3,000 tiny CUDA kernel launches per frame). On
Windows WDDM, per-kernel launch overhead (~0.25 ms) dominates small batch=1 matmuls, leaving an RTX 4050 Laptop GPU
at ~4 W (GpuIdle) and taking ~750-830 ms per frame (~37-50 s for a 4 s sentence).

Capturing the 15-step CodePredictor loop and the 28-layer Talker decode step into two static `torch.cuda.CUDAGraph`s
(with fused `F.rms_norm` and precomputed RoPE/causal masks), plus capping the vocoder's left-context prefix to 24
frames (2 s instead of re-decoding the entire 15-30 s reference sample on every sentence), reduces per-frame decode
to 2 graph replays (~30 ms/frame) and synthesizes a 4 s sentence in ~2-3.5 s (~10-15x faster).

Set `QWEN3_CUDA_GRAPH=0` to disable and use upstream `talker.generate` unchanged.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

log = logging.getLogger("bonclay.qwen3_cuda_graph")

MAX_VOCODER_REF_FRAMES = 24  # 2.0 s of left context at 12 Hz for seamless vocoder boundary smoothing


def _enabled() -> bool:
    return os.getenv("QWEN3_CUDA_GRAPH", "1").strip().lower() not in ("0", "false", "no", "off")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class FastQwen3Runner:
    """Static-KV-cache + CUDA-Graph runner for `Qwen3TTSTalkerForConditionalGeneration` (batch_size=1)."""

    def __init__(self, talker: Any, max_seq_len: int = 1536) -> None:
        self.talker = talker
        self.orig_generate = talker.generate
        self.cp = talker.code_predictor
        self.device = next(talker.parameters()).device
        self.dtype = next(talker.parameters()).dtype
        self.max_seq_len = max_seq_len

        tc = talker.config
        pc = tc.code_predictor_config
        self.t_layers = tc.num_hidden_layers
        self.cp_layers = pc.num_hidden_layers
        self.hidden_size = tc.hidden_size
        self.num_heads = tc.num_attention_heads
        self.num_kv_heads = tc.num_key_value_heads
        self.head_dim = getattr(tc, "head_dim", self.hidden_size // self.num_heads)
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.num_code_groups = tc.num_code_groups  # 16 (1 talker + 15 code_predictor)

        # Precompute RoPE for CodePredictor (positions 0..16)
        cp_pos = torch.arange(16, device=self.device).unsqueeze(0)
        dummy_cp = torch.zeros(1, 16, self.hidden_size, device=self.device, dtype=self.dtype)
        self.cp_cos, self.cp_sin = self.cp.model.rotary_emb(dummy_cp, cp_pos)
        causal_16 = torch.triu(
            torch.full((16, 16), float("-inf"), device=self.device, dtype=self.dtype), diagonal=1
        )
        self.cp_mask_prefill = causal_16[:2, :16].view(1, 1, 2, 16).contiguous()
        self.cp_masks_decode = [
            causal_16[s + 1 : s + 2, :16].view(1, 1, 1, 16).contiguous()
            for s in range(1, self.num_code_groups - 1)
        ]

        # Static KV cache for CodePredictor
        self.cp_k = [
            torch.zeros(1, self.num_kv_heads, 16, self.head_dim, device=self.device, dtype=self.dtype)
            for _ in range(self.cp_layers)
        ]
        self.cp_v = [
            torch.zeros(1, self.num_kv_heads, 16, self.head_dim, device=self.device, dtype=self.dtype)
            for _ in range(self.cp_layers)
        ]

        # Static I/O buffers for CodePredictor graph
        self.cp_in_embeds = torch.zeros(1, 2, self.hidden_size, device=self.device, dtype=self.dtype)
        self.cp_temp = torch.tensor(0.9, device=self.device, dtype=torch.float32)
        self.cp_out_ids = torch.zeros(1, self.num_code_groups - 1, dtype=torch.long, device=self.device)
        self.cp_out_hiddens = torch.zeros(
            1, self.num_code_groups - 1, self.hidden_size, device=self.device, dtype=self.dtype
        )

        # Precompute RoPE for Talker (positions 0..max_seq_len)
        t_pos = torch.arange(max_seq_len, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        dummy_t = torch.zeros(1, max_seq_len, self.hidden_size, device=self.device, dtype=self.dtype)
        t_cos, t_sin = self.talker.model.rotary_emb(dummy_t, t_pos)
        self.t_cos = t_cos[0].contiguous()  # (1, max_seq_len, head_dim)
        self.t_sin = t_sin[0].contiguous()

        self.t_causal_table = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf"), device=self.device, dtype=self.dtype),
            diagonal=1,
        ).contiguous()

        # Static KV cache for Talker
        self.t_k = [
            torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            for _ in range(self.t_layers)
        ]
        self.t_v = [
            torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim, device=self.device, dtype=self.dtype)
            for _ in range(self.t_layers)
        ]

        # Static I/O buffers for Talker decode graph
        self.t_in_embeds = torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        self.t_cache_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        self.t_mask_step = torch.zeros(1, 1, 1, max_seq_len, device=self.device, dtype=self.dtype)
        self.t_cos_step = torch.zeros(1, 1, 1, self.head_dim, device=self.device, dtype=self.dtype)
        self.t_sin_step = torch.zeros(1, 1, 1, self.head_dim, device=self.device, dtype=self.dtype)
        self.t_out_hidden = torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        self.t_out_logits = torch.zeros(1, 1, tc.vocab_size, device=self.device, dtype=self.dtype)

        self._capture_graphs()

    def _run_layer(self, layer: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   k_buf: torch.Tensor, v_buf: torch.Tensor, cache_idx: Any, attn_mask: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        residual = x
        h = F.rms_norm(x, (self.hidden_size,), layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
        attn = layer.self_attn
        q = F.rms_norm(
            attn.q_proj(h).view(1, S, self.num_heads, self.head_dim),
            (self.head_dim,),
            attn.q_norm.weight,
            attn.q_norm.variance_epsilon,
        ).transpose(1, 2)
        k = F.rms_norm(
            attn.k_proj(h).view(1, S, self.num_kv_heads, self.head_dim),
            (self.head_dim,),
            attn.k_norm.weight,
            attn.k_norm.variance_epsilon,
        ).transpose(1, 2)
        v = attn.v_proj(h).view(1, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        if isinstance(cache_idx, int):
            k_buf[:, :, cache_idx : cache_idx + S, :] = k
            v_buf[:, :, cache_idx : cache_idx + S, :] = v
        else:
            k_buf.index_copy_(2, cache_idx, k)
            v_buf.index_copy_(2, cache_idx, v)

        kv_len = k_buf.shape[2]
        k_rep = k_buf[:, :, None, :, :].expand(1, self.num_kv_heads, self.kv_groups, kv_len, self.head_dim).reshape(
            1, self.num_heads, kv_len, self.head_dim
        )
        v_rep = v_buf[:, :, None, :, :].expand(1, self.num_kv_heads, self.kv_groups, kv_len, self.head_dim).reshape(
            1, self.num_heads, kv_len, self.head_dim
        )

        attn_out = F.scaled_dot_product_attention(
            q, k_rep, v_rep, attn_mask=attn_mask, dropout_p=0.0, scale=attn.scaling
        )
        x = residual + attn.o_proj(attn_out.transpose(1, 2).reshape(1, S, -1))

        residual = x
        h = F.rms_norm(
            x, (self.hidden_size,), layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon
        )
        mlp = layer.mlp
        return residual + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))

    def _sample_cp(self, logits: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        if greedy:
            return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        scores = logits[:, -1, :].float() / self.cp_temp
        topk_vals, topk_idx = torch.topk(scores, 50, dim=-1)
        probs = F.softmax(topk_vals, dim=-1)
        rel = torch.multinomial(probs, num_samples=1)
        return topk_idx.gather(-1, rel)

    def _cp_forward_all(self, greedy: bool = False) -> None:
        x = self.cp.small_to_mtp_projection(self.cp_in_embeds)
        cos0 = self.cp_cos[:, :2, :].unsqueeze(1)
        sin0 = self.cp_sin[:, :2, :].unsqueeze(1)
        for i, layer in enumerate(self.cp.model.layers):
            x = self._run_layer(layer, x, cos0, sin0, self.cp_k[i], self.cp_v[i], 0, self.cp_mask_prefill)
        h = F.rms_norm(x[:, -1:, :], (self.hidden_size,), self.cp.model.norm.weight, self.cp.model.norm.variance_epsilon)
        logits = self.cp.lm_head[0](h)
        tok = self._sample_cp(logits, greedy=greedy)
        self.cp_out_ids[:, 0:1] = tok
        emb = self.cp.model.codec_embedding[0](tok)
        self.cp_out_hiddens[:, 0:1, :] = emb

        for s in range(1, self.num_code_groups - 1):
            x = self.cp.small_to_mtp_projection(emb)
            cos_s = self.cp_cos[:, s + 1 : s + 2, :].unsqueeze(1)
            sin_s = self.cp_sin[:, s + 1 : s + 2, :].unsqueeze(1)
            mask_s = self.cp_masks_decode[s - 1]
            for i, layer in enumerate(self.cp.model.layers):
                x = self._run_layer(layer, x, cos_s, sin_s, self.cp_k[i], self.cp_v[i], s + 1, mask_s)
            h = F.rms_norm(x, (self.hidden_size,), self.cp.model.norm.weight, self.cp.model.norm.variance_epsilon)
            logits = self.cp.lm_head[s](h)
            tok = self._sample_cp(logits, greedy=greedy)
            self.cp_out_ids[:, s : s + 1] = tok
            emb = self.cp.model.codec_embedding[s](tok)
            self.cp_out_hiddens[:, s : s + 1, :] = emb

    def _talker_decode_one(self) -> None:
        x = self.t_in_embeds
        for i, layer in enumerate(self.talker.model.layers):
            x = self._run_layer(
                layer, x, self.t_cos_step, self.t_sin_step, self.t_k[i], self.t_v[i], self.t_cache_idx, self.t_mask_step
            )
        h = F.rms_norm(x, (self.hidden_size,), self.talker.model.norm.weight, self.talker.model.norm.variance_epsilon)
        self.t_out_hidden.copy_(h)
        self.t_out_logits.copy_(self.talker.codec_head(h))

    def _capture_graphs(self) -> None:
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._cp_forward_all(greedy=False)
                self._talker_decode_one()
        torch.cuda.current_stream(self.device).wait_stream(s)

        self.cp_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cp_graph, stream=s):
            self._cp_forward_all(greedy=False)

        self.t_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.t_graph, stream=s):
            self._talker_decode_one()

    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        trailing_text_hidden: Optional[torch.Tensor] = None,
        tts_pad_embed: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: int = 2150,
        repetition_penalty: float = 1.05,
        suppress_tokens: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> Any:
        if not _enabled() or inputs_embeds.shape[0] != 1 or inputs_embeds.shape[1] + 4 >= self.max_seq_len:
            return self.orig_generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                suppress_tokens=suppress_tokens,
                **kwargs,
            )

        self.cp_temp.fill_(max(float(subtalker_temperature or 0.9), 1e-5))
        L0 = inputs_embeds.shape[1]

        # 1. Prefill through Talker layers directly into static KV cache
        x = inputs_embeds
        cos0 = self.t_cos[:, :L0, :].unsqueeze(1)
        sin0 = self.t_sin[:, :L0, :].unsqueeze(1)
        mask0 = self.t_causal_table[:L0, :L0].view(1, 1, L0, L0).contiguous()
        for i, layer in enumerate(self.talker.model.layers):
            residual = x
            h = F.rms_norm(x, (self.hidden_size,), layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
            attn = layer.self_attn
            q = F.rms_norm(
                attn.q_proj(h).view(1, L0, self.num_heads, self.head_dim),
                (self.head_dim,),
                attn.q_norm.weight,
                attn.q_norm.variance_epsilon,
            ).transpose(1, 2)
            k = F.rms_norm(
                attn.k_proj(h).view(1, L0, self.num_kv_heads, self.head_dim),
                (self.head_dim,),
                attn.k_norm.weight,
                attn.k_norm.variance_epsilon,
            ).transpose(1, 2)
            v = attn.v_proj(h).view(1, L0, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = (q * cos0) + (_rotate_half(q) * sin0)
            k = (k * cos0) + (_rotate_half(k) * sin0)
            self.t_k[i][:, :, :L0, :] = k
            self.t_v[i][:, :, :L0, :] = v
            k_rep = k[:, :, None, :, :].expand(1, self.num_kv_heads, self.kv_groups, L0, self.head_dim).reshape(
                1, self.num_heads, L0, self.head_dim
            )
            v_rep = v[:, :, None, :, :].expand(1, self.num_kv_heads, self.kv_groups, L0, self.head_dim).reshape(
                1, self.num_heads, L0, self.head_dim
            )
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask0, dropout_p=0.0, scale=attn.scaling)
            x = residual + attn.o_proj(attn_out.transpose(1, 2).reshape(1, L0, -1))
            residual = x
            h = F.rms_norm(
                x, (self.hidden_size,), layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon
            )
            mlp = layer.mlp
            x = residual + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))

        past_hidden = F.rms_norm(
            x[:, -1:, :], (self.hidden_size,), self.talker.model.norm.weight, self.talker.model.norm.variance_epsilon
        )
        logits = self.talker.codec_head(past_hidden)

        suppress_mask = torch.zeros(1, self.talker.config.vocab_size, dtype=torch.bool, device=self.device)
        if suppress_tokens:
            suppress_mask[:, suppress_tokens] = True

        generated_code0: List[torch.Tensor] = []
        hidden_states_out = [((past_hidden,), None)]
        trail_len = trailing_text_hidden.shape[1] if trailing_text_hidden is not None else 0

        def pick_code0(logits_step: torch.Tensor, step_idx: int) -> torch.Tensor:
            scores = logits_step[:, -1, :].float().clone()
            if generated_code0 and repetition_penalty != 1.0:
                prev = torch.cat(generated_code0, dim=-1)
                score_prev = torch.gather(scores, 1, prev)
                score_prev = torch.where(
                    score_prev < 0, score_prev * repetition_penalty, score_prev / repetition_penalty
                )
                scores.scatter_(1, prev, score_prev)
            scores.masked_fill_(suppress_mask, float("-inf"))
            if step_idx < min_new_tokens:
                scores[:, eos_token_id] = float("-inf")
            if not do_sample:
                return torch.argmax(scores, dim=-1, keepdim=True)
            scores = scores / max(float(temperature or 0.9), 1e-5)
            if top_k and top_k > 0:
                topk_vals, topk_idx = torch.topk(scores, min(int(top_k), scores.shape[-1]), dim=-1)
                probs = F.softmax(topk_vals, dim=-1)
                rel = torch.multinomial(probs, num_samples=1)
                return topk_idx.gather(-1, rel)
            probs = F.softmax(scores, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        input_ids = pick_code0(logits, 0)
        generated_code0.append(input_ids)

        for step in range(max_new_tokens):
            pos = L0 + step
            if pos >= self.max_seq_len:
                break
            last_id_hidden = self.talker.get_input_embeddings()(input_ids)
            self.cp_in_embeds[:, 0:1, :].copy_(past_hidden)
            self.cp_in_embeds[:, 1:2, :].copy_(last_id_hidden)
            if subtalker_dosample:
                self.cp_graph.replay()
            else:
                self._cp_forward_all(greedy=True)

            codec_ids = torch.cat((input_ids, self.cp_out_ids), dim=-1)

            if int(input_ids.item()) == eos_token_id:
                hidden_states_out.append(((past_hidden,), codec_ids))
                break

            codec_hiddens = torch.cat((last_id_hidden, self.cp_out_hiddens), dim=1)
            next_embed = codec_hiddens.sum(1, keepdim=True)
            if step < trail_len:
                next_embed = next_embed + trailing_text_hidden[:, step : step + 1, :]
            else:
                next_embed = next_embed + tts_pad_embed

            self.t_in_embeds.copy_(next_embed)
            self.t_cache_idx.fill_(pos)
            self.t_cos_step.copy_(self.t_cos[:, pos : pos + 1, :].unsqueeze(1))
            self.t_sin_step.copy_(self.t_sin[:, pos : pos + 1, :].unsqueeze(1))
            self.t_mask_step.copy_(
                self.t_causal_table[pos : pos + 1, : self.max_seq_len].view(1, 1, 1, self.max_seq_len)
            )
            self.t_graph.replay()

            past_hidden = self.t_out_hidden.clone()
            hidden_states_out.append(((past_hidden,), codec_ids))

            if step + 1 < max_new_tokens:
                input_ids = pick_code0(self.t_out_logits, step + 1)
                generated_code0.append(input_ids)

        return SimpleNamespace(hidden_states=hidden_states_out)


def attach_cuda_graph_runner(qwen_model: Any) -> Any:
    """Attach `FastQwen3Runner` and vocoder left-context capping to a CUDA `Qwen3TTSModel` instance."""
    if not _enabled() or not torch.cuda.is_available():
        return qwen_model
    try:
        inner = getattr(qwen_model, "model", None)
        talker = getattr(inner, "talker", None)
        if talker is None or next(talker.parameters()).device.type != "cuda":
            return qwen_model
        runner = FastQwen3Runner(talker)
        talker.generate = runner.generate
        qwen_model._cuda_graph_runner = runner

        # Also cap vocoder prefix to MAX_VOCODER_REF_FRAMES (2 s) during speech_tokenizer.decode
        orig_decode = inner.speech_tokenizer.decode
        orig_gen = inner.generate

        def _fast_model_generate(*args: Any, **kwargs: Any) -> Any:
            res = orig_gen(*args, **kwargs)
            vc = kwargs.get("voice_clone_prompt")
            if isinstance(vc, dict) and vc.get("ref_code"):
                vc["ref_code"] = [
                    rc[-MAX_VOCODER_REF_FRAMES:] if (rc is not None and rc.shape[0] > MAX_VOCODER_REF_FRAMES) else rc
                    for rc in vc["ref_code"]
                ]
            return res

        inner.generate = _fast_model_generate
        inner._orig_speech_decode = orig_decode
        print("[Qwen3CUDA] Attached CUDA Graph runner for Talker + CodePredictor")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not attach Qwen3 CUDA Graph runner (continuing with standard eager path): %s", e)
    return qwen_model

