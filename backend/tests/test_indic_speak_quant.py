import json
import sys
from pathlib import Path

from core import memory_guard
from services import indic_speak as isp
from services import indic_speak_quant as q


def _cfg(tmp_path, **extra):
    cfg = {"architectures": ["LlamaForCausalLM"], "rope_parameters": {
        "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
        "rope_theta": 500000.0, "rope_type": "llama3"}, **extra}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def test_rope_fix_adds_the_keys_mlx_lm_reads(tmp_path):
    _cfg(tmp_path)
    out = q.fix_rope_config(str(tmp_path))
    assert out["rope_theta"] == 500000.0
    assert out["rope_scaling"]["rope_type"] == "llama3" and out["rope_scaling"]["factor"] == 32.0
    assert json.loads((tmp_path / "config.json").read_text())["rope_theta"] == 500000.0


def test_rope_fix_is_idempotent_and_leaves_other_configs_alone(tmp_path):
    _cfg(tmp_path)
    first = q.fix_rope_config(str(tmp_path))
    assert q.fix_rope_config(str(tmp_path)) == first
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"rope_theta": 10000.0}), encoding="utf-8")
    assert q.fix_rope_config(str(other)) == {"rope_theta": 10000.0}


def test_quantized_mode_needs_the_variable_and_a_complete_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("INDIC_SPEAK_MLX_MODEL", raising=False)
    assert isp.quantized_model_dir() is None
    monkeypatch.setenv("INDIC_SPEAK_MLX_MODEL", str(tmp_path))  # folder without model.safetensors
    assert isp.quantized_model_dir() is None
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert isp.quantized_model_dir() == str(tmp_path)


def test_quantized_mode_uses_its_own_virtualenv_and_a_smaller_memory_figure(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"x")
    monkeypatch.delenv("INDIC_SPEAK_PYTHON", raising=False)
    monkeypatch.delenv("MEMORY_NEED_INDIC_SPEAK_MB", raising=False)
    monkeypatch.delenv("INDIC_SPEAK_MLX_MODEL", raising=False)
    assert memory_guard.need_mb("indic_speak") == memory_guard.DEFAULT_NEED_MB["indic_speak"] == 8400   # measured 27 Oct 2026
    monkeypatch.setenv("INDIC_SPEAK_MLX_MODEL", str(tmp_path))
    assert memory_guard.need_mb("indic_speak") == 3500
    monkeypatch.setenv("MEMORY_NEED_INDIC_SPEAK_MB", "4000")  # an explicit override still wins
    assert memory_guard.need_mb("indic_speak") == 4000
    py = isp.indic_speak_python()
    assert py is None or ".venv-indic-q4" in py


def test_worker_has_the_mlx_flag():
    src = (Path(__file__).resolve().parent.parent / "services" / "indic_speak_worker.py").read_text(encoding="utf-8")
    assert "--mlx-model" in src and "QuantTTS" in src
