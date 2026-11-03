#!/usr/bin/env bash
# One-time setup for voice cloning on Apple Silicon through MLX (GPU).
# Creates backend/.venv-mlx, installs mlx-audio, mlx-lm and mlx-whisper, and pre-downloads the Qwen3-TTS voice model,
# the local language model that runs agents when no Groq key is set, the Whisper models for speech recognition, and
# Kokoro-82M for MLX (the live preset voice; needs espeak-ng from Homebrew for Hindi, Spanish, French, Italian, Portuguese).
# Usage: bash backend/scripts/setup_mlx.sh   (works from any directory)
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "setup_mlx.sh: this setup is only for Apple Silicon Macs (macOS arm64)." >&2
    echo "Detected: $(uname -s) $(uname -m). On Windows or Linux, voice cloning uses PyTorch (CUDA when an NVIDIA GPU is present)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$BACKEND_DIR/.venv-mlx"
VENV_PY="$VENV_DIR/bin/python"
# 8-bit weights: same speaker similarity as bf16, faster, smaller (about 2.0 GB to download against 2.3 GB). Keep in step with DEFAULT_MODEL.
MODEL_ID="${MLX_CLONE_MODEL:-mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit}"
LLM_ID="${LOCAL_LLM_MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
KOKORO_ID="${MLX_KOKORO_MODEL:-mlx-community/Kokoro-82M-bf16}"

find_python() {
    local candidate
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

if ! BASE_PY="$(find_python)"; then
    echo "setup_mlx.sh: Python 3.10 or newer is required. Install one (for example: brew install python@3.12) and run this again." >&2
    exit 1
fi
echo "Using $("$BASE_PY" -c 'import sys; print(sys.version.split()[0])') from $BASE_PY"

if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating virtualenv at $VENV_DIR"
    "$BASE_PY" -m venv "$VENV_DIR"
fi

echo "Installing MLX packages (this can take a few minutes)"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install "mlx-audio[tts]" mlx-lm mlx-whisper soundfile numpy hf_transfer

# Kokoro on MLX needs misaki (its text front end: English G2P, spaCy, espeak bindings) and the spaCy English model.
# Pinned to the versions the PyTorch Kokoro path uses, so both engines phonemize identically.
echo "Installing the Kokoro text front end (misaki, spaCy English model)"
"$VENV_PY" -m pip install "misaki[en]==0.9.4"
"$VENV_PY" -m spacy download en_core_web_sm
# Mandarin and Japanese voices are optional (pyopenjtalk compiles from source and needs a C++ compiler):
# a failure here only means those two voices are spoken by PyTorch Kokoro instead.
"$VENV_PY" -m pip install cn2an jieba pypinyin pypinyin-dict ordered-set ||
    echo "Mandarin packages not installed; Mandarin falls back to PyTorch Kokoro." >&2
"$VENV_PY" -m pip install -r "$BACKEND_DIR/requirements-ja.txt" ||
    echo "Japanese packages not installed; Japanese falls back to PyTorch Kokoro." >&2

download() {
    local id="$1" attempt
    echo "Downloading $id"
    for attempt in 1 2 3; do
        if HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 "$VENV_PY" -c \
            "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$id"; then
            return 0
        fi
        if [[ "$attempt" -lt 3 ]]; then
            echo "Download of $id failed (attempt $attempt of 3). Retrying in 5 seconds..." >&2
            sleep 5
        fi
    done
    echo "setup_mlx.sh: could not download $id after 3 attempts. Check your network and run this again." >&2
    return 1
}
download "$MODEL_ID"
download "$LLM_ID"
download "$KOKORO_ID"
download "${MLX_ASR_FAST_MODEL:-mlx-community/whisper-small-mlx}"
download "${MLX_ASR_ACCURATE_MODEL:-mlx-community/whisper-large-v3-turbo}"

echo
echo "MLX voice cloning, Kokoro voices and the local language model are ready: $VENV_PY"
echo "Kokoro runs on MLX by default here (KOKORO_ENGINE=torch turns that off; PyTorch Kokoro stays the fallback)."
if ! command -v espeak-ng >/dev/null 2>&1 && [[ ! -f /opt/homebrew/lib/libespeak-ng.dylib ]]; then
    echo "Note: install espeak-ng (brew install espeak-ng) for the Hindi, Spanish, French, Italian and Portuguese voices." >&2
fi
echo "The backend finds this environment automatically; no extra setting is needed."
echo "Start BonClay as usual from the repo root:"
echo "  ./bonclay start"
