#!/usr/bin/env bash
# One-time setup for DhVaani-0.5 (ARTPARK-IISc): voice cloning in 22+ Indian languages (Hindi, Marathi, Tamil, Bengali,
# Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia, Urdu, Assamese and more) from a few seconds of a consented voice.
# Creates backend/.venv-dhvaani, installs its dependencies, downloads the gated model (~0.5 GB) and checks that it speaks.
# Usage: bash backend/scripts/setup_dhvaani.sh   (works from any directory; also: ./bonclay setup --dhvaani)
#
# Before you run it:
#   1. The model repository https://huggingface.co/ARTPARK-IISc/DhVaani-0.5 is gated: the Hugging Face account behind
#      your token needs access granted by ARTPARK-IISc. Read the licence notes in docs/decisions/indian-language-cloning.md.
#   2. Put a Hugging Face "Read" token in backend/.env as HF_TOKEN=hf_...
#
# DhVaani is a small model (about 0.1B parameters, ZipVoice-based). It runs on the Apple GPU (MPS), an NVIDIA GPU (CUDA)
# or the CPU, in its own virtualenv (its torch / numpy pins differ from the others), about 1.3-3x slower than real time
# on an M4: for Studio renders, Say it once and pre-rendered clips, not for live calls.
#
# The model's own requirements.txt misses two packages it needs: tensorboard (imported by its utilities) and torchcodec
# (torchaudio 2.11 loads audio through it, and it needs ffmpeg). Both are installed here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$BACKEND_DIR/.venv-dhvaani"
MODEL_ID="${DHVAANI_MODEL:-ARTPARK-IISc/DhVaani-0.5}"
WORKER="$BACKEND_DIR/services/dhvaani_worker.py"

# The token comes from the environment or backend/.env. It is read, never echoed, and never sourced as shell.
if [[ -z "${HF_TOKEN:-}" && -f "$BACKEND_DIR/.env" ]]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' "$BACKEND_DIR/.env" | head -1 | cut -d= -f2- | tr -d "\"'" || true)"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "setup_dhvaani.sh: HF_TOKEN is not set. Put HF_TOKEN=hf_... in backend/.env (see the notes at the top of this script)." >&2
    exit 1
fi
export HF_TOKEN

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "setup_dhvaani.sh: warning: ffmpeg is not installed. DhVaani's audio loader (torchcodec) needs it; the worker falls back" >&2
    echo "to soundfile for WAV files, but install it anyway (for example: brew install ffmpeg)." >&2
fi

# Python 3.11 through uv when it is available (uv fetches the interpreter itself), else the system python3.
USE_UV=0
if command -v uv >/dev/null 2>&1; then
    USE_UV=1
fi
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

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating virtualenv at $VENV_DIR"
    if [[ "$USE_UV" == 1 ]]; then
        uv venv --python 3.11 "$VENV_DIR"
    else
        if ! BASE_PY="$(find_python)"; then
            echo "setup_dhvaani.sh: Python 3.10 or newer is required. Install uv (https://docs.astral.sh/uv/) or Python (for example: brew install python@3.12) and run this again." >&2
            exit 1
        fi
        echo "Using $("$BASE_PY" -c 'import sys; print(sys.version.split()[0])') from $BASE_PY"
        "$BASE_PY" -m venv "$VENV_DIR"
    fi
fi

pip_install() {
    if [[ "$USE_UV" == 1 ]]; then
        uv pip install --python "$VENV_PY" "$@"
    else
        "$VENV_PY" -m pip install "$@"
    fi
}

echo "Installing DhVaani packages (this can take several minutes)"
if [[ "$USE_UV" != 1 ]]; then
    "$VENV_PY" -m pip install --upgrade pip
fi
if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    # NVIDIA: the CUDA build of torch (what the model card recommends), the other packages from PyPI.
    echo "NVIDIA GPU found: installing PyTorch with CUDA"
    pip_install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
    pip_install torchcodec
else
    # The versions below are the ones the model was tried with (Apple M4, MPS, 0 failures in 50 renders).
    pip_install "torch==2.14.1" "torchaudio==2.11.0" "torchcodec==0.17.0"
fi
pip_install "numpy==2.4.6" "soundfile==0.14.0" "safetensors==0.8.0" "einops==0.8.2" "vocos==0.1.0" \
    "tensorboard==2.21.0" "pydub==0.25.1" "scipy" "huggingface_hub==2.2.0" hf_transfer

echo "Downloading $MODEL_ID (about 0.5 GB, plus the 24 kHz vocoder; resumes if interrupted)"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
downloaded=0
for attempt in 1 2 3; do
    set +e
    "$VENV_PY" "$WORKER" --download --model "$MODEL_ID"
    code=$?
    set -e
    if [[ "$code" -eq 0 ]]; then
        downloaded=1
        break
    fi
    if [[ "$code" -eq 2 ]]; then  # access not granted: retrying cannot help
        exit 2
    fi
    if [[ "$attempt" -lt 3 ]]; then
        echo "Download failed (attempt $attempt of 3). Retrying in 5 seconds..." >&2
        sleep 5
    fi
done
if [[ "$downloaded" -ne 1 ]]; then
    echo "setup_dhvaani.sh: could not download $MODEL_ID after 3 attempts. Check your network and run this again." >&2
    exit 1
fi

echo "Checking that it speaks (one Hindi sentence)"
if ! "$VENV_PY" "$WORKER" --selftest --model "$MODEL_ID"; then
    echo "setup_dhvaani.sh: the model downloaded but did not produce speech; see the messages above." >&2
    exit 1
fi

echo
echo "DhVaani is ready: $VENV_PY"
echo "The backend finds this environment automatically; no extra setting is needed."
echo "Use it: clone a voice (any consented sample), then generate speech in an Indian language in Studio, in Say it once,"
echo "or via POST /api/voices/generate with language=mr, ta, bn, te, kn, gu, ml, pa, or, ur, ... (Hindi too)."
echo "Credit: Voice cloning in Indian languages by DhVaani from ARTPARK-IISc. Every clip is also watermarked by BonClay."
echo "Licence: see docs/decisions/indian-language-cloning.md (written approval from ARTPARK-IISc for this use; keep it on file)."
