#!/usr/bin/env bash
# One-time setup for Chatterbox Multilingual (Resemble AI, MIT licence): Hindi voice cloning, plus 22 other languages.
# Creates backend/.venv-chatterbox, installs its dependencies, and downloads the model (about 3 GB).
# Usage: bash backend/scripts/setup_chatterbox.sh   (works from any directory; also: ./bonclay setup --chatterbox)
#
# Chatterbox pins its own torch / numpy / transformers, so it runs in its own virtualenv, like Indic-Speak and MLX.
# It runs on the CPU (the Apple GPU path is slower and sometimes produces NaNs), roughly 4-6x slower than real time:
# use it for Studio renders and pre-rendered clips, not for live agent calls (those use a faster preset voice).
# Of the Indian languages it clones Hindi only. No Hugging Face token is needed: the weights are public.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$BACKEND_DIR/.venv-chatterbox"
MODEL_ID="${CHATTERBOX_MODEL:-ResembleAI/chatterbox}"

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
    echo "setup_chatterbox.sh: Python 3.10 or newer is required. Install one (for example: brew install python@3.12) and run this again." >&2
    exit 1
fi
echo "Using $("$BASE_PY" -c 'import sys; print(sys.version.split()[0])') from $BASE_PY"

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating virtualenv at $VENV_DIR"
    "$BASE_PY" -m venv "$VENV_DIR"
fi

echo "Installing Chatterbox packages (this can take several minutes)"
"$VENV_PY" -m pip install --upgrade pip
# setuptools<81: newer releases drop pkg_resources, which the Perth watermarker needs. Without it the watermarker
# import yields None and the model fails to load.
"$VENV_PY" -m pip install "chatterbox-tts==0.1.7" "setuptools<81" soundfile numpy hf_transfer

echo "Downloading $MODEL_ID (about 3 GB; resumes if interrupted)"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1
downloaded=0
for attempt in 1 2 3; do
    set +e
    "$VENV_PY" "$BACKEND_DIR/services/chatterbox_worker.py" --download --model "$MODEL_ID"
    code=$?
    set -e
    if [[ "$code" -eq 0 ]]; then
        downloaded=1
        break
    fi
    if [[ "$attempt" -lt 3 ]]; then
        echo "Download failed (attempt $attempt of 3). Retrying in 5 seconds..." >&2
        sleep 5
    fi
done
if [[ "$downloaded" -ne 1 ]]; then
    echo "setup_chatterbox.sh: could not download $MODEL_ID after 3 attempts. Check your network and run this again." >&2
    exit 1
fi

echo
echo "Chatterbox Multilingual is ready: $VENV_PY"
echo "The backend finds this environment automatically; no extra setting is needed."
echo "Use it: clone a voice (English or Hindi sample), then generate Hindi speech in Studio or via POST /api/voices/generate"
echo "with language=hi. Expect about 4-6 seconds of compute per second of speech on a laptop CPU."
echo "Licence: MIT (Resemble AI). Every clip is also watermarked by BonClay (Verify page)."
