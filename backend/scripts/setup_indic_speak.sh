#!/usr/bin/env bash
# One-time setup for Indic-Speak: preset voices in 22 Indian languages (Bodhan AI / AI4Bharat).
# Creates backend/.venv-indic, installs its dependencies, and downloads the model (about 7 GB).
# Usage: bash backend/scripts/setup_indic_speak.sh   (works from any directory)
#
# Before you run it:
#   1. Open https://huggingface.co/bodhan-ai/indic-speak while signed in and request access (approved automatically).
#      The form asks for your details and agreement to the Indic Open Model License; read it first.
#   2. Create a Hugging Face "Read" token and put it in backend/.env as HF_TOKEN=hf_...
# Needs about 8 GB of free memory to run (the model alone is ~7.5 GB in bf16).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$BACKEND_DIR/.venv-indic"
MODEL_ID="${INDIC_SPEAK_MODEL:-bodhan-ai/indic-speak}"

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
    echo "setup_indic_speak.sh: Python 3.10 or newer is required. Install one (for example: brew install python@3.12) and run this again." >&2
    exit 1
fi
echo "Using $("$BASE_PY" -c 'import sys; print(sys.version.split()[0])') from $BASE_PY"

# The token comes from the environment or backend/.env. It is read, never echoed, and never sourced as shell.
if [[ -z "${HF_TOKEN:-}" && -f "$BACKEND_DIR/.env" ]]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' "$BACKEND_DIR/.env" | head -1 | cut -d= -f2- | tr -d "\"'" || true)"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "setup_indic_speak.sh: HF_TOKEN is not set. Put HF_TOKEN=hf_... in backend/.env (see the notes at the top of this script)." >&2
    exit 1
fi
export HF_TOKEN

VENV_PY="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
    echo "Creating virtualenv at $VENV_DIR"
    "$BASE_PY" -m venv "$VENV_DIR"
fi

echo "Installing Indic-Speak packages (this can take a few minutes)"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install "transformers>=5" torch snac soundfile numpy huggingface_hub

echo "Downloading $MODEL_ID (about 7 GB; resumes if interrupted)"
downloaded=0
for attempt in 1 2 3; do
    set +e
    "$VENV_PY" "$BACKEND_DIR/services/indic_speak_worker.py" --download --model "$MODEL_ID"
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
    echo "setup_indic_speak.sh: could not download $MODEL_ID after 3 attempts. Check your network and run this again." >&2
    exit 1
fi

echo
echo "Indic-Speak is ready: $VENV_PY"
echo "The backend finds this environment automatically; no extra setting is needed."
echo "Licence: Indic Open Model License v1.0. Show \"Built with Indic-Speak from Bodhan AI / AI4Bharat.\" wherever its output reaches"
echo "users (the web app does this for you). Offering it as an API to other organisations needs Bodhan AI's approval."
