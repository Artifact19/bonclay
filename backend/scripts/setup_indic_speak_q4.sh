#!/usr/bin/env bash
# One-time setup for the 4-bit Indic-Speak (Apple Silicon): about 2 GB of memory instead of 7 GB, about 3.7x faster.
# Needs the normal Indic-Speak install first (bash backend/scripts/setup_indic_speak.sh) so the weights are cached.
# Creates backend/.venv-indic-q4 (torch, snac, transformers 5, mlx-lm), converts the language model to 4-bit
# (default folder ~/.cache/bonclay/indic-speak-q4) and prints the line to add to backend/.env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$BACKEND_DIR/.venv-indic-q4"
OUT="${INDIC_SPEAK_Q4_DIR:-$HOME/.cache/bonclay/indic-speak-q4}"
MODEL_ID="${INDIC_SPEAK_MODEL:-bodhan-ai/indic-speak}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "setup_indic_speak_q4.sh: this needs an Apple Silicon Mac (MLX)." >&2; exit 1
fi
BASE_PY="$(command -v python3.12 || command -v python3.11 || command -v python3.10 || true)"
[[ -n "$BASE_PY" ]] || { echo "Python 3.10+ is required (brew install python@3.12)." >&2; exit 1; }

[[ -x "$VENV_DIR/bin/python" ]] || "$BASE_PY" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "transformers>=5" torch snac soundfile numpy huggingface_hub mlx-lm

SNAP="$("$VENV_DIR/bin/python" - <<PY
import os
from huggingface_hub import try_to_load_from_cache
p = try_to_load_from_cache("$MODEL_ID", "config.json")
if not isinstance(p, str):
    raise SystemExit("Indic-Speak weights are not cached. Run: bash backend/scripts/setup_indic_speak.sh")
print(os.path.dirname(p))
PY
)"
echo "Converting $SNAP to 4-bit at $OUT"
mkdir -p "$(dirname "$OUT")"
"$VENV_DIR/bin/python" "$BACKEND_DIR/services/indic_speak_quant.py" convert --snapshot "$SNAP" --out "$OUT" --bits 4
echo
echo "Done. Add this line to backend/.env to use it, then restart the API:"
echo "INDIC_SPEAK_MLX_MODEL=$OUT"
