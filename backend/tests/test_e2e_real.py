"""Real clone + generate round trip. Needs CUDA and downloaded models; skipped otherwise."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import models_present  # noqa: E402

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not models_present(), reason="Qwen3-TTS model not in local HF cache"),
]

HEADERS = {"storage-mode": "local", "Authorization": "Bearer test"}


def _clone(client, wav, transcript):
    data = {"display_name": "e2e", "language": "en", "consent_confirmed": "true"}
    if transcript:
        data["reference_text"] = transcript
    with open(wav, "rb") as f:
        return client.post("/api/voices/clone", headers=HEADERS, data=data,
                           files={"reference_audio": ("ref.wav", f, "audio/wav")})


@pytest.mark.parametrize("transcript", [None, "This is a short reference recording."],
                         ids=["xvector_only", "with_transcript"])
def test_clone_then_generate(app_client, ref_wav, transcript):
    r = _clone(app_client, ref_wav, transcript)
    assert r.status_code == 200, r.text
    body = r.json()
    voice_id = body.get("voice_id") or body.get("id") or (body.get("voice") or {}).get("id")
    assert voice_id
    g = app_client.post("/api/voices/generate", headers=HEADERS,
                        data={"text": "Hello from BonClay.", "voice_id": voice_id, "mode": "premium"})
    assert g.status_code == 200, g.text
    assert len(g.content) > 1000
