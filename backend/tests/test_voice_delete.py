"""DELETE /api/voices/{id} removes the voice completely (sample, embedding, consent recording) and unlinks agents."""
import glob
import os

import pytest

from tests.test_consent import CHALLENGE, LOCAL, backend_mod, clone, fake_groq  # noqa: F401  (fixtures reused)


@pytest.fixture()
def cloned(app_client, backend_mod, ref_wav, tmp_path, monkeypatch):
    from services.local_store import LocalStore

    monkeypatch.setattr(backend_mod, "groq_client", fake_groq(CHALLENGE))
    monkeypatch.setattr(backend_mod, "local_db", LocalStore(str(tmp_path)))
    r = clone(app_client, ref_wav)
    assert r.status_code == 200, r.text
    return r.json()


def test_delete_removes_voice_consent_files_and_unlinks_agents(app_client, backend_mod, cloned, tmp_path):
    vid, cid = cloned["id"], cloned["consent_id"]
    assert glob.glob(str(tmp_path / f"consent_{cid}.*")) and (tmp_path / f"{vid}.wav").exists()
    assert (tmp_path / f"{vid}.pt").exists()
    agent = backend_mod.local_db.insert("agents", {"user_id": "local-user", "project_id": "local_project",
                                                   "name": "A", "voice_id": vid, "supported_languages": ["en"]})
    other = backend_mod.local_db.insert("agents", {"user_id": "local-user", "project_id": "local_project",
                                                   "name": "B", "voice_id": None, "supported_languages": ["en"]})

    r = app_client.delete(f"/api/voices/{vid}", headers=LOCAL)
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    assert r.json()["removed"] == {"consents": 1, "files": 3, "agents_unlinked": 1}

    assert not glob.glob(str(tmp_path / f"consent_{cid}.*"))
    assert not (tmp_path / f"{vid}.wav").exists() and not (tmp_path / f"{vid}.pt").exists()
    assert vid not in app_client.get("/api/voices", headers=LOCAL).text
    assert backend_mod.local_db.get("agents", agent["id"])["voice_id"] is None
    assert app_client.get(f"/api/agents/{agent['id']}", headers=LOCAL).json()["voice_id"] is None
    assert backend_mod.local_db.get("agents", other["id"]) is not None
    assert [p for p in os.listdir(tmp_path) if p.startswith("consent_")] == []


def test_deleting_an_unknown_voice_is_404(app_client, backend_mod):
    r = app_client.delete("/api/voices/00000000-0000-0000-0000-000000000000", headers=LOCAL)
    assert r.status_code == 404 and r.json()["code"] == "VOICE_NOT_FOUND"
