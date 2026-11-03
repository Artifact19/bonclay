"""Cloud mode (Supabase): the audio bucket is private, so clients get signed, time-limited URLs, never public ones.
The Supabase client is a fake that records what it was asked."""
import asyncio

import pytest

import backend


class FakeBucket:
    def __init__(self, store):
        self.store = store

    def upload(self, path, data, opts=None):
        self.store.uploads.append(path)

    def get_public_url(self, path):
        raise AssertionError("the bucket is private: public URLs must not be used")

    def create_signed_url(self, path, expires_in):
        self.store.signed.append((path, expires_in))
        if self.store.fail_signing:
            raise RuntimeError("storage down")
        return {"signedURL": f"https://x.supabase.co/storage/v1/object/sign/audio-assets/{path}?token=t"}


class FakeStorage:
    def __init__(self):
        self.uploads, self.signed, self.fail_signing = [], [], False

    def from_(self, bucket):
        assert bucket == backend.AUDIO_BUCKET
        return FakeBucket(self)


class FakeSb:
    def __init__(self):
        self.storage = FakeStorage()


@pytest.fixture()
def cloud(monkeypatch, tmp_path):
    sb = FakeSb()
    monkeypatch.setattr(backend, "supabase", sb)
    monkeypatch.setattr(backend, "SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setattr(backend.provenance, "apply_provenance",
                        lambda **kw: (kw["wav_bytes"], {"receipt_id": "rcpt"}))
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    recorded = []

    def record(user_id, voice_id, voice_name, text, engine, language, duration_s, audio_url, use_supabase, gen_id):
        row = {"id": gen_id, "user_id": user_id, "audio_url": audio_url, "duration_s": duration_s}
        recorded.append(row)
        return row

    monkeypatch.setattr(backend.generations_store, "record_generation", record)
    sb.recorded = recorded
    return sb


def test_signed_url_uses_the_configured_lifetime(cloud, monkeypatch):
    monkeypatch.setattr(backend, "SIGNED_URL_EXPIRES_S", 900)
    url = backend._signed_audio_url(cloud, "u/a.wav")
    assert url.startswith("https://x.supabase.co/storage/v1/object/sign/") and cloud.storage.signed == [("u/a.wav", 900)]


def test_default_lifetime_is_one_hour():
    assert backend.SIGNED_URL_EXPIRES_S == 3600


def test_relative_signed_paths_are_made_absolute(cloud):
    cloud.storage.from_ = lambda b: type("B", (), {"create_signed_url": lambda self, p, e: {"signedURL": "/object/sign/b/p?t=1"}})()
    assert backend._signed_audio_url(cloud, "p") == "https://x.supabase.co/storage/v1/object/sign/b/p?t=1"


def test_signing_failure_gives_none_not_an_error(cloud):
    cloud.storage.fail_signing = True
    assert backend._signed_audio_url(cloud, "u/a.wav") is None


def test_generation_returns_a_signed_url_and_stores_the_path(cloud):
    gen = asyncio.run(backend.persist_generation("u1", "cloud", b"RIFF", 1.0, text="hi", engine="kokoro",
                                                 language="en"))
    path = f"u1/generations/{gen['id']}.wav"
    assert cloud.storage.uploads == [path]
    assert gen["audio_url"] == f"https://x.supabase.co/storage/v1/object/sign/audio-assets/{path}?token=t"
    assert cloud.storage.signed == [(path, backend.SIGNED_URL_EXPIRES_S)]
    # history keeps the path, because a signed link would expire
    assert cloud.recorded[0]["audio_url"] == backend.STORED_PATH_PREFIX + path


def test_stored_paths_are_signed_when_history_is_read(cloud):
    rows = [{"id": "1", "audio_url": "storage:u1/generations/1.wav"},
            {"id": "2", "audio_url": "https://old.example/public.wav"},  # rows from before signed URLs stay as they are
            {"id": "3", "audio_url": None}]
    out = backend._sign_stored_audio(rows)
    assert out[0]["audio_url"].endswith("/u1/generations/1.wav?token=t")
    assert out[1]["audio_url"] == "https://old.example/public.wav" and out[2]["audio_url"] is None
    assert rows[0]["audio_url"] == "storage:u1/generations/1.wav"  # the input is not mutated


def test_turn_audio_url_is_signed(cloud, monkeypatch):
    class Run:
        local, sb, user_id, conversation_id, user_text, language = False, cloud, "u1", "c1", "q", "en"
        start_time, timings, tool_objs = 0.0, {}, []

        def __init__(self):
            self.sb = cloud

    class Table:
        def insert(self, rows):
            return self

        def execute(self):
            return None

    cloud.table = lambda name: Table()
    url, _ = asyncio.run(backend._turn_store(Run(), "t1", b"RIFF", "hello", engine_key="kokoro", llm_provider=None,
                                             llm_model=None, fallback=False))
    assert "/object/sign/audio-assets/u1/" in url and url.endswith("?token=t")
    assert cloud.storage.signed[0][1] == backend.SIGNED_URL_EXPIRES_S


def test_local_mode_urls_are_unchanged(app_client, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "LOCAL_AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(backend.provenance, "apply_provenance",
                        lambda **kw: (kw["wav_bytes"], {"receipt_id": "rcpt"}))
    monkeypatch.setattr(backend.generations_store, "record_generation",
                        lambda *a: {"id": a[-1], "audio_url": a[7]})
    gen = asyncio.run(backend.persist_generation("u1", "local", b"RIFF", 1.0, text="hi", engine="kokoro",
                                                 language="en"))
    assert gen["audio_url"] == f"/api/generations/{gen['id']}/audio"
