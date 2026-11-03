"""Kokoro voices whose language needs optional packages (Japanese, Mandarin) are reported honestly: listed as
unavailable and refused with a clear 503 when the packages are missing, instead of an opaque 500 mid-request."""
import importlib.util
from pathlib import Path

import pytest

kokoro_router = pytest.importorskip("routers.kokoro")
BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture()
def packages(monkeypatch):
    """Control which optional packages 'exist'. `packages.absent` is the set of module names reported missing.

    The Japanese and Mandarin packages start out present, so the result does not depend on what this machine has
    installed (CI installs neither)."""
    real = importlib.util.find_spec
    state = type("State", (), {"absent": set(), "present": JA | ZH | {"unidic_lite", "unidic"}})()

    def fake(name, *a, **k):
        if name in state.absent:
            return None
        return object() if name in state.present else real(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    return state


JA = {"pyopenjtalk", "fugashi", "jaconv", "mojimoji"}
ZH = {"jieba", "pypinyin", "cn2an", "ordered_set"}


def test_english_and_the_espeak_languages_never_need_extra_packages(packages):
    packages.absent = JA | ZH | {"unidic_lite", "unidic"}
    for code in "abefhip":  # English, Spanish, French, Hindi, Italian, Portuguese
        assert kokoro_router.missing_packages(code) == []


def test_japanese_needs_the_tokenizer_and_one_dictionary(packages):
    packages.absent = {"pyopenjtalk"}
    assert kokoro_router.missing_packages("j") == ["pyopenjtalk"]
    packages.present = JA | {"unidic"}
    packages.absent = {"unidic_lite"}                      # unidic-lite missing but the full unidic present: fine
    assert kokoro_router.missing_packages("j") == []
    packages.present = JA
    packages.absent = {"unidic_lite", "unidic"}            # no dictionary at all
    assert kokoro_router.missing_packages("j") == ["unidic_lite/unidic"]


def test_chinese_needs_its_four_small_packages(packages):
    packages.absent = {"ordered_set", "jieba"}
    assert sorted(kokoro_router.missing_packages("z")) == ["jieba", "ordered_set"]
    packages.absent = set()
    assert kokoro_router.missing_packages("z") == []


def test_voices_endpoint_marks_unusable_voices(app_client, packages):
    packages.absent = {"pyopenjtalk", "ordered_set"}
    voices = {v["id"]: v for v in app_client.get("/api/kokoro/voices").json()["voices"]}
    assert voices["jf_alpha"]["available"] is False and "requirements-ja.txt" in voices["jf_alpha"]["unavailable_reason"]
    assert voices["zf_xiaobei"]["available"] is False and "misaki[zh]" in voices["zf_xiaobei"]["unavailable_reason"]
    assert all(voices[v]["available"] for v in ("af_heart", "hf_alpha", "ef_dora", "if_sara"))
    packages.absent = set()
    assert all(v["available"] for v in app_client.get("/api/kokoro/voices").json()["voices"])


def test_generating_with_a_voice_that_cannot_speak_is_a_clear_503_not_a_500(app_client, packages, monkeypatch):
    packages.absent = {"pyopenjtalk"}
    called = []
    monkeypatch.setattr(kokoro_router, "_synthesize", lambda *a: called.append(a) or b"RIFF")
    r = app_client.post("/api/kokoro/generate", headers={"storage-mode": "local"},
                        json={"voice": "jf_alpha", "text": "こんにちは"})
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "ENGINE_UNAVAILABLE" and body["details"]["missing"] == ["pyopenjtalk"]
    assert "requirements-ja.txt" in body["details"]["hint"]
    assert called == []  # refused before any model work
    # an English voice is unaffected, and so is Japanese once the packages are there
    assert app_client.post("/api/kokoro/generate", headers={"storage-mode": "local"},
                           json={"voice": "af_heart", "text": "hello"}).status_code == 200
    packages.absent = set()
    assert app_client.post("/api/kokoro/generate", headers={"storage-mode": "local"},
                           json={"voice": "jf_alpha", "text": "こんにちは"}).status_code == 200


def test_engine_list_stops_claiming_languages_it_cannot_speak(app_client, packages):
    def kokoro_languages():
        engines = app_client.get("/api/models/status").json()["engines"]
        return next(e for e in engines if e["id"] == "kokoro")["languages"]

    packages.absent = {"pyopenjtalk", "ordered_set"}
    assert "ja" not in kokoro_languages() and "zh" not in kokoro_languages() and "hi" in kokoro_languages()
    packages.absent = set()
    assert {"ja", "zh"} <= set(kokoro_languages())


# ---- the install files themselves --------------------------------------------------------------------------

def _requirements(name):
    return {line.split("#")[0].strip().split(">")[0].split("[")[0].split("=")[0].lower()
            for line in (BACKEND / name).read_text().splitlines() if line.split("#")[0].strip()}


def test_requirements_install_kokoro_so_a_readme_only_setup_has_the_engine():
    """Regression: kokoro was missing from requirements.txt, so following the README gave a backend with no Kokoro."""
    assert {"kokoro", "misaki"} <= _requirements("requirements.txt")


def test_the_compiled_japanese_packages_stay_optional():
    """pyopenjtalk compiles from source; in requirements.txt it would break slim Docker images and CI."""
    assert "pyopenjtalk" not in _requirements("requirements.txt")
    assert {"pyopenjtalk", "fugashi", "jaconv", "mojimoji", "unidic-lite"} <= _requirements("requirements-ja.txt")
