#!/usr/bin/env python3
"""Seconds from the consent clip to a usable cloned voice.

Both input clips are synthetic: a Kokoro voice (am_michael) reads a neutral passage (the reference) and the
consent sentence (the consent clip). No real person's voice is used. Each run:
  1. POST /api/voices/clone  (reference + consent audio; the server verifies the consent speech, transcribes the
     reference and builds the voice) -> timed, "clone"
  2. POST /api/voices/generate with the new voice for a short sentence -> timed, "first generation"
Run 1 is "cold" (models may need to load); the later runs are "warm". Every voice this script creates is deleted at
the end, even on failure. The API limits cloning to 5 per minute, so runs are paced (waiting is not timed).

    backend/.venv/bin/python backend/bench/clone_time.py            # 10 runs
    backend/.venv/bin/python backend/bench/clone_time.py --quick    # 2 runs
"""
import argparse
import sys

from common import (LIMITS, Api, ApiError, RateGate, add_common_args, err_text, fmt, kokoro_voices, kokoro_wav,
                    new_result, pick_voice, print_header, require_api, stats, wav_info, write_result)

REFERENCE_PASSAGE = ("The morning light came slowly over the quiet hills, and a gentle wind moved through the tall "
                     "grass beside the river. Birds began to sing.")
FALLBACK_CONSENT = "I consent to creating an AI version of my voice."  # the statement the API requires verbatim
GEN_TEXT = "Thank you for calling Sunrise Clinic. How can I help you today?"
TEXT_KEYS = ["challenge_text", "text", "phrase", "challenge", "consent_text", "statement", "sentence"]
ID_KEYS = ["challenge_id", "id"]


def fetch_challenge(api: Api):
    """(challenge_id, text) from GET /api/consent/challenge when the API has it, else (None, None)."""
    try:
        r, _, _ = api.get("/api/consent/challenge", timeout=15)
    except Exception:  # noqa: BLE001
        return None, None
    if r.status_code != 200:
        return None, None
    try:
        j = r.json()
    except ValueError:
        return None, None
    if isinstance(j.get("challenge"), dict):
        j = j["challenge"]
    text = next((j[k] for k in TEXT_KEYS if isinstance(j.get(k), str) and j[k].strip()), None)
    cid = next((j[k] for k in ID_KEYS if j.get(k)), None)
    return (str(cid) if cid else None), text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("-n", "--runs", type=int, default=None, help="number of clone runs (default 10; quick 2)")
    a = ap.parse_args()
    n = a.runs if a.runs is not None else (2 if a.quick else 10)

    api = Api(a.api)
    health = require_api(api)
    settings = {"quick": a.quick, "runs": n, "reference_voice": "Kokoro am_michael (synthetic, no real person)",
                "reference_text": REFERENCE_PASSAGE, "generation_text": GEN_TEXT, "language": "en",
                "models_loaded_at_start": health.get("models_loaded")}
    res = new_result("clone_time", api, health, settings)
    print_header("Clone time (consent clip to usable voice)", res)

    avail = kokoro_voices(api)
    voice = pick_voice(avail, ["am_michael"]) or sys.exit("Kokoro voice am_michael is not available")
    ref_wav, _ = kokoro_wav(api, voice, REFERENCE_PASSAGE)
    ref_info = wav_info(ref_wav)
    settings["reference_seconds"] = round(ref_info["seconds"], 2)
    print(f"reference clip: {ref_info['seconds']:.1f} s (synthetic Kokoro {voice})")
    cid0, ctext0 = fetch_challenge(api)
    settings["consent_source"] = "GET /api/consent/challenge" if ctext0 else "fixed sentence (no challenge endpoint)"
    print(f"consent: {settings['consent_source']}")
    fixed_consent_wav = None if ctext0 else kokoro_wav(api, voice, FALLBACK_CONSENT)[0]

    gates = {k: RateGate(LIMITS[k]) for k in ("voices_clone", "voices_generate")}
    created = []
    try:
        for i in range(n):
            item = {"run": i + 1, "cold": i == 0}
            try:
                if ctext0:  # a challenge may be single use: take a fresh one for every run (rendering is not timed)
                    cid, ctext = (cid0, ctext0) if i == 0 else fetch_challenge(api)
                    consent_wav = kokoro_wav(api, voice, ctext)[0]
                    consent_fields = {"consent_text": ctext, **({"challenge_id": cid} if cid else {})}
                else:
                    consent_wav, ctext = fixed_consent_wav, FALLBACK_CONSENT
                    consent_fields = {"consent_text": ctext}
                item["consent_text"] = ctext
                gates["voices_clone"].wait()
                r, clone_s, _ = api.post(
                    "/api/voices/clone",
                    files={"reference_audio": ("reference.wav", ref_wav, "audio/wav"),
                           "consent_audio": ("consent.wav", consent_wav, "audio/wav")},
                    data={"display_name": f"bench-clone-{i + 1}", "language": "en", "consent_confirmed": "true",
                          **consent_fields}, timeout=900)
                item["clone_status"] = r.status_code
                if r.status_code != 200:
                    raise ApiError(r.status_code, "clone: " + err_text(r))
                j = r.json()
                created.append(j["id"])
                item.update({"clone_s": round(clone_s, 3), "voice_id": j["id"], "engine": j.get("engine"),
                             "mode": j.get("mode"), "consent": j.get("consent")})
                gates["voices_generate"].wait()
                r, gen_s, _ = api.post("/api/voices/generate",
                                       data={"text": GEN_TEXT, "voice_id": j["id"], "language": "en"}, timeout=900)
                item["generate_status"] = r.status_code
                if r.status_code != 200:
                    raise ApiError(r.status_code, "generate: " + err_text(r))
                info = wav_info(r.content)
                item.update({"first_generation_s": round(gen_s, 3), "audio_s": round(info["seconds"], 3)})
            except Exception as e:  # noqa: BLE001
                item["failure"] = f"{type(e).__name__}: {e}"
            res["items"].append(item)
            print(f"  run {i + 1}/{n}{' (cold)' if i == 0 else ''}: clone {fmt(item.get('clone_s'), 1)} s, first "
                  f"generation {fmt(item.get('first_generation_s'), 1)} s {item.get('failure', '')}", flush=True)
    finally:
        left = []
        for vid in created:
            try:
                r, _, _ = api.delete(f"/api/voices/{vid}", timeout=60)
                if r.status_code not in (200, 204):
                    left.append(vid)
                    print(f"WARNING: DELETE voice {vid} -> HTTP {r.status_code} {err_text(r)}")
            except Exception as e:  # noqa: BLE001
                left.append(vid)
                print(f"WARNING: could not delete voice {vid}: {e}")
        print(f"deleted {len(created) - len(left)}/{len(created)} created voices")
        res["cleanup"] = {"created": len(created), "not_deleted": left}
        summarize(res)
        print_summary(res)
        print(f"\nraw results: {write_result(res)}")


def summarize(res: dict) -> None:
    rows = res["items"]
    first = rows[0] if rows else {}
    warm = [r for r in rows[1:] if "failure" not in r]
    res["summary"] = {
        "runs": len(rows), "failures": sum(1 for r in rows if "failure" in r),
        "first_run": {"clone_s": first.get("clone_s"), "first_generation_s": first.get("first_generation_s")},
        "warm": {"n": len(warm), "clone_s": stats([r["clone_s"] for r in warm]),
                 "first_generation_s": stats([r["first_generation_s"] for r in warm])},
        "note": "clone = POST /api/voices/clone (consent check, reference transcription, voice build); inputs are "
                "synthetic Kokoro audio, not a real person",
    }


def print_summary(res: dict) -> None:
    s = res["summary"]
    w = s["warm"]
    print(f"\nruns {s['runs']} (failures {s['failures']})")
    print(f"  first (cold) run : clone {fmt(s['first_run']['clone_s'], 1)} s, first generation "
          f"{fmt(s['first_run']['first_generation_s'], 1)} s")
    print(f"  warm p50 (n={w['n']})  : clone {fmt(w['clone_s']['p50'], 1)} s, first generation "
          f"{fmt(w['first_generation_s']['p50'], 1)} s")


if __name__ == "__main__":
    main()
