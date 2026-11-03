#!/usr/bin/env python3
"""Spoken-turn latency: a spoken question in, the agent's full spoken reply (audio ready) out.

Each question is pre-rendered to WAV with Kokoro (not timed), then posted to /api/conversations/{agent}/turn.
The clock runs from request start to the JSON response being received. This is NOT time-to-first-audio: there is
no streaming, the reply audio is complete when the response arrives.

With --stream the turn goes to /api/conversations/{agent}/turn/stream instead (server-sent events) and the clock
also records when the first `sentence` event arrives (time to first audio: the caller can start hearing the reply),
when the first sentence of the actual answer arrives (the AI disclosure, when the agent says it, is its own first
sentence) and when `done` arrives (the whole reply ready). Results go to latency_stream-*.json.

    backend/.venv/bin/python backend/bench/latency.py            # 30 turns
    backend/.venv/bin/python backend/bench/latency.py --quick    # 3 turns, smoke test
    backend/.venv/bin/python backend/bench/latency.py --stream   # 30 streamed turns
"""
import argparse
import json
import sys
import time

from common import (Api, ApiError, add_common_args, delete_agent, ensure_agent, err_text, fmt, kokoro_voices,
                    kokoro_wav, new_result, pick_voice, print_header, require_api, spoken_turn, stats, wav_info,
                    write_result)

# Answerable from backend/templates/agents/clinic_receptionist.md (Sunrise Family Clinic).
QUESTIONS_EN = [
    "What time does the clinic open on Monday?",
    "Is the clinic open on Sunday?",
    "Which days does Dr. Meera Kulkarni see patients?",
    "How much is a dentist consultation?",
    "What is the paediatrician's fee for a first visit?",
    "When is the dermatologist available?",
    "What is the clinic's phone number?",
    "How do I book an appointment?",
    "Can I pay by UPI?",
    "What time is the lunch break?",
    "Is there a pharmacy inside the clinic?",
    "When are blood test reports ready?",
    "How early must I cancel an appointment?",
    "Where is the clinic located?",
    "What should I do in a chest pain emergency?",
]
QUESTIONS_HI = [
    "क्लिनिक सोमवार को कितने बजे खुलता है?",
    "क्या क्लिनिक रविवार को खुला रहता है?",
    "डॉक्टर मीरा कुलकर्णी किन दिनों में बैठती हैं?",
    "दांतों के डॉक्टर की फीस कितनी है?",
    "बच्चों के डॉक्टर की पहली विज़िट की फीस क्या है?",
    "स्किन के डॉक्टर कब मिलते हैं?",
    "क्लिनिक का फोन नंबर क्या है?",
    "अपॉइंटमेंट कैसे बुक करूं?",
    "क्या मैं यूपीआई से पेमेंट कर सकता हूं?",
    "लंच ब्रेक कितने बजे होता है?",
    "क्या क्लिनिक के अंदर फार्मेसी है?",
    "खून की जांच की रिपोर्ट कब मिलती है?",
    "अपॉइंटमेंट कैंसिल करने के लिए कितना पहले बताना होगा?",
    "क्लिनिक का पता क्या है?",
    "सीने में दर्द हो तो क्या करूं?",
]
VOICES_EN = ["af_heart", "am_michael"]
VOICES_HI = ["hf_alpha", "hm_omega"]  # hm_omega is used when the server offers it, else hf_alpha for both


def build_questions(n: int, avail: list):
    """n questions alternating English/Hindi, each with a voice (the two voices of a language alternate)."""
    en_voices = [v for v in VOICES_EN if v in avail] or sys.exit("No English Kokoro voice is available")
    hi_voices = [v for v in VOICES_HI if v in avail] or sys.exit("No Hindi Kokoro voice (hf_alpha) is available")
    out = []
    for i in range(n):
        lang = "en" if i % 2 == 0 else "hi"
        k = i // 2
        if lang == "en":
            out.append({"language": lang, "text": QUESTIONS_EN[k % len(QUESTIONS_EN)], "voice": en_voices[k % len(en_voices)]})
        else:
            out.append({"language": lang, "text": QUESTIONS_HI[k % len(QUESTIONS_HI)], "voice": hi_voices[k % len(hi_voices)]})
    return out


def streamed_turn(api: Api, agent_id: str, wav: bytes, conversation_id=None, timeout: float = 240):
    """POST a spoken turn to the streaming endpoint. Returns (HTTP status, events, error text).

    events is a list of (client seconds since the request started, event name, data); the clock runs from just before
    the request is sent, so it includes the upload, like the blocking turn's clock."""
    data = {"conversation_id": conversation_id} if conversation_id else None
    t0 = time.perf_counter()
    r = api.s.post(f"{api.base}/api/conversations/{agent_id}/turn/stream", data=data,
                   files={"audio": ("question.wav", wav, "audio/wav")}, stream=True, timeout=timeout)
    if r.status_code != 200:
        return r.status_code, [], err_text(r)
    events, name = [], None
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and name:
            events.append((time.perf_counter() - t0, name, json.loads(line[5:].strip())))
            if name in ("done", "error"):
                break
    r.close()
    return r.status_code, events, None


def stream_item(item: dict, status: int, events: list, error) -> None:
    """Fill a result item from the events of one streamed turn."""
    item["status"] = status
    if error:
        item["error"] = error
        return
    sentences = [(t, d) for t, n, d in events if n == "sentence"]
    done = next(((t, d) for t, n, d in events if n == "done"), None)
    err = next((d for _, n, d in events if n == "error"), None)
    transcript = next(((t, d) for t, n, d in events if n == "transcript"), None)
    if transcript:
        item["client_transcript_s"] = round(transcript[0], 3)
        item["heard"] = transcript[1].get("user_text")
    if sentences:
        item["client_first_sentence_s"] = round(sentences[0][0], 3)
    if err or not done:
        item["status"] = None
        item["error"] = f"{(err or {}).get('code')}: {(err or {}).get('message')}" if err else "stream ended without done"
        return
    t_done, d = done
    timings = d.get("timings") or {}
    # When the agent says it is an AI, that is sentence 0 and the answer starts at sentence 1.
    # A "one moment" filler (`filler: true`) can lead the reply instead, when the answer is slow.
    fillers = sum(1 for _, sd in sentences if sd.get("filler"))
    lead = "first_answer_audio_ms" in timings
    disclosed = lead and not fillers
    answer = sentences[1 if lead and len(sentences) > 1 else 0] if sentences else None
    item.update({"client_s": round(t_done, 3), "client_done_s": round(t_done, 3),
                 "client_first_answer_s": round(answer[0], 3) if answer else None, "disclosed_ai": disclosed,
                 "filler_used": bool(fillers),
                 "sentences": len(sentences), "server_total_ms": d.get("total_ms"), "timings": timings,
                 "server_first_audio_ms": timings.get("first_audio_ms"),
                 "server_first_answer_audio_ms": timings.get("first_answer_audio_ms", timings.get("first_audio_ms")),
                 "llm_first_token_ms": timings.get("llm_first_token_ms"), "engine": d.get("engine"),
                 "llm_provider": d.get("llm_provider"), "llm_model": d.get("llm_model"),
                 "language_reported": d.get("language"), "fallback": d.get("fallback"),
                 "interrupted": d.get("interrupted"), "reply": d.get("agent_text")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("-n", "--turns", type=int, default=None, help="number of measured turns (default 30; quick 3)")
    ap.add_argument("--agent-id", help="use this agent instead of creating one from the clinic_receptionist template")
    ap.add_argument("--keep-agent", action="store_true", help="do not delete the agent this script created")
    ap.add_argument("--same-conversation", action="store_true",
                    help="reuse one conversation (history grows) instead of a new conversation per turn")
    ap.add_argument("--stream", action="store_true",
                    help="use the streaming endpoint and record time to the first sentence (first audio) and to done")
    ap.add_argument("--no-disclosure", action="store_true",
                    help="turn the AI disclosure off on the agent this script creates, so the first sentence is the "
                         "answer itself")
    ap.add_argument("--warmup", type=int, default=None, help="warm-up turns, excluded from results (default 2; quick 1)")
    a = ap.parse_args()
    n = a.turns if a.turns is not None else (3 if a.quick else 30)
    warmup = a.warmup if a.warmup is not None else (1 if a.quick else 2)

    api = Api(a.api)
    health = require_api(api)
    settings = {"quick": a.quick, "turns": n, "warmup": warmup, "same_conversation": a.same_conversation,
                "agent_template": "clinic_receptionist", "agent_id_given": a.agent_id,
                "stream": a.stream, "no_disclosure": a.no_disclosure,
                "method": ("client clock from request start: first `sentence` event (first audio), first sentence of "
                           "the answer itself, and `done` (whole reply ready); streaming endpoint"
                           if a.stream else
                           "client wall time: request start to full JSON received; speech in to reply audio ready, "
                           "no streaming")}
    res = new_result("latency_stream" if a.stream else "latency", api, health, settings)
    print_header("Spoken-turn latency" + (" (streaming)" if a.stream else ""), res)

    avail = kokoro_voices(api)
    voices_used = {"en": [v for v in VOICES_EN if v in avail], "hi": [v for v in VOICES_HI if v in avail]}
    settings["voices_used"] = voices_used
    agent_id, created = ensure_agent(api, a.agent_id)
    if a.no_disclosure:
        if not created:
            sys.exit("--no-disclosure only applies to the agent this script creates (do not pass --agent-id)")
        r, _, _ = api.request("PUT", f"/api/agents/{agent_id}", json={"disclose_ai": False}, timeout=60)
        if r.status_code != 200:
            sys.exit(f"Could not turn the AI disclosure off: HTTP {r.status_code} {err_text(r)}")
    res["agent_id"] = agent_id
    try:
        questions = build_questions(warmup + n, avail)
        print(f"rendering {len(questions)} questions with Kokoro (not timed)...")
        for q in questions:
            try:
                wav, _ = kokoro_wav(api, q["voice"], q["text"])
            except ApiError as e:
                sys.exit(f"Could not render the question {q['text']!r} with {q['voice']}: {e}")
            q["wav"] = wav
            q["question_seconds"] = round(wav_info(wav)["seconds"], 2)

        conv = None
        for i, q in enumerate(questions):
            is_warm = i < warmup
            tag = "warm-up" if is_warm else f"{i - warmup + 1}/{n}"
            item = {"index": i, "warmup": is_warm, "language_asked": q["language"], "question": q["text"],
                    "voice": q["voice"], "question_seconds": q["question_seconds"]}
            try:
                if a.stream:
                    status, events, error = streamed_turn(api, agent_id, q["wav"],
                                                          conv if a.same_conversation else None)
                    stream_item(item, status, events, error)
                    done = next((d for _, n, d in events if n == "done"), None)
                    conv = (done or {}).get("conversation_id") or conv
                    res["items"].append(item)
                    print(f"  [{tag}] {q['language']} first audio {item.get('client_first_sentence_s', 0):.2f}s  "
                          f"answer {item.get('client_first_answer_s') or 0:.2f}s  done {item.get('client_s', 0):.2f}s  "
                          f"{item.get('engine')}  {item.get('error', '')}", flush=True)
                    continue
                r, secs = spoken_turn(api, agent_id, q["wav"], conv if a.same_conversation else None)
                item["client_s"] = round(secs, 3)
                item["status"] = r.status_code
                if r.status_code == 200:
                    j = r.json()
                    conv = j.get("conversation_id") or conv
                    item.update({"server_total_ms": j.get("total_ms"), "timings": j.get("timings"),
                                 "engine": j.get("engine"), "llm_provider": j.get("llm_provider"),
                                 "llm_model": j.get("llm_model"), "language_reported": j.get("language"),
                                 "fallback": j.get("fallback"), "heard": j.get("user_text"),
                                 "reply": j.get("agent_text")})
                else:
                    item["error"] = err_text(r)
            except Exception as e:  # noqa: BLE001
                item["status"] = None
                item["error"] = f"{type(e).__name__}: {e}"
            if a.stream:  # a failed streamed turn lands here
                res["items"].append(item)
                print(f"  [{tag}] {q['language']} FAILED {item['error']}", flush=True)
                continue
            res["items"].append(item)
            print(f"  [{tag}] {q['language']} {item.get('client_s', 0):.2f}s  server {item.get('server_total_ms')} ms  "
                  f"{item.get('engine')}  {item.get('error', '')}", flush=True)
    finally:
        if created and not a.keep_agent:
            delete_agent(api, agent_id)
        elif created:
            print(f"kept agent {agent_id}")
        (summarize_stream if a.stream else summarize)(res)
        (print_stream_summary if a.stream else print_summary)(res)
        print(f"\nraw results: {write_result(res)}")


def summarize(res: dict) -> None:
    items = [i for i in res["items"] if not i["warmup"]]
    ok = [i for i in items if i.get("status") == 200]

    def block(rows):
        client = [r["client_s"] * 1000 for r in rows]
        server = [r["server_total_ms"] for r in rows if r.get("server_total_ms") is not None]
        stages = {}
        for k in ("asr_ms", "retrieval_ms", "llm_ms", "tts_ms"):
            stages[k] = stats([r["timings"].get(k) for r in rows if r.get("timings")])["p50"]
        within = (sum(1 for c in client if c <= 3000) / len(client) * 100) if client else None
        within_s = (sum(1 for c in server if c <= 3000) / len(server) * 100) if server else None
        return {"n": len(rows), "client_ms": stats(client), "server_ms": stats(server), "stage_p50_ms": stages,
                "pct_client_within_3s": within, "pct_server_within_3s": within_s,
                "fallback_replies": sum(1 for r in rows if r.get("fallback"))}

    res["summary"] = {"measured_turns": len(items), "failed_turns": len(items) - len(ok), "overall": block(ok),
                      "by_language": {l: block([r for r in ok if r["language_asked"] == l]) for l in ("en", "hi")},
                      "note": "speech in to full reply audio ready; no streaming, so this is not time-to-first-audio"}


def summarize_stream(res: dict) -> None:
    items = [i for i in res["items"] if not i["warmup"]]
    ok = [i for i in items if i.get("status") == 200]

    def ms(rows, key):
        return [r[key] * 1000 for r in rows if r.get(key) is not None]

    def within(vals):
        return (sum(1 for v in vals if v <= 3000) / len(vals) * 100) if vals else None

    def block(rows):
        first, answer, done = ms(rows, "client_first_sentence_s"), ms(rows, "client_first_answer_s"), ms(rows, "client_done_s")
        srv_first = [r["server_first_audio_ms"] for r in rows if r.get("server_first_audio_ms") is not None]
        srv_answer = [r["server_first_answer_audio_ms"] for r in rows if r.get("server_first_answer_audio_ms") is not None]
        srv_total = [r["server_total_ms"] for r in rows if r.get("server_total_ms") is not None]
        stages = {k: stats([r["timings"].get(k) for r in rows if r.get("timings")])["p50"]
                  for k in ("asr_ms", "retrieval_ms", "llm_first_token_ms", "llm_ms", "tts_ms")}
        return {"n": len(rows), "client_first_audio_ms": stats(first), "client_first_answer_ms": stats(answer),
                "client_full_reply_ms": stats(done), "server_first_audio_ms": stats(srv_first),
                "server_first_answer_ms": stats(srv_answer), "server_total_ms": stats(srv_total),
                "stage_p50_ms": stages, "pct_first_audio_within_3s": within(first),
                "pct_first_answer_within_3s": within(answer), "pct_full_reply_within_3s": within(done),
                "disclosed_ai": sum(1 for r in rows if r.get("disclosed_ai")),
                "filler_turns": sum(1 for r in rows if r.get("filler_used")),
                "fallback_replies": sum(1 for r in rows if r.get("fallback")),
                "interrupted_replies": sum(1 for r in rows if r.get("interrupted"))}

    res["summary"] = {"measured_turns": len(items), "failed_turns": len(items) - len(ok), "overall": block(ok),
                      "by_language": {l: block([r for r in ok if r["language_asked"] == l]) for l in ("en", "hi")},
                      "note": "first audio = the first sentence is ready to play (it can be the spoken AI disclosure); "
                              "first answer = the first sentence of the answer itself; full reply = `done` received"}


def print_stream_summary(res: dict) -> None:
    s = res["summary"]
    print(f"\nstreamed turns measured: {s['measured_turns']} (failed: {s['failed_turns']})")
    for name, b in [("overall", s["overall"])] + list(s["by_language"].items()):
        if not b["n"]:
            continue
        f, a_, d = b["client_first_audio_ms"], b["client_first_answer_ms"], b["client_full_reply_ms"]
        sf = b["server_first_audio_ms"]
        print(f"  {name:8s} n={b['n']:<3d} first audio p50 {fmt(f['p50'])} ms p95 {fmt(f['p95'])} ms "
              f"({fmt(b['pct_first_audio_within_3s'])}% within 3 s) | first answer p50 {fmt(a_['p50'])} ms "
              f"p95 {fmt(a_['p95'])} ms | full reply p50 {fmt(d['p50'])} ms p95 {fmt(d['p95'])} ms | "
              f"server first audio p50 {fmt(sf['p50'])} ms p95 {fmt(sf['p95'])} ms")
        print("           stage p50 (ms): " + ", ".join(f"{k[:-3]} {fmt(x)}" for k, x in b["stage_p50_ms"].items())
              + f"  | AI disclosure spoken in {b['disclosed_ai']}/{b['n']} turns, filler {b['filler_turns']}/{b['n']}, fallback {b['fallback_replies']}, "
              f"interrupted {b['interrupted_replies']}")
    print("  note: " + s["note"])


def print_summary(res: dict) -> None:
    s = res["summary"]
    print(f"\nturns measured: {s['measured_turns']} (failed: {s['failed_turns']})")
    for name, b in [("overall", s["overall"])] + list(s["by_language"].items()):
        if not b["n"]:
            continue
        c, v = b["client_ms"], b["server_ms"]
        print(f"  {name:8s} n={b['n']:<3d} client p50 {fmt(c['p50'])} ms  p95 {fmt(c['p95'])} ms | "
              f"server p50 {fmt(v['p50'])} ms  p95 {fmt(v['p95'])} ms | within 3 s: {fmt(b['pct_client_within_3s'])}%"
              f" (client), {fmt(b['pct_server_within_3s'])}% (server)")
        print("           stage p50 (ms): " + ", ".join(f"{k[:-3]} {fmt(x)}" for k, x in b["stage_p50_ms"].items()))
    print("  note: " + s["note"])


if __name__ == "__main__":
    main()
