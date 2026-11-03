#!/usr/bin/env python3
"""Phone-call latency, measured on the real phone WebSocket of a running backend, without any phone call.

The script plays Exotel: it mints a stream token for an agent exactly like the router does (routers/phone_exotel.py
`make_token`, signed with the same secret: WIDGET_TOKEN_SECRET from the environment or backend/.env, else the
backend's backend/data/widget_secret), opens ws://.../api/phone/exotel/stream/<token>, sends `connected` and `start`
(8 kHz PCM), listens to the greeting, then streams a spoken question as `media` events in real time (100 ms chunks,
each sent once its 100 ms have "been spoken", then silence, like a phone line) and waits for the reply.

Per run it measures
  * start -> first greeting byte: from sending `start` to the first `media` message of the greeting;
  * end of speech -> first reply byte: from the moment the question's last word ended (in real time) to the first
    `media` message of the reply. This includes the end-of-utterance wait (PHONE_END_SILENCE_MS, default 500 ms):
    it is what a caller hears as the pause.
and prints the medians over --runs runs. Results go to bench/results/phone_latency-*.json.

The question is spoken by Kokoro's preset voice af_heart through the running backend (/api/kokoro/generate) and kept
in bench/data/phone_question.wav; --wav uses another file (never a cloned or a real person's voice). No phone call is
placed and Exotel is never contacted. Each run is a real call for the backend: it creates a phone conversation for
the agent (visible in its conversations), and uses speech recognition and the language model like a call does.
Secrets and tokens are never printed.

    backend/.venv/bin/python backend/bench/phone_latency.py --agent <agent id>             # 5 runs
    backend/.venv/bin/python backend/bench/phone_latency.py --agent <agent id> --runs 10 --question "Is the clinic open on Sunday?"
"""
import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

BENCH_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BENCH_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from common import Api, add_common_args, kokoro_wav, new_result, print_header, require_api, write_result  # noqa: E402

RATE = 8000
CHUNK_S = 0.1
STREAM_SID = "MZbench0000000000000000000000001"
QUESTION_WAV = BENCH_DIR / "data" / "phone_question.wav"


# ------------------------------------------------------------------------------------------------ the token

def _secret() -> bytes:
    """The stream-token secret, read the way routers/widget.py `_secret` reads it (never printed)."""
    env = os.getenv("WIDGET_TOKEN_SECRET", "").strip()
    if not env:
        try:
            from dotenv import dotenv_values  # noqa: PLC0415
            env = str(dotenv_values(BACKEND_DIR / ".env").get("WIDGET_TOKEN_SECRET") or "").strip()
        except ImportError:
            env = ""
    if env:
        return env.encode("utf-8")
    path = BACKEND_DIR / "data" / "widget_secret"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if not value:
        sys.exit("No stream-token secret found (WIDGET_TOKEN_SECRET or backend/data/widget_secret). Open the Call "
                 "dialog or the website widget settings once so the backend creates it, then run this again.")
    return value.encode("utf-8")


def mint_token(agent_id: str, user_id: str, mode: str, ttl: int = 600) -> str:
    """A stream token from the router's own make_token, with its secret lookup pointed at _secret() above."""
    from routers import phone_exotel as px  # noqa: PLC0415
    secret = _secret()
    px._widget = lambda: type("W", (), {"_secret": staticmethod(lambda: secret)})
    return px.make_token(agent_id, user_id, mode, ttl)


def agent_owner(agent_id: str) -> Optional[str]:
    """The owner of a local agent, from the backend's local store (backend/data/local_db/agents.json)."""
    from services.local_store import LocalStore  # noqa: PLC0415
    row = LocalStore(str(BACKEND_DIR / "data")).get("agents", agent_id)
    return row.get("user_id") if row else None


# ------------------------------------------------------------------------------------------------ the audio

def question_pcm(api: Api, wav_path: Optional[str], question: str) -> Dict[str, Any]:
    """The question as 8 kHz PCM16 and where its speech ends (seconds)."""
    if wav_path:
        data = Path(wav_path).read_bytes()
    elif QUESTION_WAV.exists() and QUESTION_WAV.with_suffix(".txt").exists() \
            and QUESTION_WAV.with_suffix(".txt").read_text(encoding="utf-8") == question:
        data = QUESTION_WAV.read_bytes()
    else:
        data, _secs = kokoro_wav(api, "af_heart", question)   # a preset voice, never a cloned one
        QUESTION_WAV.parent.mkdir(parents=True, exist_ok=True)
        QUESTION_WAV.write_bytes(data)
        QUESTION_WAV.with_suffix(".txt").write_text(question, encoding="utf-8")
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    from services import phone_audio as pa  # noqa: PLC0415
    x8 = pa.resample(x, int(sr), RATE)
    frame = int(RATE * 0.02)
    loud = [i for i in range(0, len(x8) - frame, frame) if float(np.sqrt(np.mean(x8[i:i + frame] ** 2))) > 0.01]
    if not loud:
        sys.exit("The question audio is silent.")
    return {"pcm": pa.float_to_pcm16(x8), "speech_end_s": (loud[-1] + frame) / RATE, "seconds": len(x8) / RATE}


# ------------------------------------------------------------------------------------------------ one call

def _media(pcm: bytes, seq: int) -> str:
    return json.dumps({"event": "media", "sequence_number": str(seq), "stream_sid": STREAM_SID,
                       "media": {"chunk": str(seq), "timestamp": str(seq * 100),
                                 "payload": base64.b64encode(pcm).decode("ascii")}})


async def one_call(url: str, question: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import websockets  # noqa: PLC0415
    out: Dict[str, Any] = {"greeting_ms": None, "reply_ms": None, "greeting_s": 0.0, "cleared": False, "error": None}
    step = int(RATE * 2 * CHUNK_S)
    async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        t_start = time.perf_counter()
        await ws.send(json.dumps({"event": "start", "sequence_number": "1", "stream_sid": STREAM_SID,
                                  "start": {"stream_sid": STREAM_SID, "call_sid": "CAbench", "account_sid": "ACbench",
                                            "from": "bench", "to": "bench", "custom_parameters": {},
                                            "media_format": {"encoding": "audio/x-raw", "sample_rate": str(RATE),
                                                             "bit_rate": "16"}}}))
        # the greeting: first byte, then all of it (the caller listens to the end before speaking)
        t_first = None
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            if m.get("event") == "media":
                if t_first is None:
                    t_first = time.perf_counter()
                    out["greeting_ms"] = (t_first - t_start) * 1000
                out["greeting_s"] += len(base64.b64decode(m["media"]["payload"])) / 2 / RATE
            elif m.get("event") == "mark":
                break
        if t_first is None:
            out["error"] = "no greeting audio"
            return out
        await asyncio.sleep(max(0.0, t_first + out["greeting_s"] + 0.3 - time.perf_counter()))

        # the question, in real time, then silence until the reply has been sent
        pcm = question["pcm"]
        reply_first: Dict[str, float] = {}
        done = asyncio.Event()

        async def listen() -> None:
            while not done.is_set():
                try:
                    m = json.loads(await ws.recv())
                except Exception:  # noqa: BLE001 - closed
                    done.set()
                    return
                ev = m.get("event")
                if ev == "media" and "t" not in reply_first:
                    reply_first["t"] = time.perf_counter()
                elif ev == "clear":
                    out["cleared"] = True
                elif ev == "mark" and str((m.get("mark") or {}).get("name", "")).endswith("-end") and "t" in reply_first:
                    done.set()

        listener = asyncio.create_task(listen())
        t0 = time.perf_counter()
        t_eos = t0 + question["speech_end_s"]
        seq, i = 2, 0
        silence = b"\x00\x00" * int(RATE * CHUNK_S)
        deadline = t0 + question["seconds"] + timeout
        while not done.is_set() and time.perf_counter() < deadline:
            chunk = pcm[i:i + step] if i < len(pcm) else silence
            i += step
            # a chunk leaves the phone network once its 100 ms have been spoken
            await asyncio.sleep(max(0.0, t0 + (seq - 1) * CHUNK_S - time.perf_counter()))
            await ws.send(_media(chunk.ljust(step, b"\x00"), seq))
            seq += 1
        done.set()
        listener.cancel()
        if "t" in reply_first:
            out["reply_ms"] = (reply_first["t"] - t_eos) * 1000
        else:
            out["error"] = "no reply audio within the timeout"
        try:
            await ws.send(json.dumps({"event": "stop", "stream_sid": STREAM_SID, "stop": {"reason": "callended"}}))
        except Exception:  # noqa: BLE001
            pass
    return out


def median(values: List[Optional[float]]) -> Optional[float]:
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(p)
    p.add_argument("--agent", required=True, help="agent id (local storage unless --mode cloud)")
    p.add_argument("--mode", default="local", choices=("local", "cloud"))
    p.add_argument("--user-id", default=None, help="owner of the agent (looked up for a local agent)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--question", default="What time does the clinic open on Monday?")
    p.add_argument("--wav", default=None, help="a WAV of the spoken question (default: Kokoro af_heart)")
    p.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for the greeting / the reply")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between runs")
    p.add_argument("--concurrency", type=int, default=1, help="calls at once per run (2 = two callers ringing together)")
    args = p.parse_args()
    runs = 1 if args.quick else max(1, args.runs)

    api = Api(args.api)
    health = require_api(api)
    user_id = args.user_id or (agent_owner(args.agent) if args.mode == "local" else None)
    if not user_id:
        sys.exit("Unknown agent (or give --user-id for a cloud agent).")
    question = question_pcm(api, args.wav, args.question)
    base = args.api.rstrip("/").replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    result = new_result("phone_latency", api, health, {"runs": runs, "question": args.question, "rate": RATE,
                                                       "agent": args.agent, "mode": args.mode, "quick": args.quick})
    print_header("Phone latency (Exotel stream, no real call)", result)
    print(f"stream: {base}/api/phone/exotel/stream/<token>   question: {question['seconds']:.1f} s, speech ends at "
          f"{question['speech_end_s']:.2f} s")
    conc = max(1, args.concurrency)
    result["settings"]["concurrency"] = conc

    async def one_round() -> List[Dict[str, Any]]:
        """`conc` calls at once, each with its own token: what two callers ringing together look like."""
        urls = [f"{base}/api/phone/exotel/stream/{mint_token(args.agent, user_id, args.mode)}" for _ in range(conc)]
        outs = await asyncio.gather(*(one_call(u, question, args.timeout) for u in urls), return_exceptions=True)
        return [o if isinstance(o, dict) else
                {"greeting_ms": None, "reply_ms": None, "error": f"{type(o).__name__}: {str(o)[:120]}"} for o in outs]

    for n in range(runs):
        try:
            batch = asyncio.run(one_round())
        except Exception as e:  # noqa: BLE001
            batch = [{"greeting_ms": None, "reply_ms": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}] * conc
        for i, r in enumerate(batch):
            r["round"], r["slot"] = n + 1, i + 1
            result["items"].append(r)
            g = f"{r['greeting_ms']:.0f} ms" if r.get("greeting_ms") is not None else "n/a"
            rep = f"{r['reply_ms']:.0f} ms" if r.get("reply_ms") is not None else "n/a"
            who = f"run {n + 1}/{runs}" + (f" call {i + 1}/{conc}" if conc > 1 else "")
            print(f"  {who}: start -> greeting {g}   end of speech -> reply {rep}"
                  f"{'   (barge-in clear seen)' if r.get('cleared') else ''}{'   ' + r['error'] if r.get('error') else ''}")
        if n + 1 < runs:
            time.sleep(args.pause)
    replies = sorted(r["reply_ms"] for r in result["items"] if r.get("reply_ms") is not None)
    result["summary"] = {"greeting_ms_median": median([r.get("greeting_ms") for r in result["items"]]),
                         "reply_ms_median": median([r.get("reply_ms") for r in result["items"]]),
                         "reply_ms_p95": replies[min(len(replies) - 1, int(round(0.95 * (len(replies) - 1))))] if replies else None,
                         "reply_ms_max": replies[-1] if replies else None,
                         "ok_runs": sum(1 for r in result["items"] if not r.get("error")), "concurrency": conc}
    s = result["summary"]
    fmt = lambda v: "n/a" if v is None else f"{v:.0f} ms"  # noqa: E731
    print(f"\nmedian start -> first greeting byte: {fmt(s['greeting_ms_median'])}")
    print(f"median end of speech -> first reply byte: {fmt(s['reply_ms_median'])}, p95 {fmt(s['reply_ms_p95'])}, "
          f"max {fmt(s['reply_ms_max'])} over {len(replies)} calls at concurrency {conc} "
          f"(includes the end-of-speech wait, PHONE_END_SILENCE_MS)")
    print(f"results: {write_result(result)}")


if __name__ == "__main__":
    main()
