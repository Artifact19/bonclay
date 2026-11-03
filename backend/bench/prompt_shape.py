"""Shape of the chat messages one agent sends to its language model across consecutive turns.

Builds the messages of three turns with the real assembly code (backend._turn_messages, then services.tools.prepare_messages
for an agent with tools, as the tool loop does) and prints, per turn:
  - characters and approximate tokens of the serialized messages (tiktoken o200k_base, an approximation of the
    model's tokenizer; chars/4 when tiktoken is missing), and of the system message alone;
  - the longest common prefix with the previous turn's serialized messages, in characters and as a share of this turn;
  - whether the system message is byte-identical to the previous turn's, and which message the prefix ends in.

Retrieval, the fact sheet and the clock are faked (fixed passages, fixed facts, fixed times), so the output is the same on
every run. No model is called, no server is started, and no file under backend/data is read.

Usage (from backend/):  .venv/bin/python bench/prompt_shape.py [--out FILE.txt]
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("FACT_SHEET", "off")          # the fact sheet is replaced by a fake below
os.environ.setdefault("CONFIDENCE_GATE", "off")
os.environ.setdefault("ANSWER_GUARD", "off")

import backend  # noqa: E402  (the real assembly functions)
from services import fact_sheet, language_policy  # noqa: E402
from services import tools as tools_mod  # noqa: E402

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    TOKEN_METHOD = "tiktoken o200k_base"
except Exception:  # noqa: BLE001 - offline or not installed
    def count_tokens(text: str) -> int:
        return len(text) // 4
    TOKEN_METHOD = "chars/4"


CLINIC = {"name": "Ria", "system_instructions": "the front desk of Sahyadri Clinic, Pune. Help callers with OPD timings, "
          "consultation fees and appointments.", "supported_languages": ["en", "hi"]}
CLINIC_NO_TOOLS = dict(CLINIC, name="Asha")

PASSAGES = [
    {"header": "Sahyadri Clinic > OPD", "content": "OPD timings: Monday to Saturday, 9:30 AM to 1:00 PM. Sunday closed."},
    {"header": "Sahyadri Clinic > Fees", "content": "Consultation fee: Rs. 500 (general physician), Rs. 800 (cardiology)."},
    {"header": "Sahyadri Clinic > Doctors", "content": "Dr. Mehta (Cardiology) sits on Tuesday and Friday."},
    {"header": "Sahyadri Clinic > Contact", "content": "Helpline +91 98765 43210. Address: 12 FC Road, Pune 411004."},
    {"header": "Sahyadri Clinic > Lab", "content": "Blood tests are reported within 24 hours; fasting samples before 10 AM."},
]
FACTS = {
    "fee": "Consultation fee, general physician: Rs. 500",
    "open": "OPD timings, Monday to Saturday: 9:30 AM to 1:00 PM",
    "doctor": "Dr. Mehta (Cardiology) days: Tuesday and Friday",
}
SPEC = tools_mod.ToolSpec(
    server="appointments", tool="book_appointment",
    description="Book an appointment for the caller with a doctor on a day and time.",
    schema={"type": "object", "required": ["name", "phone", "date", "time"],
            "properties": {"name": {"type": "string", "description": "the caller's name"},
                           "phone": {"type": "string", "description": "10 digits"},
                           "date": {"type": "string", "description": "YYYY-MM-DD"},
                           "time": {"type": "string", "description": "HH:MM, 24-hour"},
                           "doctor": {"type": "string"}, "reason": {"type": "string"}}})

# Three consecutive turns of one caller. Turn 2 switches to Hindi in one scenario.
TURNS = [
    {"says": "What is the consultation fee for a general physician?", "at": datetime(2026, 10, 23, 10, 14),
     "reply": "The consultation fee is Rs. 500 for the general physician."},
    {"says": "और ओपीडी कब खुलती है?", "at": datetime(2026, 10, 23, 10, 16),
     "reply": "ओपीडी सोमवार से शनिवार, सुबह 9:30 बजे से दोपहर 1:00 बजे तक खुलती है।"},
    {"says": "Can I book with Dr. Mehta on Friday at 11 am?", "at": datetime(2026, 10, 23, 10, 19),
     "reply": "Dr. Mehta sits on Friday. May I have your name and phone number to book?"},
]
CALLER_NAME_FROM_TURN = 3   # the caller has given a name from the third turn on (the details form)


def passages_for(question: str):
    """The four passages that share the most words with the question (a stand-in for retrieval)."""
    words = {w.strip("?.,!").lower() for w in question.split() if len(w) > 3}
    score = lambda p: sum(1 for w in words if w in (p["header"] + " " + p["content"]).lower())  # noqa: E731
    return sorted(PASSAGES, key=lambda p: -score(p))[:4]


def facts_for(question: str):
    q = question.lower()
    out = []
    if "fee" in q or "consultation" in q:
        out.append(FACTS["fee"])
    if "opd" in q or "ओपीडी" in q or "open" in q:
        out.append(FACTS["open"])
    if "doctor" in q or "dr." in q:
        out.append(FACTS["doctor"])
    return out


def build_turns(agent: dict, specs, turns, *, set_clock):
    """The messages each turn sends, as the turn code builds them."""
    sent, history, previous_agent = [], [], None
    for i, t in enumerate(turns):
        run = backend._TurnRun(None, "agent-bench", "user-bench", storage_mode="local")
        run.agent = agent
        run.user_text = t["says"]
        run.conversation_id = "conv-bench"
        run.history = list(history)
        run.history_known = True
        run.chunks = passages_for(t["says"])
        run.tool_specs = list(specs)
        decision = language_policy.decide(run.user_text, agent["supported_languages"], None, previous_agent)
        run.language, run.language_decision = decision.target, decision
        if i + 1 >= CALLER_NAME_FROM_TURN:
            run.caller_form = {"name": "Priya"}
        set_clock(t["at"])
        messages = backend._turn_messages(run)
        if specs:   # the tool loop's first request (services/tools.py run_tool_turn / ToolStream)
            messages = tools_mod.prepare_messages(messages, specs, tools_mod.tool_protocol() != "native",
                                                  known_slots=tools_mod.known_slots(run.history))
        sent.append(messages)
        history += [{"role": "user", "text_content": t["says"]}, {"role": "agent", "text_content": t["reply"]}]
        previous_agent = t["reply"]
    return sent


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo = 0
    while lo < n and a[lo] == b[lo]:
        lo += 1
    return lo


def first_difference_message(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def report(title: str, sent: list) -> list:
    lines = [f"== {title}", f"tokens: {TOKEN_METHOD} of the JSON-serialized messages",
             f"{'turn':>4} {'chars':>7} {'tokens':>7} {'system tok':>10} {'LCP chars':>10} {'LCP %':>7} "
             f"{'system same':>12} {'prefix ends in message':>24}"]
    prev = None
    for i, messages in enumerate(sent, 1):
        ser = json.dumps(messages, ensure_ascii=False)
        sys_tokens = count_tokens(messages[0]["content"])
        if prev is None:
            lcp, pct, same, ends = 0, 0.0, "-", "-"
        else:
            lcp = common_prefix(prev, ser)
            pct = 100.0 * lcp / len(ser)
            same = "yes" if messages[0] == prev_messages[0] else "NO"
            idx = first_difference_message(prev_messages, messages)
            ends = f"{idx} ({messages[idx]['role']})" if idx < len(messages) else "-"
        lines.append(f"{i:>4} {len(ser):>7} {count_tokens(ser):>7} {sys_tokens:>10} {lcp:>10} {pct:>6.1f}% "
                     f"{same:>12} {ends:>24}")
        prev, prev_messages = ser, messages
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="also write the report to this file")
    args = ap.parse_args()

    real_now_line = tools_mod.now_line
    clock = {"at": None}

    def set_clock(at):
        clock["at"] = at
    tools_mod.now_line = lambda now=None: real_now_line(clock["at"])
    fact_sheet.turn_facts = lambda run: facts_for(run.user_text)

    out = []
    scenarios = [
        ("A. agent with tools, caller speaks English all three turns", CLINIC, [SPEC], TURNS[0:1] + [
            {"says": "Is Dr. Mehta available on Friday?", "at": datetime(2026, 10, 23, 10, 16),
             "reply": "Dr. Mehta sits on Friday."}, TURNS[2]]),
        ("B. agent with tools, caller switches to Hindi on turn 2, back to English on turn 3", CLINIC, [SPEC], [
            TURNS[0], TURNS[1], {"says": "Okay, then book me with Dr. Mehta on Friday at 11 am.",
                                 "at": datetime(2026, 10, 23, 10, 19), "reply": TURNS[2]["reply"]}]),
        ("C. agent without tools, caller speaks English all three turns", CLINIC_NO_TOOLS, [], [
            TURNS[0], {"says": "What time does the OPD open on Saturday?", "at": datetime(2026, 10, 23, 10, 16),
                       "reply": "The OPD opens at 9:30 AM on Saturday."},
            {"says": "Thanks. Where is the clinic?", "at": datetime(2026, 10, 23, 10, 19),
             "reply": "We are at 12 FC Road, Pune 411004."}]),
    ]
    try:
        for title, agent, specs, turns in scenarios:
            sent = build_turns(agent, specs, turns, set_clock=set_clock)
            out += report(title, sent) + [""]
    finally:
        tools_mod.now_line = real_now_line
    text = "\n".join(out).rstrip() + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
