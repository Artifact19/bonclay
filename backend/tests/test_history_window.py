"""The language model sees only the conversation's last HISTORY_TURNS turns, and only that conversation's: local mode
asks the store for a filtered tail, Supabase mode filters, orders and limits server-side."""
import asyncio

from tests.test_offline_agents import LOCAL, env, make_agent  # noqa: F401  (env is a fixture)
from tests.test_turn_stream import parse, rig  # noqa: F401  (rig is a fixture)


def seed(db, conversation_id, n, prefix):
    for i in range(n):
        db.insert("conversation_turns", {"conversation_id": conversation_id, "role": "user" if i % 2 == 0 else "agent",
                                         "text_content": f"{prefix}{i}"})


def test_select_last_returns_the_newest_matching_rows_oldest_first(tmp_path):
    from services.local_store import LocalStore
    db = LocalStore(str(tmp_path))
    seed(db, "a", 5, "a")
    seed(db, "b", 3, "b")
    rows = db.select_last("conversation_turns", lambda r: r["conversation_id"] == "a", 3)
    assert [r["text_content"] for r in rows] == ["a2", "a3", "a4"]
    assert [r["text_content"] for r in db.select_last("conversation_turns", lambda r: r["conversation_id"] == "b", 9)] \
        == ["b0", "b1", "b2"]
    assert db.select_last("conversation_turns", lambda r: True, 0) == []


def test_a_turn_sends_only_this_conversations_last_turns(rig):
    env = rig.env
    a = make_agent(rig.client)
    conv = rig.client.post(f"/api/conversations/{a['id']}/turn/stream", headers=LOCAL,
                           files={"audio": ("rec.wav", rig.wav, "audio/wav")})
    conv_id = parse(conv)[-1][1]["conversation_id"]
    env.local_db.delete("conversation_turns", lambda r: True)
    seed(env.local_db, conv_id, 10, "mine")
    seed(env.local_db, "other-conversation", 4, "theirs")
    rig.asked.clear()

    parse(rig.post(a["id"], conv_id))
    messages = rig.asked[-1]["messages"]
    assert [m["content"] for m in messages[1:-1]] == [f"mine{i}" for i in range(4, 10)]
    assert env.HISTORY_TURNS == 6


def test_cloud_history_is_filtered_ordered_and_limited_server_side(monkeypatch):
    import backend

    class Query:
        def __init__(self, calls, rows):
            self.calls, self.rows = calls, rows

        def __getattr__(self, name):
            def step(*a, **k):
                self.calls.append((name, a, k))
                return self
            return step

        def execute(self):
            self.calls.append(("execute", (), {}))
            return type("Res", (), {"data": self.rows})()

    calls = []
    newest_first = [{"role": "agent", "text_content": "c"}, {"role": "user", "text_content": "b"},
                    {"role": "agent", "text_content": "a"}]

    class FakeSb:
        def table(self, name):
            calls.append(("table", (name,), {}))
            return Query(calls, [{"id": "conv"}] if name == "conversations" else newest_first)

    run = backend._TurnRun.__new__(backend._TurnRun)
    run.local, run.sb, run.agent_id, run.user_id, run.storage_mode = False, FakeSb(), "ag", "u", "cloud"
    run.user_text, run.timings = "q", {}

    async def no_chunks(*a, **k):
        return []

    monkeypatch.setattr(backend, "retrieve_context", no_chunks)
    asyncio.run(backend._turn_context(run, "conv", 0.0))
    assert [h["text_content"] for h in run.history] == ["a", "b", "c"]  # oldest first for the model
    assert ("eq", ("conversation_id", "conv"), {}) in calls
    assert ("order", ("created_at",), {"desc": True}) in calls
    assert ("limit", (backend.HISTORY_TURNS,), {}) in calls


def test_a_phone_call_sees_far_more_history_than_a_widget_turn(rig):
    """A caller says a phone number in pieces and each piece is a turn, so six rows forgot the name and doctor."""
    import time
    env = rig.env
    a = make_agent(rig.client)
    first = rig.client.post(f"/api/conversations/{a['id']}/turn/stream", headers=LOCAL,
                            files={"audio": ("rec.wav", rig.wav, "audio/wav")})
    conv_id = parse(first)[-1][1]["conversation_id"]
    env.local_db.delete("conversation_turns", lambda r: True)
    seed(env.local_db, conv_id, 60, "t")

    async def history_len(source):
        run = await env._turn_begin(None, a["id"], "local-user", storage_mode="local", agent=a, source=source)
        await env._turn_context(run, conv_id, time.time())
        return [h["text_content"] for h in run.history]

    phone = asyncio.run(history_len("phone"))
    widget = asyncio.run(history_len("widget"))
    assert len(phone) == env.PHONE_HISTORY_TURNS == 40 and phone[-1] == "t59"
    assert len(widget) == env.HISTORY_TURNS == 6
