"""Standalone checks for src/memory.py. Run from the repo root:
PYTHONPATH=. .venv/bin/python test/test_memory.py
"""
import os
import sqlite3
import tempfile

from src import memory


def make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)")
    conn.execute(
        "CREATE TABLE chats (id INTEGER PRIMARY KEY AUTOINCREMENT, topic_id INTEGER, "
        "role TEXT, type TEXT DEFAULT 'text', message TEXT, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO topics (name) VALUES ('test')")
    conn.commit()
    conn.close()
    return path


def add_message(db, role, text):
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO chats (topic_id, role, message) VALUES (1, ?, ?)",
            (role, text),
        )
        conn.commit()
    finally:
        conn.close()


def read_one(db, sql, params=()):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def test_init_db_and_seed():
    db = make_db()
    add_message(db, "user", "早就说过的话")
    memory.init_db(db)
    assert read_one(db, "SELECT last_chat_id FROM memory_runs WHERE topic_id = 1") == 1
    print("init_db seed OK")


def test_crud_and_expiry():
    db = make_db()
    memory.init_db(db)
    memory.init_topic(db, 1)
    memory.upsert_memory(db, "story", "阿黄", "一只会开飞船的狗。")
    memory.upsert_memory(db, "story", "阿黄", "一只会开飞船、爱吃西瓜的狗。")
    assert len(memory.list_memories(db)) == 1
    assert "爱吃西瓜" in memory.list_memories(db)[0]["content"]

    memory.upsert_memory(
        db, "mood", "旧心情", "已经过去很久了。", expires_at="2000-01-01 00:00:00"
    )
    assert len(memory.list_memories(db)) == 1
    assert len(memory.list_memories(db, include_expired=True)) == 2

    story_id = memory.list_memories(db)[0]["id"]
    memory.update_memory_content(db, story_id, "新的内容")
    assert memory.list_memories(db)[0]["content"] == "新的内容"
    memory.delete_memory(db, story_id)
    assert memory.list_memories(db) == []
    print("CRUD OK")


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.chat = type(
            "Chat", (), {"completions": type("Completions", (), {"create": self.create})()}
        )()

    def create(self, **kwargs):
        self.calls += 1
        message = type("Message", (), {"content": self.payload})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


def test_extract_parses_json_and_caps():
    payload = (
        '前缀 {"story":[{"key":"阿黄","content":"会开飞船的狗。"}],'
        '"life":[{"key":"","content":"no key"},{"key":"学校","content":"三年级"}],'
        '"mood":[],"learning":[],"unknown":[{"key":"x","content":"y"}]} 后缀'
    )
    items = memory.extract_memories(FakeClient(payload), "test-model", "小西瓜：阿黄是谁？")
    assert len(items) == 2, items
    assert items[0]["kind"] == "story" and items[0]["key"] == "阿黄"
    assert items[1]["key"] == "学校"
    print("extract parse OK")


def test_mood_flag():
    payload = (
        '{"story":[],"life":[],"mood":[{"key":"被同学笑",'
        '"content":"有点难过。","flag":"需要关注"}],"learning":[]}'
    )
    items = memory.extract_memories(FakeClient(payload), "m", "小西瓜：今天被笑了")
    assert items[0]["flag"] == "需要关注", items
    print("mood flag OK")


def test_extract_bad_json_raises():
    try:
        memory.extract_memories(FakeClient("not json"), "test-model", "hi")
    except ValueError:
        print("bad JSON OK")
        return
    raise AssertionError("expected ValueError")


def test_maybe_extract_threshold_and_force():
    db = make_db()
    memory.init_db(db)
    memory.init_topic(db, 1)
    for i in range(5):
        add_message(db, "user", f"问题{i}")
        add_message(db, "assistant", f"回答{i}")
    payload = (
        '{"story":[],"life":[{"key":"学校","content":"三年级"}],'
        '"mood":[],"learning":[]}'
    )
    client = FakeClient(payload)
    result = memory.maybe_extract_memories(db, 1, client, "test-model")
    assert result["extracted"] == 0 and client.calls == 0, result
    result = memory.maybe_extract_memories(db, 1, client, "test-model", force=True)
    assert result["extracted"] == 1 and client.calls == 1, result
    assert memory.list_memories(db)[0]["key"] == "学校"
    result = memory.maybe_extract_memories(db, 1, client, "test-model", force=True)
    assert result["skipped"] == "no new messages" and client.calls == 1, result
    print("maybe_extract OK")


def test_maybe_extract_failure_keeps_cursor():
    db = make_db()
    memory.init_db(db)
    memory.init_topic(db, 1)
    add_message(db, "user", "你好")
    client = FakeClient("not json")
    result = memory.maybe_extract_memories(db, 1, client, "test-model", force=True)
    assert "error" in result, result
    last = read_one(db, "SELECT last_chat_id FROM memory_runs WHERE topic_id = 1")
    assert last == 0, last
    print("failure cursor OK")


def test_topic_extraction_merges():
    db = make_db()
    memory.init_db(db)
    for i in range(3):
        add_message(db, "user", f"继续讲故事{i}")
        add_message(db, "assistant", f"好呀{i}")
    payload = (
        '{"story":[{"key":"阿黄","content":"会开飞船"}],'
        '"life":[],"mood":[],"learning":[]}'
    )
    items = memory.extract_topic_memories(db, 1, FakeClient(payload), "test-model", chunk_size=2)
    assert len(items) == 1 and items[0]["key"] == "阿黄", items
    print("topic extraction merge OK")


def test_context_ordering_and_cap():
    db = make_db()
    memory.init_db(db)
    memory.upsert_memory(db, "story", "阿黄", "会开飞船的狗。")
    memory.upsert_memory(db, "story", "小白", "会种西瓜的猫。")
    memory.upsert_memory(db, "learning", "分数", "分母相同才能直接相加。")
    memory.upsert_memory(db, "life", "好朋友", "最好的朋友叫小明。")
    context = memory.build_memory_context(db, "小白今天怎么样")
    assert "故事书" in context and "学习" in context and "生活" in context
    assert context.index("小白") < context.index("阿黄"), context
    assert len(context) <= memory.MAX_CONTEXT_CHARS + 50, len(context)

    memory.upsert_memory(
        db,
        "mood",
        "旧心情",
        "已经过去很久了。",
        expires_at="2000-01-01 00:00:00",
    )
    assert "旧心情" not in memory.build_memory_context(db)
    print("context OK")


def test_memory_counts():
    db = make_db()
    memory.init_db(db)
    memory.upsert_memory(db, "learning", "分数", "分母相同才能相加。")
    memory.upsert_memory(
        db,
        "mood",
        "被同学笑",
        "有点难过。",
        flag=memory.ATTENTION_FLAG,
    )
    counts = memory.memory_counts(db)
    assert counts["total"] == 2, counts
    assert counts["learning_week"] == 1, counts
    assert counts["attention"] == 1, counts
    print("counts OK")


if __name__ == "__main__":
    test_init_db_and_seed()
    test_crud_and_expiry()
    test_extract_parses_json_and_caps()
    test_mood_flag()
    test_extract_bad_json_raises()
    test_maybe_extract_threshold_and_force()
    test_maybe_extract_failure_keeps_cursor()
    test_topic_extraction_merges()
    test_context_ordering_and_cap()
    test_memory_counts()
    print("memory tests OK")
