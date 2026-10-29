# 小南瓜长期记忆 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Load `developing-with-streamlit` before editing `pages/10_🎙️_FreeChat.py`. Execute tasks in order. Do not commit unless Jerry asks.

**Goal:** 给 FreeChat 的小南瓜加上四类长期记忆——故事书、生活、心情、学习——每 10 轮增量提取、切换话题时补一次、随时手动“记住这段”；记忆在对话时临时注入；旧话题可回填；侧边栏可看、可改、可忘。OpenChat 不改。

**Architecture:** `src/memory.py` 是纯 Python + SQLite 的记忆核心，不导入 Streamlit。页面只做四件事：初始化表、在合适的时机调用提取、把记忆临时拼进 system message、展示和编辑记忆。提取失败只打印日志，不阻塞聊天。记忆不写回 `FreeChatMessages` 或 `chats` 表。

**Tech Stack:** Python 3.11、`sqlite3`、OpenAI SDK（OpenRouter）、Streamlit 1.66.1。仓库没有 pytest；测试是 `test/` 下的独立断言脚本，加 Streamlit `AppTest` 冒烟测试。

---

## 当前状态与约束

- 在仓库根目录运行，Python 用 `.venv`。页面从仓库根目录解析 `./setting/...`，所以每条命令都从根目录执行。
- 工作区里有 Jerry 自己的未提交改动（`config.json`、`key.txt`）和未跟踪的 `.env`、`.vscode/` 等；不要动它们。依赖清理已经在最近的提交里。
- `setting/chat_history_nangua.db` 是真实的聊天库，已被 git 跟踪。运行页面会在这个库里建新表；这是预期行为。跑回填的 `--apply` 前先备份。
- `key.txt` 和 `.streamlit/secrets.toml` 有真实凭据，不要打印内容，也不要把草稿或备份提交进 git。
- `test/` 是手动实验目录，不是 pytest 测试集；不要引入 pytest、向量库或 RAG。
- 只改和小南瓜有关的文件。OpenChat、`src/llms.py`、其他页面不动。
- 提交只在 Jerry 明确要求时做；如果提交，先建分支。计划里的“Checkpoint”只查看 diff，不提交。

## 设计决策

- **四类，一类表**：`memories.kind` 取 `story | life | mood | learning`，不为每类建表。
- **单孩子**：不加 `child` 列。现在只有小西瓜；将来第二个孩子用这个应用时再加。
- **不做向量检索**：记忆量小（每类最多几十条），按关键词/名字优先，加全量注入即可。`build_memory_context()` 上限约 2000 字，每类约 500 字，避免故事书挤掉学习。
- **心情 30 天淡出**：`mood` 记录 `expires_at`；过期行保留在库里但不注入、不显示。回填的旧心情按话题最后日期 + 30 天计算，旧心情不会突然“复活”。
- **提取游标**：`memory_runs(topic_id, last_chat_id)` 记录每个话题已经整理到哪条消息。`init_db()` 给已有话题的游标设到当前最新消息，避免第一次运行就把 974 条旧消息全读一遍；旧历史交给回填脚本。新话题从 0 开始。
- **提取时机**：自动提取放在下一轮用户发言的开头（此时上一轮的回复已经入库），每 10 条新用户消息触发一次。这样不会在回复结束后、孩子听语音时卡住。切换话题时强制补一次；手动按钮随时强制。
- **记忆注入**：只改发给模型的临时副本；`FreeChatMessages` 和数据库里的 system message 不变。第二次工具调用重新生成临时副本，确保包含 tool 消息。
- **无 PIN**：故事书、生活、心情、学习都对小西瓜可见；孩子可以编辑、忘掉、清空。信任孩子。
- **需要关注**：心情条目可以由提取模型带 `flag="需要关注"`。它只是记忆上的标记，家长视图能看到；聊天里仍然走已有的安全提示，绝不说“我会告诉家长”。
- **后续**：TTS 改成边生成边播不在本计划内。先把记忆跑稳，再单独做 TTS。

---

## Task 1: 记忆表与 CRUD

**Files:**
- Create: `src/memory.py`
- Create: `test/test_memory.py`

### Step 1: 写失败的测试

新建 `test/test_memory.py`：

```python
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


if __name__ == "__main__":
    test_init_db_and_seed()
    test_crud_and_expiry()
    print("memory CRUD tests OK")
```

### Step 2: 运行测试，确认失败

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
```

Expected: `ModuleNotFoundError: No module named 'src.memory'`（`src/memory.py` 还不存在）。

### Step 3: 实现表和 CRUD

新建 `src/memory.py`：

```python
"""Long-term memory for 小南瓜.

Pure Python + SQLite. No Streamlit imports, so the same module can be tested
headlessly and reused by a future UI. The caller passes an OpenAI-compatible
client and model name.
"""

import sqlite3
from contextlib import contextmanager
from typing import List, Optional

KINDS = ("story", "life", "mood", "learning")
KIND_LABELS = {"story": "故事书", "life": "生活", "mood": "心情", "learning": "学习"}
MOOD_TTL_DAYS = 30
AUTO_EXTRACT_EVERY = 10
MAX_KEY_CHARS = 40
MAX_CONTENT_CHARS = 300


@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Create the memory tables and give existing topics a cursor at their current end."""
    with _connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                source_topic_id INTEGER,
                flag TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memory_runs (
                topic_id INTEGER PRIMARY KEY,
                last_chat_id INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        has_tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('topics', 'chats')"
        ).fetchone()[0] == 2
        if has_tables:
            conn.execute(
                """INSERT OR IGNORE INTO memory_runs (topic_id, last_chat_id)
                   SELECT t.id, COALESCE((SELECT MAX(c.id) FROM chats c
                                          WHERE c.topic_id = t.id), 0)
                   FROM topics t"""
            )


def init_topic(db_path: str, topic_id: int) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO memory_runs (topic_id, last_chat_id) VALUES (?, 0)",
            (topic_id,),
        )


def list_memories(db_path: str, include_expired: bool = False) -> List[dict]:
    query = (
        "SELECT id, kind, key, content, source_topic_id, flag, "
        "created_at, updated_at, expires_at FROM memories"
    )
    if not include_expired:
        query += " WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP"
    query += (
        " ORDER BY CASE kind WHEN 'story' THEN 0 WHEN 'learning' THEN 1 "
        "WHEN 'life' THEN 2 ELSE 3 END, updated_at DESC"
    )
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(query).fetchall()]


def upsert_memory(
    db_path: str,
    kind: str,
    key: str,
    content: str,
    source_topic_id: Optional[int] = None,
    expires_at: Optional[str] = None,
    flag: Optional[str] = None,
) -> None:
    kind = (kind or "").strip()
    key = (key or "").strip()[:MAX_KEY_CHARS]
    content = (content or "").strip()[:MAX_CONTENT_CHARS]
    if kind not in KINDS or not key or not content:
        return
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM memories WHERE kind = ? AND key = ?", (kind, key)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE memories SET content = ?, "
                "source_topic_id = COALESCE(?, source_topic_id), "
                "updated_at = CURRENT_TIMESTAMP, expires_at = ?, flag = ? WHERE id = ?",
                (content, source_topic_id, expires_at, flag, row["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO memories (kind, key, content, source_topic_id, expires_at, flag) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, key, content, source_topic_id, expires_at, flag),
            )


def update_memory_content(db_path: str, memory_id: int, content: str) -> None:
    content = (content or "").strip()[:MAX_CONTENT_CHARS]
    if not content:
        return
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE memories SET content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (content, memory_id),
        )


def delete_memory(db_path: str, memory_id: int) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


def clear_memories(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM memories")


def has_due_memories(
    db_path: str, topic_id: int, threshold: int = AUTO_EXTRACT_EVERY
) -> bool:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_chat_id FROM memory_runs WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        last_id = row["last_chat_id"] if row else 0
        count = conn.execute(
            "SELECT COUNT(*) FROM chats WHERE topic_id = ? AND id > ? AND role = 'user'",
            (topic_id, last_id),
        ).fetchone()[0]
    return count >= threshold


def has_new_messages(db_path: str, topic_id: int) -> bool:
    """True when a topic has any user/assistant message after its memory cursor."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_chat_id FROM memory_runs WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        last_id = row["last_chat_id"] if row else 0
        count = conn.execute(
            "SELECT COUNT(*) FROM chats WHERE topic_id = ? AND id > ? "
            "AND role IN ('user', 'assistant')",
            (topic_id, last_id),
        ).fetchone()[0]
    return count > 0
```

### Step 4: 运行测试，确认通过

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
```

Expected:

```text
init_db seed OK
CRUD OK
memory CRUD tests OK
```

### Step 5: 编译检查

```bash
.venv/bin/python -m py_compile src/memory.py test/test_memory.py
```

Expected: 没有输出。

### Step 6: Checkpoint

```bash
git status --short
```

Expected: 看到未跟踪的 `src/memory.py`、`test/test_memory.py`、`docs/plans/...`；`config.json`/`key.txt` 是 Jerry 自己的改动，不要动。不要提交。

---

## Task 2: 提取、游标和回填用的整话题提取

**Files:**
- Modify: `src/memory.py`
- Modify: `test/test_memory.py`

### Step 1: 追加失败的测试

在 `test/test_memory.py` 的 `test_crud_and_expiry` 后面加：

```python
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
```

把这六个函数加进 `if __name__ == "__main__":` 的调用列表，最后保留 `print("memory tests OK")`。

### Step 2: 运行测试，确认失败

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
```

Expected: `AttributeError: module 'src.memory' has no attribute 'extract_memories'`。

### Step 3: 实现提取

在 `src/memory.py` 顶部补 import：

```python
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
```

在常量区补：

```python
MAX_NEW_MESSAGES = 80
MAX_ITEMS_PER_KIND = 5
EXISTING_MEMORY_CHARS = 1800
ATTENTION_FLAG = "需要关注"

_EXTRACTION_SYSTEM_PROMPT = """你是小南瓜的记忆整理员。请从下面的对话里提取值得长期记住的内容，只输出 JSON。

分类：
- story（故事书）：小西瓜和小南瓜一起编的故事，包括角色名字和性格、世界观、还没讲完的情节。
- life（生活）：小西瓜生活中稳定的事，包括家人朋友、学校、喜好、最近发生的事、计划。
- mood（心情）：小西瓜最近的心情、原因，以及什么让他好受一些。
- learning（学习）：小西瓜学过的知识、卡住的地方、哪种讲法听懂了、他好奇的问题。

规则：
1. 只记录对话里明确出现的内容，不要推测，不要编造。
2. 每条 content 用 1-2 句孩子能听懂的中文。
3. 可以保留故事角色名和常用称呼；不要记录家庭住址、电话、学校全名、身份证号等敏感信息。
4. 如果某条和“已有记忆”重复，用同一个 key 返回更新后的内容，不要重复列出。
5. 每类最多 5 条；没有值得记的就返回空数组。
6. 如果小西瓜表达持续的低落、害怕、被欺负或伤害自己的想法，在 mood 条目里加 "flag": "需要关注"；不要写“我会告诉家长”。
7. 只输出 JSON，不要 Markdown 代码块，不要解释。

输出格式：
{"story":[{"key":"...","content":"..."}],"life":[],"mood":[],"learning":[]}
"""
```

在 `has_new_messages` 后面追加：

```python
def _parse_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_items(data: dict) -> List[dict]:
    items = []
    seen = set()
    for kind in KINDS:
        raw_items = data.get(kind) or []
        if not isinstance(raw_items, list):
            continue
        for raw in raw_items[:MAX_ITEMS_PER_KIND]:
            if not isinstance(raw, dict):
                continue
            key = raw.get("key")
            content = raw.get("content")
            if not isinstance(key, str) or not isinstance(content, str):
                continue
            key = key.strip()[:MAX_KEY_CHARS]
            content = content.strip()[:MAX_CONTENT_CHARS]
            if not key or not content:
                continue
            flag_value = raw.get("flag")
            flag = (
                ATTENTION_FLAG
                if kind == "mood"
                and isinstance(flag_value, str)
                and flag_value.strip() == ATTENTION_FLAG
                else None
            )
            dedupe_key = (kind, key)
            if dedupe_key in seen:
                items = [
                    item
                    for item in items
                    if (item["kind"], item["key"]) != dedupe_key
                ]
            seen.add(dedupe_key)
            items.append(
                {"kind": kind, "key": key, "content": content, "flag": flag}
            )
    return items


def _transcript(rows: Iterable[sqlite3.Row]) -> str:
    lines = []
    for row in rows:
        speaker = "小西瓜" if row["role"] == "user" else "小南瓜"
        text = "[发了一张图片]" if row["type"] == "image_url" else row["message"]
        lines.append(f"{speaker}：{text}")
    return "\n".join(lines)


def _existing_text(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT kind, key, content FROM memories "
        "WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP "
        "ORDER BY updated_at DESC LIMIT 60"
    ).fetchall()
    text = "\n".join(
        f"- [{row['kind']}] {row['key']}：{row['content'][:100]}" for row in rows
    )
    return text[:EXISTING_MEMORY_CHARS]


def extract_memories(
    client: Any, model: str, transcript: str, existing_text: str = ""
) -> List[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"已有记忆：\n{existing_text or '（无）'}\n\n对话：\n{transcript}",
            },
        ],
        temperature=0.2,
        max_tokens=4000,
        stream=False,
    )
    text = response.choices[0].message.content or ""
    data = _parse_json(text)
    if data is None:
        raise ValueError("memory extraction returned invalid JSON")
    return _normalize_items(data)


def apply_memories(
    db_path: str, items: Iterable[dict], source_topic_id: Optional[int] = None
) -> int:
    count = 0
    for item in items:
        kind = item.get("kind")
        key = item.get("key")
        content = item.get("content")
        if (
            kind not in KINDS
            or not isinstance(key, str)
            or not isinstance(content, str)
        ):
            continue
        expires_at = item.get("expires_at")
        flag = None
        if kind == "mood":
            if not expires_at:
                expires_at = (
                    datetime.now(timezone.utc) + timedelta(days=MOOD_TTL_DAYS)
                ).strftime("%Y-%m-%d %H:%M:%S")
            if item.get("flag") == ATTENTION_FLAG:
                flag = ATTENTION_FLAG
        else:
            expires_at = None
        upsert_memory(
            db_path,
            kind,
            key,
            content,
            source_topic_id=source_topic_id or item.get("source_topic_id"),
            expires_at=expires_at,
            flag=flag,
        )
        count += 1
    return count


def maybe_extract_memories(
    db_path: str,
    topic_id: int,
    client: Any,
    model: str,
    force: bool = False,
    threshold: int = AUTO_EXTRACT_EVERY,
    window: int = 40,
) -> Dict[str, Any]:
    """Extract new messages for one topic.

    force=False: only when at least `threshold` new user messages exist.
    force=True: process the last `window` new messages (topic switch / manual button).
    Returns {"extracted": n, "skipped": reason} or {"extracted": 0, "error": str}.
    """
    if not topic_id:
        return {"extracted": 0, "skipped": "no topic"}
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_chat_id FROM memory_runs WHERE topic_id = ?", (topic_id,)
        ).fetchone()
        last_id = row["last_chat_id"] if row else 0
        rows = conn.execute(
            "SELECT id, role, type, message FROM chats "
            "WHERE topic_id = ? AND id > ? AND role IN ('user', 'assistant') "
            "ORDER BY id",
            (topic_id, last_id),
        ).fetchall()
        user_count = sum(1 for row in rows if row["role"] == "user")
        if not rows:
            return {"extracted": 0, "skipped": "no new messages"}
        if not force and user_count < threshold:
            return {"extracted": 0, "skipped": f"{user_count} new user messages"}
        selected = rows[-max(1, window):] if force else rows
        if len(selected) > MAX_NEW_MESSAGES:
            selected = selected[-MAX_NEW_MESSAGES:]
        transcript = _transcript(selected)
        existing_text = _existing_text(conn)

    try:
        items = extract_memories(client, model, transcript, existing_text)
    except Exception as exc:
        return {"extracted": 0, "error": str(exc)}

    extracted = apply_memories(db_path, items, source_topic_id=topic_id)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE memory_runs SET last_chat_id = ?, updated_at = CURRENT_TIMESTAMP "
            "WHERE topic_id = ?",
            (selected[-1]["id"], topic_id),
        )
    return {"extracted": extracted, "skipped": ""}


def extract_topic_memories(
    db_path: str, topic_id: int, client: Any, model: str, chunk_size: int = 40
) -> List[dict]:
    """Extract a whole existing topic in chunks, without writing to the DB."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, type, message FROM chats "
            "WHERE topic_id = ? AND role IN ('user', 'assistant') ORDER BY id",
            (topic_id,),
        ).fetchall()
    items: List[dict] = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        existing_text = "\n".join(
            f"- [{item['kind']}] {item['key']}：{item['content'][:100]}"
            for item in items
        )[-EXISTING_MEMORY_CHARS:]
        extracted = extract_memories(client, model, _transcript(chunk), existing_text)
        for item in extracted:
            items = [
                old
                for old in items
                if (old["kind"], old["key"]) != (item["kind"], item["key"])
            ]
            items.append(item)
    return items
```

### Step 4: 运行测试，确认通过

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
```

Expected:

```text
init_db seed OK
CRUD OK
extract parse OK
mood flag OK
bad JSON OK
maybe_extract OK
failure cursor OK
topic extraction merge OK
memory tests OK
```

### Step 5: 编译检查

```bash
.venv/bin/python -m py_compile src/memory.py test/test_memory.py
```

Expected: 没有输出。

### Step 6: Checkpoint

```bash
git status --short
```

Expected: 只看到 `src/memory.py` 和 `test/test_memory.py` 的变化。不要提交。

---

## Task 3: 记忆注入文本和统计

**Files:**
- Modify: `src/memory.py`
- Modify: `test/test_memory.py`

### Step 1: 追加失败的测试

在 `test/test_memory.py` 里加：

```python
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
```

把这两个函数加进 `__main__` 的调用列表。

### Step 2: 运行测试，确认失败

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
```

Expected: `AttributeError: module 'src.memory' has no attribute 'build_memory_context'`。

### Step 3: 实现注入文本和统计

在 `src/memory.py` 的常量区补：

```python
MAX_CONTEXT_CHARS = 2000
```

在 `has_new_messages` 后面追加：

```python
def build_memory_context(
    db_path: str, user_message: str = "", max_chars: int = MAX_CONTEXT_CHARS
) -> str:
    """Build a temporary system-message block. Never persisted."""
    memories = list_memories(db_path)
    if not memories:
        return ""
    header = [
        "# 小南瓜记得的事",
        "只在和当前话题相关时自然使用；不要逐条复述，"
        "不要说“根据我的记忆”；每条回复最多提一次。",
    ]
    sections = []
    used = sum(len(line) for line in header)
    per_kind_cap = max_chars // len(KINDS)

    for kind in KINDS:
        items = [m for m in memories if m["kind"] == kind]
        if not items:
            continue
        # Recent first, then move items whose key appears in the current message to the front.
        items.sort(key=lambda m: m["updated_at"] or "", reverse=True)
        items.sort(key=lambda m: 0 if m["key"] in user_message else 1)
        lines = [f"## {KIND_LABELS[kind]}"]
        section_used = len(lines[0])
        for item in items:
            line = f"- {item['key']}：{item['content']}"
            if section_used + len(line) > per_kind_cap or used + len(line) > max_chars:
                break
            lines.append(line)
            section_used += len(line)
            used += len(line)
        if len(lines) > 1:
            sections.extend(lines)
            used += len(lines[0])
    if not sections:
        return ""
    return "\n".join(header + sections)


def memory_counts(db_path: str) -> Dict[str, int]:
    memories = list_memories(db_path)
    week_ago = (
        datetime.now(timezone.utc) - timedelta(days=7)
    ).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "total": len(memories),
        "learning_week": sum(
            1
            for m in memories
            if m["kind"] == "learning" and (m["updated_at"] or "") >= week_ago
        ),
        "attention": sum(1 for m in memories if m.get("flag")),
    }
```

### Step 4: 运行测试，确认通过

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
```

Expected: 最后一行是 `memory tests OK`，并包含 `context OK` 和 `counts OK`。

### Step 5: 编译检查

```bash
.venv/bin/python -m py_compile src/memory.py test/test_memory.py
```

Expected: 没有输出。

### Step 6: Checkpoint

```bash
git status --short
```

不要提交。

---

## Task 4: 接入 FreeChat（初始化、注入、自动提取、切话题）

**Files:**
- Modify: `pages/10_🎙️_FreeChat.py:1-6`（import）
- Modify: `pages/10_🎙️_FreeChat.py:40-60`（`init_database`）
- Modify: `pages/10_🎙️_FreeChat.py:72-91`（`create_topic`）
- Modify: `pages/10_🎙️_FreeChat.py:94-99`（在 `save_chat_to_db` 后加 helper）
- Modify: `pages/10_🎙️_FreeChat.py:349-351`（切话题）
- Modify: `pages/10_🎙️_FreeChat.py:528-529`（自动提取）
- Modify: `pages/10_🎙️_FreeChat.py:554`（第一次调用）
- Modify: `pages/10_🎙️_FreeChat.py:632`（第二次调用）
- Create: `test/test_freechat_memory_ui.py`

> 这一层是 UI 胶水；核心逻辑已在 Task 1–3 用独立脚本覆盖。这里用 `py_compile`、AppTest 冒烟和手动跑页面验证。

### Step 1: 改 import

在 `pages/10_🎙️_FreeChat.py` 第 5 行后加：

```python
from src import memory
```

### Step 2: 初始化记忆表

在 `init_database()` 的最后一个 `c.execute(...)` 后，`with` 块外，加：

```python
    memory.init_db(CHAT_HISTORY_PATH)
```

改完后 `init_database` 的结尾应是：

```python
        # Add indexes
        c.execute('''CREATE INDEX IF NOT EXISTS idx_topic_id ON chats(topic_id)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_created_at ON chats(created_at)''')
    memory.init_db(CHAT_HISTORY_PATH)
```

### Step 3: 新话题初始化游标

在 `create_topic()` 里，插入话题并拿到 `topic_id` 后加：

```python
            memory.init_topic(CHAT_HISTORY_PATH, topic_id)
```

位置：

```python
            with DatabaseConnection(CHAT_HISTORY_PATH) as c:
                c.execute("INSERT INTO topics (name) VALUES (?)", (st.session_state.new_topic,))
                topic_id = c.lastrowid
            memory.init_topic(CHAT_HISTORY_PATH, topic_id)
            st.session_state.current_topic_id = topic_id
```

### Step 4: 加临时注入 helper

在 `save_chat_to_db()` 后面加：

```python
def build_api_messages(prompt: str) -> List[dict]:
    """Copy the history and add long-term memory to the system message.

    The memory block is never written back to FreeChatMessages or the database.
    """
    messages = list(st.session_state.FreeChatMessages)
    try:
        context = memory.build_memory_context(CHAT_HISTORY_PATH, prompt)
    except Exception as e:
        print(f"Memory context failed: {e}")
        context = ""
    if context and messages and messages[0].get("role") == "system":
        system_message = dict(messages[0])
        content = system_message.get("content")
        if isinstance(content, list) and content:
            system_text = content[0].get("text", "")
        elif isinstance(content, str):
            system_text = content
        else:
            system_text = ""
        system_message["content"] = [
            {"type": "text", "text": system_text + "\n\n" + context}
        ]
        messages[0] = system_message
    return messages
```

### Step 5: 切话题时补一次提取

在 `pages/10_🎙️_FreeChat.py:349` 的 topic 按钮里，把：

```python
                        if st.button(f"📝 {topic_name}", key=f"topic_{topic_id}"):
                            st.session_state.FreeChatReloadMode = True
                            st.session_state.current_topic_id = topic_id
```

改成：

```python
                        if st.button(f"📝 {topic_name}", key=f"topic_{topic_id}"):
                            previous_topic_id = st.session_state.current_topic_id
                            if (
                                previous_topic_id
                                and previous_topic_id != topic_id
                                and memory.has_new_messages(
                                    CHAT_HISTORY_PATH, previous_topic_id
                                )
                            ):
                                with st.spinner("小南瓜在回想刚才的聊天..."):
                                    try:
                                        memory.maybe_extract_memories(
                                            CHAT_HISTORY_PATH,
                                            previous_topic_id,
                                            st.session_state.FreeChatChain,
                                            st.session_state.FreeChatSetting["model"],
                                            force=True,
                                        )
                                    except Exception as e:
                                        print(f"Memory extraction failed: {e}")
                            st.session_state.FreeChatReloadMode = True
                            st.session_state.current_topic_id = topic_id
```

### Step 6: 每 10 轮自动提取

在 `pages/10_🎙️_FreeChat.py:529` 的 `if (prompt := ...):` 后、`# Add user message to chat history` 前加：

```python
            if st.session_state.current_topic_id and memory.has_due_memories(
                CHAT_HISTORY_PATH, st.session_state.current_topic_id
            ):
                with st.spinner("小南瓜在回想之前聊过的事..."):
                    try:
                        memory.maybe_extract_memories(
                            CHAT_HISTORY_PATH,
                            st.session_state.current_topic_id,
                            st.session_state.FreeChatChain,
                            st.session_state.FreeChatSetting["model"],
                        )
                    except Exception as e:
                        print(f"Memory extraction failed: {e}")
```

这段在保存当前用户消息之前运行；它读取的是上一轮为止已经入库的消息。提取失败只打印，继续聊天。

### Step 7: 两次模型调用都换成临时副本

第一次调用（约 554 行）：

```python
                            messages=st.session_state.FreeChatMessages,
```

改成：

```python
                            messages=build_api_messages(prompt),
```

第二次调用（约 632 行）同样改成：

```python
                                messages=build_api_messages(prompt),
```

第二次调用发生在 tool 消息已经 append 到 `FreeChatMessages` 之后；helper 重新复制，所以 tool 消息也在里面。

### Step 8: 写冒烟测试

新建 `test/test_freechat_memory_ui.py`：

```python
"""Smoke test for FreeChat. Run from the repo root:
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parent.parent / "pages" / "10_🎙️_FreeChat.py"

at = AppTest.from_file(str(PAGE), default_timeout=60).run()
assert not at.exception, at.exception
print("FreeChat smoke OK")
```

### Step 9: 验证

```bash
.venv/bin/python -m py_compile "pages/10_🎙️_FreeChat.py" src/memory.py
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
```

Expected: 编译没有输出；冒烟输出 `FreeChat smoke OK`。

手动检查：

```bash
streamlit run home_page.py
```

打开 FreeChat，选一个已有话题，发一句话，确认回复正常；用 sqlite 看表存在：

```bash
sqlite3 setting/chat_history_nangua.db ".tables"
```

Expected: 输出里包含 `memories` 和 `memory_runs`。注意：这会在真实聊天库里建空表，是预期行为。

### Step 10: Checkpoint

```bash
git status --short
```

不要提交。

---

## Task 5: 手动“记住这段” + 侧边栏故事书/家长视图

**Files:**
- Modify: `pages/10_🎙️_FreeChat.py:392-394`（在 Topics expander 后插入视图）
- Modify: `pages/10_🎙️_FreeChat.py:528`（在聊天输入前插入按钮）
- Modify: `test/test_freechat_memory_ui.py`

### Step 1: 先让冒烟测试失败

把 `test/test_freechat_memory_ui.py` 改成：

```python
"""Smoke test for FreeChat memory UI. Run from the repo root:
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parent.parent / "pages" / "10_🎙️_FreeChat.py"

at = AppTest.from_file(str(PAGE), default_timeout=60).run()
assert not at.exception, at.exception
labels = [getattr(expander, "label", "") for expander in at.sidebar.expander]
assert any("小南瓜记得的事" in label for label in labels), labels
print("FreeChat memory UI smoke OK")
```

运行：

```bash
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
```

Expected: `AssertionError`，labels 里没有“小南瓜记得的事”。

### Step 2: 加手动记住按钮

在 `pages/10_🎙️_FreeChat.py` 的 `if st.session_state.current_topic_id:` 主区域里，`# Accept user input` 前加：

```python
        if st.button(
            "📝 让小南瓜记住这段",
            help="把小南瓜刚才和你说的话记进故事书、生活、心情或学习里。",
        ):
            try:
                with st.spinner("小南瓜在记..."):
                    result = memory.maybe_extract_memories(
                        CHAT_HISTORY_PATH,
                        st.session_state.current_topic_id,
                        st.session_state.FreeChatChain,
                        st.session_state.FreeChatSetting["model"],
                        force=True,
                        window=20,
                    )
            except Exception as e:
                print(f"Memory extraction failed: {e}")
                result = {"error": str(e)}
            if result.get("extracted"):
                st.toast(f"记住啦，{result['extracted']} 条。")
            elif result.get("error"):
                st.toast("这次没记住，等一下再试。")
            else:
                st.toast("这段已经记过了。")
```

### Step 3: 加侧边栏记忆视图

在 Topics expander 结束后（约 392 行）、`st.subheader("2. STT")` 前，插入：

```python
        with st.expander("📖 小南瓜记得的事"):
            memories = memory.list_memories(CHAT_HISTORY_PATH)
            counts = memory.memory_counts(CHAT_HISTORY_PATH)
            if not memories:
                st.caption("还没有记忆。聊满 10 轮，或点“让小南瓜记住这段”。")
            else:
                st.caption(
                    f"共 {counts['total']} 条 · 本周学习 {counts['learning_week']} 条 · "
                    f"需要关注 {counts['attention']} 条 · 心情 30 天后淡出"
                )
                for kind in ("story", "learning", "life", "mood"):
                    kind_items = [m for m in memories if m["kind"] == kind]
                    if not kind_items:
                        continue
                    st.markdown(f"**{memory.KIND_LABELS[kind]}**")
                    for item in kind_items:
                        flag = f" ⚠️ {item['flag']}" if item.get("flag") else ""
                        st.markdown(f"**{item['key']}**：{item['content']}{flag}")
                        with st.popover(
                            "编辑 / 忘掉", key=f"mem_popover_{item['id']}"
                        ):
                            new_content = st.text_area(
                                "内容",
                                item["content"],
                                key=f"mem_text_{item['id']}",
                            )
                            if st.button("保存", key=f"mem_save_{item['id']}"):
                                memory.update_memory_content(
                                    CHAT_HISTORY_PATH, item["id"], new_content
                                )
                                st.rerun()
                            if st.button("忘掉", key=f"mem_forget_{item['id']}"):
                                memory.delete_memory(CHAT_HISTORY_PATH, item["id"])
                                st.rerun()
                with st.form("clear_memory_form"):
                    confirmed = st.checkbox("我确认要清空全部记忆")
                    if st.form_submit_button("清空全部记忆") and confirmed:
                        memory.clear_memories(CHAT_HISTORY_PATH)
                        st.rerun()
```

视图对四类记忆都可见；没有 PIN。孩子可以编辑、忘掉、清空；家长也能看到“需要关注”的标记。

### Step 4: 运行冒烟测试

```bash
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
```

Expected: `FreeChat memory UI smoke OK`。

### Step 5: 编译检查

```bash
.venv/bin/python -m py_compile "pages/10_🎙️_FreeChat.py" src/memory.py
```

Expected: 没有输出。

### Step 6: 手动验证

1. 打开 FreeChat，进入一个话题。
2. 跟小南瓜说一个带新名字的短故事，例如“阿黄是一只开飞船的狗”。
3. 点“📝 让小南瓜记住这段”，等 toast。
4. 打开侧边栏“📖 小南瓜记得的事”，确认故事书里出现“阿黄”。
5. 点“编辑 / 忘掉”，改内容、保存、刷新确认；再点“忘掉”，确认消失。
6. 点“清空全部记忆”（勾选确认），确认列表为空。

### Step 7: Checkpoint

```bash
git status --short
```

不要提交。

---

## Task 6: 对话方式自动路由和记忆使用规则

**Files:**
- Modify: `src/ClsChatBot.py:65-74`

### Step 1: 先确认提示词里没有这两节

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from src.ClsChatBot import ChatRobotOpenRouter
prompt = ChatRobotOpenRouter().select_chat_mode("西瓜一家-小南瓜")
assert "## 对话方式" in prompt
assert "## 记忆" in prompt
print("prompt OK")
PY
```

Expected: `AssertionError`。

### Step 2: 插入提示词

在 `src/ClsChatBot.py` 的 `elif mode == "西瓜一家-小南瓜":` 的 prompt 里，找到：

```text
        请你扮演小南瓜，使用三年级的小西瓜能听懂的语言来进行所有对话吧。你的回答要详略得当，避免在不重要的部分说得太长。请不要回复网址链接。
```

在这行后面、`你是小西瓜的姐姐兼 AI 指导` 前，插入（保持和周围一样的 8 空格缩进）：

```text
        ## 对话方式
        每次回应前，先判断小西瓜这句话属于哪一种，然后用对应方式回应，不要混在一起：
        - 聊天/心情：先听懂他的感受，简短回应，不急着讲道理；如果他只是想聊天，不要切换到教学模式。
        - 故事：接着故事书往下讲，可以讲得长；用生动的对话和细节；结尾可以问一个开放的问题。
        - 学习：一次只教一步，用苏格拉底式提问，先问清楚他卡在哪里，不直接给答案。
        - 画画：调用画图工具，先用一句话回应他的画或想法，再生成图片。

        ## 记忆
        你会收到“小南瓜记得的事”。只在和当前话题相关时自然使用；不要逐条复述；不要说“根据我的记忆”；每条回复最多提一次；不确定的记忆不要编。故事书、学习、生活可以在相关时自然提起；心情只在孩子主动提起或明显相关时使用，不要主动翻旧账。
```

### Step 3: 验证提示词

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from src.ClsChatBot import ChatRobotOpenRouter
prompt = ChatRobotOpenRouter().select_chat_mode("西瓜一家-小南瓜")
assert "## 对话方式" in prompt
assert "## 记忆" in prompt
print("prompt OK")
PY
```

Expected: `prompt OK`。

### Step 4: 手动检查

在 FreeChat 里分别试：

- 讲故事：“给我讲阿黄的新冒险。”——应该接着故事讲，可以长。
- 学习：“这道分数题我不会。”——应该先问卡在哪，一次一步。
- 心情：“今天有点难过。”——应该先听感受，不急着讲课。
- 画画：“画一只戴帽子的猫。”——应该调用画图工具。

这是主观检查，不需要每条都完美；确认路由存在、没有把学习模式套到闲聊上。

### Step 5: Checkpoint

```bash
git status --short
```

不要提交。

---

## Task 7: 旧话题回填脚本

**Files:**
- Create: `test/backfill_nangua_memory.py`

脚本默认 dry-run：只读聊天库，生成带个人内容的草稿文件，不写 `memories` 表。`--apply` 才写库。

### Step 1: 写脚本

新建 `test/backfill_nangua_memory.py`：

```python
"""Backfill long-term memory from existing 小南瓜 topics.

Dry run (default): writes setting/memory_backfill_draft.json and .md, no DB writes.
Apply: --apply imports the draft into the DB given by --db.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py --list
    PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py --topics 2,5
    PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py --apply
"""
import argparse
import json
import sqlite3
from datetime import datetime, timedelta

from src import memory
from src.ClsChatBot import ChatRobotOpenRouter

DB_PATH = "./setting/chat_history_nangua.db"
OUT_JSON = "./setting/memory_backfill_draft.json"
OUT_MD = "./setting/memory_backfill_draft.md"


def topic_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT t.id, t.name, t.created_at,
                      COUNT(c.id) AS messages,
                      MAX(c.created_at) AS last_at
               FROM topics t LEFT JOIN chats c ON c.topic_id = t.id
               GROUP BY t.id ORDER BY t.created_at"""
        ).fetchall()
    finally:
        conn.close()


def print_topics(rows):
    print("id | messages | last_at | name")
    for row in rows:
        print(f"{row['id']:>2} | {row['messages']:>8} | {row['last_at'] or '-'} | {row['name']}")


def write_draft(topic_names, items):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    lines = ["# 小南瓜记忆回填草稿", "", "请先检查、删掉测试话题，再运行 `--apply`。", ""]
    for topic_id, topic_name in topic_names.items():
        topic_items = [i for i in items if i["source_topic_id"] == topic_id]
        if not topic_items:
            continue
        lines.append(f"## 话题 {topic_id}：{topic_name}")
        for kind in memory.KINDS:
            kind_items = [i for i in topic_items if i["kind"] == kind]
            if not kind_items:
                continue
            lines.append(f"### {memory.KIND_LABELS[kind]}")
            for item in kind_items:
                flag = " ⚠️ 需要关注" if item.get("flag") else ""
                lines.append(f"- **{item['key']}**：{item['content']}{flag}")
        lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--topics", help="逗号分隔的话题 id，例如 2,5")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    args = parser.parse_args()

    rows = topic_rows(args.db)
    if args.list or (not args.topics and not args.all and not args.apply):
        print_topics(rows)
        print("\n用 --topics 2,5 或 --all 生成草稿；--apply 把草稿写进数据库。")
        return

    if args.apply:
        with open(OUT_JSON, encoding="utf-8") as f:
            items = json.load(f)
        count = memory.apply_memories(args.db, items)
        print(f"已写入 {count} 条记忆。")
        return

    topic_names = {row["id"]: row["name"] for row in rows}
    topic_last_at = {row["id"]: row["last_at"] for row in rows}
    if args.topics:
        selected_ids = [int(x) for x in args.topics.split(",") if x.strip()]
    else:
        selected_ids = [row["id"] for row in rows]

    chatbot = ChatRobotOpenRouter()
    chatbot.setup_env()
    client = chatbot.initial_llm()
    all_items = []
    for topic_id in selected_ids:
        if topic_id not in topic_names:
            print(f"跳过不存在的话题 {topic_id}")
            continue
        print(f"正在整理话题 {topic_id}：{topic_names[topic_id]} ...")
        items = memory.extract_topic_memories(args.db, topic_id, client, args.model)
        last_at = topic_last_at.get(topic_id)
        for item in items:
            item["source_topic_id"] = topic_id
            if item["kind"] == "mood" and last_at:
                # 旧心情按话题结束日期淡出，不在今天突然复活。
                expiry = datetime.fromisoformat(last_at) + timedelta(
                    days=memory.MOOD_TTL_DAYS
                )
                item["expires_at"] = expiry.strftime("%Y-%m-%d %H:%M:%S")
        all_items.extend(items)
        print(f"  提取 {len(items)} 条")

    write_draft(topic_names, all_items)
    print(
        f"草稿已写入 {OUT_JSON} 和 {OUT_MD}，共 {len(all_items)} 条。"
        "检查后再运行 --apply。"
    )


if __name__ == "__main__":
    main()
```

### Step 2: 列出话题（不调用模型，不写库）

```bash
PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py --list
```

Expected: 打印 `id | messages | last_at | name` 和 16 个话题。确认哪些是测试话题，记下要回填的 id。

### Step 3: 先生成一个话题的草稿

选一个小的、非测试话题：

```bash
PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py --topics 2
```

Expected: 调用模型几次，输出“提取 N 条”，然后写入：

- `setting/memory_backfill_draft.json`
- `setting/memory_backfill_draft.md`

打开 Markdown 检查：四类是否合理，有没有把测试内容、敏感信息或明显错的内容记进来。这两份草稿含孩子的私人聊天，不要提交进 git。

### Step 4: 在数据库副本上试 apply

先备份，再在副本上应用：

```bash
cp setting/chat_history_nangua.db /tmp/nangua-backfill-test.db
PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py \
  --db /tmp/nangua-backfill-test.db --apply
```

Expected: `已写入 N 条记忆。` 用 sqlite 检查副本：

```bash
sqlite3 /tmp/nangua-backfill-test.db \
  "SELECT kind, key, substr(content,1,40) FROM memories ORDER BY kind LIMIT 20;"
```

确认没问题后，再对真实库执行：

```bash
cp setting/chat_history_nangua.db setting/chat_history_nangua.db.bak
PYTHONPATH=. .venv/bin/python test/backfill_nangua_memory.py --apply
```

`setting/chat_history_nangua.db.bak` 含私人数据，不要提交；确认无误后可以删掉。

### Step 5: Checkpoint

```bash
git status --short
```

Expected: 新增 `test/backfill_nangua_memory.py`、草稿文件、可能的 `.bak`；真实 DB 有修改。不要提交，尤其不要 `git add` 草稿或备份。

---

## Task 8: 全部验证

### 自动化检查

```bash
PYTHONPATH=. .venv/bin/python test/test_memory.py
PYTHONPATH=. .venv/bin/python test/test_freechat_memory_ui.py
.venv/bin/python -m py_compile \
  "pages/10_🎙️_FreeChat.py" src/memory.py src/ClsChatBot.py \
  test/backfill_nangua_memory.py test/test_memory.py test/test_freechat_memory_ui.py
```

Expected:

```text
memory tests OK
FreeChat memory UI smoke OK
```

编译没有输出。

### 手动端到端

1. 启动 `streamlit run home_page.py`，打开 FreeChat。
2. 新建或进入一个话题，聊一个带新名字的故事。
3. 点“📝 让小南瓜记住这段”，侧边栏出现“故事书”记忆。
4. 刷新页面，进入另一个话题，问“你还记得阿黄吗？”——回复应自然用上，不说“根据我的记忆”。
5. 回到记忆视图，编辑一条、忘掉一条、清空全部，各确认一次。
6. 测自动提取：临时把 `src/memory.py` 的 `AUTO_EXTRACT_EVERY` 改成 `1`，**重启** `streamlit run`，聊两轮，看是否自动出现记忆；确认后改回 `10` 并再重启。
7. 心情淡出：在 sqlite 里把某条 mood 的 `expires_at` 改成过去时间，刷新页面，确认列表和注入里都没有它（`include_expired=False`）。
8. 回填：`--list`、单个话题 dry-run、检查草稿、副本 `--apply`。

### 最终 diff 检查

```bash
git diff --stat
git status --short
```

确认只改了这些（`config.json`/`key.txt` 是 Jerry 自己的改动，不算这次）：

- `src/memory.py`（新）
- `test/test_memory.py`（新）
- `test/test_freechat_memory_ui.py`（新）
- `test/backfill_nangua_memory.py`（新）
- `pages/10_🎙️_FreeChat.py`
- `src/ClsChatBot.py`
- `setting/chat_history_nangua.db`（运行时新增表；`*.db` 已被忽略，不会出现在 `git status`）
- `.gitignore`（新增 `setting/memory_backfill_draft.*`，避免含孩子聊天内容的草稿被误提交）
- `docs/plans/2026-10-29-nangua-memory.md`（计划）

不要提交；等 Jerry 明确要求。

---

## 回滚

如果记忆功能出问题，聊天本身不受影响。回滚步骤：

1. 从 `pages/10_🎙️_FreeChat.py` 移除 `from src import memory`、`memory.*` 调用和侧边栏视图。
2. 从 `src/ClsChatBot.py` 移除新增的 `## 对话方式` / `## 记忆` 两节。
3. 需要的话，在 sqlite 里删表：

```sql
DROP TABLE IF EXISTS memories;
DROP TABLE IF EXISTS memory_runs;
```

`chats` / `topics` 表不动；聊天记录不会被删。

---

## 已知的取舍

- **提取用当前聊天模型**：不新增模型配置。每 10 轮多一次非流式调用。如果以后觉得贵，再把提取模型固定成 `deepseek/deepseek-v4.1-flash-1017`。
- **切换话题只处理最近 40 条新消息**：正常情况下每次自动提取后只剩不到 20 条，够用；如果提取连续失败很久，更早的消息会被跳过。可以以后加分块补提取。
- **记忆视图没有 PIN**：按 Jerry 的决定，信任孩子。所有记忆都可看、可改、可忘、可清空。
- **回填草稿是本地文件**：含孩子的私人聊天，不提交、不发给别人；家长检查后再 apply。
- **家长周报没有单独页面**：侧边栏视图就是家长视图，里面有“本周学习 N 条”和“需要关注 N 条”的计数；真正的周报页面以后再做。
- **TTS 边生成边播不在本计划**：先把长期记忆跑稳，再单独做流式语音。

---

## 执行方式

计划已保存到 `docs/plans/2026-10-29-nangua-memory.md`。默认建议：**本会话逐任务实现**——我按 Task 1–8 做，每个任务跑对应验证，你随时看 diff 和结果。也可以另开一个会话，让它按这份计划从 Task 1 开始做。

---

## 执行记录（2026-10-29）

已完成 Task 1–7，Task 8 自动检查通过：

- `test/test_memory.py`：10 项全过。
- `test/test_freechat_memory_ui.py`：通过。
- 假客户端端到端：记忆注入、自动提取、游标推进、记忆不写回聊天历史，均通过。
- `py_compile`：通过。
- 真实库：`memories=6`（生活 3、学习 2、心情 1，其中心情已过期不显示）、`memory_runs=16`；测试数据已清理。
- 回填：已把话题 18 的 6 条写入 `setting/chat_history_nangua.db`；写入前备份为 `setting/chat_history_nangua.backup.db`（`integrity_check` 为 `ok`）。当前可见 5 条，注入内容约 309 字。

未做 / 待 Jerry：

- 真实模型聊天体验（故事、数学、心情路由）未手动跑。
- 未提交 git。

