"""Long-term memory for 小南瓜.

Pure Python + SQLite. No Streamlit imports, so the same module can be tested
headlessly and reused by a future UI. The caller passes an OpenAI-compatible
client and model name.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

KINDS = ("story", "life", "mood", "learning")
KIND_LABELS = {"story": "故事书", "life": "生活", "mood": "心情", "learning": "学习"}
MOOD_TTL_DAYS = 30
AUTO_EXTRACT_EVERY = 10
MAX_KEY_CHARS = 40
MAX_CONTENT_CHARS = 300
MAX_NEW_MESSAGES = 80
MAX_ITEMS_PER_KIND = 5
EXISTING_MEMORY_CHARS = 1800
MAX_CONTEXT_CHARS = 2000
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
        selected = rows[-max(1, window) :] if force else rows
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
