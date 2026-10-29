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
