"""
Persistent chat history using SQLite.
Works on Streamlit Cloud (writes to app's filesystem).
"""

import sqlite3
import uuid
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id     TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT 'New Chat',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
    """)
    conn.close()


# ── Chat CRUD ────────────────────────────────────────────────────────────────

def create_chat(title="New Chat") -> str:
    chat_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO chats (chat_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (chat_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return chat_id


def get_all_chats() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT chat_id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT chat_id, title, created_at, updated_at FROM chats WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_chat_title(chat_id: str, title: str):
    conn = _connect()
    conn.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))
    conn.commit()
    conn.close()


def delete_chat(chat_id: str):
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


# ── Messages CRUD ────────────────────────────────────────────────────────────

def append_message(chat_id: str, role: str, content: str):
    now = datetime.now().isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, now),
    )
    conn.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id))
    conn.commit()
    conn.close()


def get_messages(chat_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE chat_id = ? ORDER BY message_id ASC",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
