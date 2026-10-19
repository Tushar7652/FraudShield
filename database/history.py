import json
import sqlite3
import streamlit as st
from config import DB_PATH


def init_history_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                record_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigation_reviews (
                transaction_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'Pending Review',
                notes TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def load_history_from_db():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT record_json FROM transaction_history ORDER BY id ASC"
        ).fetchall()
    history = []
    for (record_json,) in rows:
        try:
            history.append(json.loads(record_json))
        except (json.JSONDecodeError, TypeError):
            continue
    return history


def save_history(record):
    record_json = json.dumps(
        record,
        default=lambda value: value.item() if hasattr(value, "item") else str(value),
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO transaction_history (record_json) VALUES (?)",
            (record_json,),
        )
    st.session_state.setdefault("history", []).append(record)


def load_investigation_review(transaction_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT status, notes FROM investigation_reviews WHERE transaction_id = ?",
            (str(transaction_id),),
        ).fetchone()
    return row if row else ("Pending Review", "")


def save_investigation_review(transaction_id, status, notes):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO investigation_reviews (transaction_id, status, notes, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(transaction_id) DO UPDATE SET
                status = excluded.status,
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
        """, (str(transaction_id), status, notes))


def clear_saved_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM transaction_history")
