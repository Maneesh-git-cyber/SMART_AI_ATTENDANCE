"""
database.py — SQLite database manager for the Attendance System
Handles: persons, face embeddings, attendance records, sessions
"""

import sqlite3
import numpy as np
import json
import os
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "attendance.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            department  TEXT DEFAULT '',
            email       TEXT DEFAULT '',
            photo_path  TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now')),
            is_active   INTEGER DEFAULT 1
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
            embedding   BLOB NOT NULL,
            quality     REAL DEFAULT 0.0,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            date        TEXT NOT NULL,
            start_time  TEXT NOT NULL,
            end_time    TEXT,
            is_active   INTEGER DEFAULT 1,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL REFERENCES persons(id),
            session_id  INTEGER NOT NULL REFERENCES sessions(id),
            check_in    TEXT NOT NULL DEFAULT (datetime('now')),
            confidence  REAL DEFAULT 0.0,
            liveness_score REAL DEFAULT 0.0,
            status      TEXT DEFAULT 'present',
            UNIQUE(person_id, session_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS spoof_attempts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT DEFAULT (datetime('now')),
            reason      TEXT,
            snapshot_path TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


# ── Person CRUD ──────────────────────────────────────────────────────────────

def add_person(name: str, employee_id: str, department: str = "", email: str = "") -> int:
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO persons (name, employee_id, department, email) VALUES (?,?,?,?)",
            (name, employee_id, department, email)
        )
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()


def get_all_persons(active_only=True):
    conn = get_connection()
    try:
        q = "SELECT * FROM persons"
        if active_only:
            q += " WHERE is_active=1"
        q += " ORDER BY name"
        rows = conn.execute(q).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_person(person_id: int):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM persons WHERE id=?", (person_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_person_by_employee_id(emp_id: str):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM persons WHERE employee_id=?", (emp_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_person(person_id: int, **kwargs):
    allowed = {"name", "department", "email", "is_active", "photo_path"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    conn = get_connection()
    try:
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [person_id]
        conn.execute(f"UPDATE persons SET {sets} WHERE id=?", vals)
        conn.commit()
    finally:
        conn.close()


def delete_person(person_id: int):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
        conn.commit()
    finally:
        conn.close()


# ── Embeddings ────────────────────────────────────────────────────────────────

def save_embedding(person_id: int, embedding: np.ndarray, quality: float = 0.0):
    conn = get_connection()
    try:
        blob = embedding.astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO face_embeddings (person_id, embedding, quality) VALUES (?,?,?)",
            (person_id, blob, quality)
        )
        conn.commit()
    finally:
        conn.close()


def get_embeddings_for_person(person_id: int):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT embedding FROM face_embeddings WHERE person_id=?",
            (person_id,)
        ).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    finally:
        conn.close()


def load_all_embeddings():
    """Returns list of (person_id, embedding_array) for all active persons."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT fe.person_id, fe.embedding
            FROM face_embeddings fe
            JOIN persons p ON p.id = fe.person_id
            WHERE p.is_active = 1
        """).fetchall()
        result = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            result.append((r["person_id"], emb))
        return result
    finally:
        conn.close()


def count_embeddings(person_id: int) -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM face_embeddings WHERE person_id=?", (person_id,)
        ).fetchone()
        return row["cnt"]
    finally:
        conn.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(name: str) -> int:
    conn = get_connection()
    try:
        now = datetime.now()
        c = conn.cursor()
        # Deactivate any previous active session
        conn.execute("UPDATE sessions SET is_active=0 WHERE is_active=1")
        c.execute(
            "INSERT INTO sessions (name, date, start_time, is_active) VALUES (?,?,?,1)",
            (name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
        )
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()


def end_session(session_id: int):
    conn = get_connection()
    try:
        now = datetime.now().strftime("%H:%M:%S")
        conn.execute(
            "UPDATE sessions SET is_active=0, end_time=? WHERE id=?",
            (now, session_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_active_session():
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE is_active=1 LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_sessions():
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Attendance ────────────────────────────────────────────────────────────────

def mark_attendance(person_id: int, session_id: int, confidence: float, liveness_score: float) -> bool:
    """Returns True if newly marked, False if already marked."""
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM attendance WHERE person_id=? AND session_id=?",
            (person_id, session_id)
        ).fetchone()
        if existing:
            return False
        conn.execute(
            """INSERT INTO attendance (person_id, session_id, confidence, liveness_score)
               VALUES (?,?,?,?)""",
            (person_id, session_id, confidence, liveness_score)
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_attendance_for_session(session_id: int):
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT a.*, p.name, p.employee_id, p.department
            FROM attendance a
            JOIN persons p ON p.id = a.person_id
            WHERE a.session_id=?
            ORDER BY a.check_in
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_today_attendance():
    conn = get_connection()
    try:
        today = date.today().strftime("%Y-%m-%d")
        rows = conn.execute("""
            SELECT a.*, p.name, p.employee_id, p.department, s.name as session_name
            FROM attendance a
            JOIN persons p ON p.id = a.person_id
            JOIN sessions s ON s.id = a.session_id
            WHERE s.date=?
            ORDER BY a.check_in DESC
        """, (today,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def is_marked_today(person_id: int, session_id: int) -> bool:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM attendance WHERE person_id=? AND session_id=?",
            (person_id, session_id)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def log_spoof_attempt(reason: str, snapshot_path: str = ""):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO spoof_attempts (reason, snapshot_path) VALUES (?,?)",
            (reason, snapshot_path)
        )
        conn.commit()
    finally:
        conn.close()


def get_attendance_stats(session_id: int = None):
    conn = get_connection()
    try:
        total_persons = conn.execute("SELECT COUNT(*) as c FROM persons WHERE is_active=1").fetchone()["c"]
        if session_id:
            present = conn.execute(
                "SELECT COUNT(*) as c FROM attendance WHERE session_id=?", (session_id,)
            ).fetchone()["c"]
        else:
            sess = get_active_session()
            if sess:
                present = conn.execute(
                    "SELECT COUNT(*) as c FROM attendance WHERE session_id=?", (sess["id"],)
                ).fetchone()["c"]
            else:
                present = 0
        return {"total": total_persons, "present": present, "absent": total_persons - present}
    finally:
        conn.close()
