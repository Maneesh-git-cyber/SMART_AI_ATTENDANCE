"""
database_manager.py
Handles all SQLite database operations for the attendance system.
"""

import sqlite3
import os
import json
from datetime import datetime, date
from typing import Optional, List, Tuple, Dict
import numpy as np


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "attendance.db")


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database():
    """Create all required tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    c = conn.cursor()

    # Persons table
    c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            department  TEXT DEFAULT '',
            email       TEXT DEFAULT '',
            phone       TEXT DEFAULT '',
            embedding   TEXT NOT NULL,          -- JSON array of 128-d FaceNet vector
            face_image  BLOB,                   -- thumbnail bytes
            registered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active   INTEGER DEFAULT 1
        )
    """)

    # Attendance records table
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id   INTEGER NOT NULL REFERENCES persons(id),
            date        TEXT NOT NULL,          -- YYYY-MM-DD
            check_in    DATETIME NOT NULL,
            check_out   DATETIME,
            status      TEXT DEFAULT 'present', -- present / late / absent
            confidence  REAL DEFAULT 0.0,
            method      TEXT DEFAULT 'live',    -- live / manual
            notes       TEXT DEFAULT '',
            UNIQUE(person_id, date)             -- one record per person per day
        )
    """)

    # Spoofing log
    c.execute("""
        CREATE TABLE IF NOT EXISTS spoof_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
            reason      TEXT,
            snapshot    BLOB
        )
    """)

    # System settings
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Default settings
    defaults = [
        ("similarity_threshold", "0.68"),
        ("late_cutoff_time",     "09:30"),
        ("work_start_time",      "09:00"),
        ("spoof_lbp_threshold",  "12.0"),
        ("spoof_blur_threshold", "80.0"),
        ("spoof_texture_threshold", "0.35"),
    ]
    c.executemany(
        "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", defaults
    )

    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


# ─── Person CRUD ────────────────────────────────────────────────────────────

def add_person(name: str, employee_id: str, embedding: np.ndarray,
               department: str = "", email: str = "", phone: str = "",
               face_image: Optional[bytes] = None) -> int:
    """Insert a new person and return their row id."""
    conn = get_connection()
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO persons (name, employee_id, department, email, phone, embedding, face_image)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, employee_id, department, email, phone,
              json.dumps(embedding.tolist()), face_image))
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()


def update_person_embedding(person_id: int, embedding: np.ndarray):
    """Update a person's face embedding (re-enrollment)."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE persons SET embedding=? WHERE id=?",
            (json.dumps(embedding.tolist()), person_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_all_persons() -> List[Dict]:
    """Return all active persons with their embeddings."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM persons WHERE is_active=1"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.array(json.loads(d["embedding"]), dtype=np.float32)
            result.append(d)
        return result
    finally:
        conn.close()


def get_person_by_employee_id(employee_id: str) -> Optional[Dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM persons WHERE employee_id=? AND is_active=1",
            (employee_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["embedding"] = np.array(json.loads(d["embedding"]), dtype=np.float32)
            return d
        return None
    finally:
        conn.close()


def delete_person(person_id: int):
    """Soft-delete a person."""
    conn = get_connection()
    try:
        conn.execute("UPDATE persons SET is_active=0 WHERE id=?", (person_id,))
        conn.commit()
    finally:
        conn.close()


# ─── Attendance CRUD ─────────────────────────────────────────────────────────

def mark_attendance(person_id: int, confidence: float,
                    today: Optional[str] = None) -> Tuple[bool, str]:
    """
    Mark attendance for today.
    Returns (success, message).
    Prevents duplicate entries for the same day.
    """
    today = today or date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id, check_in FROM attendance WHERE person_id=? AND date=?",
            (person_id, today)
        ).fetchone()

        if existing:
            return False, f"Already marked at {existing['check_in']}"

        # Determine status
        work_start = conn.execute(
            "SELECT value FROM settings WHERE key='work_start_time'"
        ).fetchone()["value"]
        late_cutoff = conn.execute(
            "SELECT value FROM settings WHERE key='late_cutoff_time'"
        ).fetchone()["value"]

        current_time = datetime.now().strftime("%H:%M")
        if current_time > late_cutoff:
            status = "late"
        else:
            status = "present"

        conn.execute("""
            INSERT INTO attendance (person_id, date, check_in, status, confidence, method)
            VALUES (?, ?, ?, ?, ?, 'live')
        """, (person_id, today, now, status, confidence))
        conn.commit()
        return True, f"Marked {status} at {now}"
    finally:
        conn.close()


def update_checkout(person_id: int, today: Optional[str] = None):
    """Update check-out time for the person's today record."""
    today = today or date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE attendance SET check_out=? WHERE person_id=? AND date=?",
            (now, person_id, today)
        )
        conn.commit()
    finally:
        conn.close()


def get_today_attendance() -> List[Dict]:
    today = date.today().isoformat()
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT a.*, p.name, p.employee_id, p.department
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date = ?
            ORDER BY a.check_in DESC
        """, (today,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_attendance_range(start_date: str, end_date: str) -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT a.*, p.name, p.employee_id, p.department
            FROM attendance a
            JOIN persons p ON a.person_id = p.id
            WHERE a.date BETWEEN ? AND ?
            ORDER BY a.date DESC, a.check_in DESC
        """, (start_date, end_date)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_spoof_attempt(reason: str, snapshot: Optional[bytes] = None):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO spoof_log (reason, snapshot) VALUES (?, ?)",
            (reason, snapshot)
        )
        conn.commit()
    finally:
        conn.close()


def get_setting(key: str) -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def update_setting(key: str, value: str):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        conn.commit()
    finally:
        conn.close()
