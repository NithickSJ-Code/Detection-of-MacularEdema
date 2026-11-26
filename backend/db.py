"""SQLite persistence. Tables are created on startup. Parameterised queries only."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from backend.settings import DB_PATH, STORAGE_DIR


def _connect(path: Optional[Path] = None) -> sqlite3.Connection:
    db_path = path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Optional[Path] = None) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    (STORAGE_DIR / "scans").mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                scan_id TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                rejection_reason TEXT,
                predicted_label TEXT,
                confidence REAL,
                probs_json TEXT,
                uncertainty_level TEXT,
                uncertainty_std REAL,
                uncertainty_entropy REAL,
                mc_passes INTEGER,
                review_recommended INTEGER,
                review_reasons_json TEXT,
                box_json TEXT,
                source TEXT,
                report_path TEXT,
                processing_ms INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);
            CREATE INDEX IF NOT EXISTS idx_scans_predicted_label ON scans(predicted_label);
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_conn(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)
