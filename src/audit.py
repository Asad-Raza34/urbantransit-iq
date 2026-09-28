"""Audit store.

Records important analytical/administrative actions so there is an operational
timeline reviewable in the dashboard. Backed by SQLite (queryable) with a
JSON-lines companion (portable).
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.paths import DATA_METADATA
from src.config import settings

_DB = DATA_METADATA / "audit.db"
_JSONL = DATA_METADATA / "audit.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    action      TEXT NOT NULL,
    component   TEXT NOT NULL,
    status      TEXT NOT NULL,
    details     TEXT,
    version     TEXT,
    duration_ms REAL
);
"""


def _connect() -> sqlite3.Connection:
    DATA_METADATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB))
    conn.execute(_SCHEMA)
    return conn


def record(action: str, component: str, status: str = "ok", details: str = "",
           version: str = None, duration_ms: float = None) -> dict:
    """Append one audit entry and return it."""
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "component": component,
        "status": status,
        "details": details,
        "version": version,
        "duration_ms": duration_ms,
    }
    if duration_ms is not None:
        entry["duration_ms"] = round(duration_ms, 1)

    conn = _connect()
    conn.execute(
        """INSERT INTO audit_log (id, ts, action, component, status, details, version, duration_ms)
           VALUES (:id, :ts, :action, :component, :status, :details, :version, :duration_ms)""",
        entry,
    )
    conn.commit()
    conn.close()

    _JSONL.open("a", encoding="utf-8").write(json.dumps(entry) + "\n")
    return entry


def timeline(limit: int = 500, action: str = None, status: str = None,
             component: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Return audit entries newest-first with optional filters."""
    conn = _connect()
    sql = "SELECT * FROM audit_log"
    where, params = [], []
    if action:
        where.append("action = ?")
        params.append(action)
    if status:
        where.append("status = ?")
        params.append(status)
    if component:
        where.append("component = ?")
        params.append(component)
    if start_date:
        where.append("ts >= ?")
        params.append(start_date)
    if end_date:
        where.append("ts <= ?")
        params.append(end_date)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(int(limit))
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def summary() -> pd.DataFrame:
    """Counts by action+status for the executive dashboard."""
    df = timeline(limit=5000)
    if df.empty:
        return pd.DataFrame(columns=["action", "status", "count"])
    return df.groupby(["action", "status"], as_index=False).size().rename(columns={"size": "count"})


def summary_by_component() -> pd.DataFrame:
    """Counts by component for the system dashboard."""
    df = timeline(limit=5000)
    if df.empty:
        return pd.DataFrame(columns=["component", "status", "count"])
    return df.groupby(["component", "status"], as_index=False).size().rename(columns={"size": "count"})


def recent_activity(hours: int = 24) -> pd.DataFrame:
    """Get activity from the last N hours."""
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    return timeline(start_date=cutoff)


def app_version() -> str:
    return getattr(settings, "APP_VERSION", "1.0.0")


__all__ = ["record", "timeline", "summary", "summary_by_component", "recent_activity", "app_version"]