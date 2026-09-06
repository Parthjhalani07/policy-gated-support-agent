import json
import sqlite3
from datetime import datetime

from app.models import AuditLogEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    provider TEXT,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    latency_ms INTEGER
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def append(conn: sqlite3.Connection, entry: AuditLogEntry) -> None:
    conn.execute(
        """
        INSERT INTO audit_log (ticket_id, timestamp, actor, provider, input, output, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.ticket_id,
            entry.timestamp.isoformat(),
            entry.actor,
            entry.provider,
            json.dumps(entry.input),
            json.dumps(entry.output),
            entry.latency_ms,
        ),
    )
    conn.commit()


def get_for_ticket(conn: sqlite3.Connection, ticket_id: str) -> list[AuditLogEntry]:
    rows = conn.execute(
        "SELECT * FROM audit_log WHERE ticket_id = ? ORDER BY id", (ticket_id,)
    ).fetchall()
    return [
        AuditLogEntry(
            ticket_id=row["ticket_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            actor=row["actor"],
            provider=row["provider"],
            input=json.loads(row["input"]),
            output=json.loads(row["output"]),
            latency_ms=row["latency_ms"],
        )
        for row in rows
    ]
