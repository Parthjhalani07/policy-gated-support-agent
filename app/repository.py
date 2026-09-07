import sqlite3
from datetime import datetime

from app.models import ExtractionResult, Message, PolicyDecision, Ticket, TicketState

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    rider_id TEXT NOT NULL,
    state TEXT NOT NULL,
    extraction TEXT,
    decision TEXT,
    created_at TEXT NOT NULL,
    triaged_at TEXT,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    ticket_id TEXT NOT NULL REFERENCES tickets(id),
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    PRIMARY KEY (ticket_id, seq)
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def save_ticket(conn: sqlite3.Connection, ticket: Ticket) -> None:
    conn.execute(
        """
        INSERT INTO tickets (id, rider_id, state, extraction, decision, created_at, triaged_at, closed_at)
        VALUES (:id, :rider_id, :state, :extraction, :decision, :created_at, :triaged_at, :closed_at)
        ON CONFLICT(id) DO UPDATE SET
            state = excluded.state,
            extraction = excluded.extraction,
            decision = excluded.decision,
            triaged_at = excluded.triaged_at,
            closed_at = excluded.closed_at
        """,
        {
            "id": ticket.id,
            "rider_id": ticket.rider_id,
            "state": ticket.state.value,
            "extraction": ticket.extraction.model_dump_json() if ticket.extraction else None,
            "decision": ticket.decision.model_dump_json() if ticket.decision else None,
            "created_at": ticket.created_at.isoformat(),
            "triaged_at": ticket.triaged_at.isoformat() if ticket.triaged_at else None,
            "closed_at": ticket.closed_at.isoformat() if ticket.closed_at else None,
        },
    )
    for message in ticket.messages:
        conn.execute(
            """
            INSERT INTO messages (ticket_id, seq, text, submitted_at)
            VALUES (:ticket_id, :seq, :text, :submitted_at)
            ON CONFLICT(ticket_id, seq) DO UPDATE SET
                text = excluded.text,
                submitted_at = excluded.submitted_at
            """,
            {
                "ticket_id": message.ticket_id,
                "seq": message.seq,
                "text": message.text,
                "submitted_at": message.submitted_at.isoformat(),
            },
        )
    conn.commit()


def get_ticket(conn: sqlite3.Connection, ticket_id: str) -> Ticket | None:
    row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if row is None:
        return None

    message_rows = conn.execute(
        "SELECT * FROM messages WHERE ticket_id = ? ORDER BY seq", (ticket_id,)
    ).fetchall()

    return Ticket(
        id=row["id"],
        rider_id=row["rider_id"],
        state=TicketState(row["state"]),
        extraction=ExtractionResult.model_validate_json(row["extraction"]) if row["extraction"] else None,
        decision=PolicyDecision.model_validate_json(row["decision"]) if row["decision"] else None,
        created_at=datetime.fromisoformat(row["created_at"]),
        triaged_at=datetime.fromisoformat(row["triaged_at"]) if row["triaged_at"] else None,
        closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
        messages=[
            Message(
                ticket_id=m["ticket_id"],
                seq=m["seq"],
                text=m["text"],
                submitted_at=datetime.fromisoformat(m["submitted_at"]),
            )
            for m in message_rows
        ],
    )


def list_tickets(conn: sqlite3.Connection, limit: int = 50) -> list[Ticket]:
    rows = conn.execute(
        "SELECT id FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [get_ticket(conn, row["id"]) for row in rows]


def count_recent_tickets(
    conn: sqlite3.Connection,
    rider_id: str,
    since: datetime,
    exclude_ticket_id: str | None = None,
) -> int:
    query = "SELECT COUNT(*) FROM tickets WHERE rider_id = ? AND created_at >= ?"
    params: list = [rider_id, since.isoformat()]
    if exclude_ticket_id is not None:
        query += " AND id != ?"
        params.append(exclude_ticket_id)
    row = conn.execute(query, params).fetchone()
    return row[0]
