from datetime import datetime, timezone

from app.audit_log import append, get_for_ticket
from app.models import AuditLogEntry


def make_entry(ticket_id="t1", **overrides) -> AuditLogEntry:
    defaults = dict(
        ticket_id=ticket_id,
        timestamp=datetime.now(timezone.utc),
        actor="policy_engine",
        input={"category": "safety"},
        output={"action": "ESCALATED_URGENT"},
    )
    defaults.update(overrides)
    return AuditLogEntry(**defaults)


class TestAppendAndQuery:
    def test_append_then_get_returns_entry(self, conn):
        entry = make_entry()
        append(conn, entry)
        entries = get_for_ticket(conn, "t1")
        assert len(entries) == 1
        assert entries[0].actor == "policy_engine"
        assert entries[0].input == {"category": "safety"}
        assert entries[0].output == {"action": "ESCALATED_URGENT"}

    def test_entries_returned_in_insertion_order(self, conn):
        append(conn, make_entry(actor="llm_extraction"))
        append(conn, make_entry(actor="policy_engine"))
        append(conn, make_entry(actor="state_machine"))
        entries = get_for_ticket(conn, "t1")
        assert [e.actor for e in entries] == ["llm_extraction", "policy_engine", "state_machine"]

    def test_entries_scoped_to_ticket(self, conn):
        append(conn, make_entry(ticket_id="t1"))
        append(conn, make_entry(ticket_id="t2"))
        assert len(get_for_ticket(conn, "t1")) == 1
        assert len(get_for_ticket(conn, "t2")) == 1

    def test_no_entries_returns_empty_list(self, conn):
        assert get_for_ticket(conn, "nonexistent") == []

    def test_optional_fields_roundtrip(self, conn):
        entry = make_entry(provider="groq", latency_ms=120)
        append(conn, entry)
        fetched = get_for_ticket(conn, "t1")[0]
        assert fetched.provider == "groq"
        assert fetched.latency_ms == 120

    def test_missing_optional_fields_roundtrip_as_none(self, conn):
        entry = make_entry()
        append(conn, entry)
        fetched = get_for_ticket(conn, "t1")[0]
        assert fetched.provider is None
        assert fetched.latency_ms is None
