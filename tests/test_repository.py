from datetime import datetime, timedelta, timezone

from app.models import ExtractionResult, Message, PolicyDecision, Ticket, TicketState
from app.repository import count_recent_tickets, get_ticket, list_tickets, save_ticket


def make_ticket(ticket_id="t1", rider_id="r1", **overrides) -> Ticket:
    defaults = dict(
        id=ticket_id,
        rider_id=rider_id,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Ticket(**defaults)


class TestSaveAndGetTicket:
    def test_roundtrip_preserves_basic_fields(self, conn):
        ticket = make_ticket()
        save_ticket(conn, ticket)
        fetched = get_ticket(conn, ticket.id)
        assert fetched is not None
        assert fetched.id == ticket.id
        assert fetched.rider_id == ticket.rider_id
        assert fetched.state == TicketState.SUBMITTED

    def test_get_missing_ticket_returns_none(self, conn):
        assert get_ticket(conn, "does-not-exist") is None

    def test_roundtrip_preserves_messages_in_order(self, conn):
        ticket = make_ticket(
            messages=[
                Message(ticket_id="t1", seq=1, text="first", submitted_at=datetime.now(timezone.utc)),
                Message(ticket_id="t1", seq=2, text="second", submitted_at=datetime.now(timezone.utc)),
            ]
        )
        save_ticket(conn, ticket)
        fetched = get_ticket(conn, ticket.id)
        assert [m.text for m in fetched.messages] == ["first", "second"]

    def test_roundtrip_preserves_extraction_and_decision(self, conn):
        extraction = ExtractionResult(
            category="safety",
            urgency="high",
            sentiment="distressed",
            is_actionable=True,
            summary="s",
            confidence=0.9,
        )
        decision = PolicyDecision(action="ESCALATED_URGENT", reason="r", rule_id="SAFETY_ALWAYS_URGENT")
        ticket = make_ticket(extraction=extraction, decision=decision, state=TicketState.TRIAGED)
        save_ticket(conn, ticket)
        fetched = get_ticket(conn, ticket.id)
        assert fetched.extraction == extraction
        assert fetched.decision == decision

    def test_save_twice_updates_rather_than_duplicates(self, conn):
        ticket = make_ticket(state=TicketState.SUBMITTED)
        save_ticket(conn, ticket)
        ticket.state = TicketState.TRIAGED
        save_ticket(conn, ticket)
        fetched = get_ticket(conn, ticket.id)
        assert fetched.state == TicketState.TRIAGED
        row_count = conn.execute("SELECT COUNT(*) FROM tickets WHERE id = ?", (ticket.id,)).fetchone()[0]
        assert row_count == 1

    def test_appending_a_message_and_resaving_keeps_both(self, conn):
        ticket = make_ticket(
            messages=[Message(ticket_id="t1", seq=1, text="first", submitted_at=datetime.now(timezone.utc))]
        )
        save_ticket(conn, ticket)
        ticket.messages.append(
            Message(ticket_id="t1", seq=2, text="second", submitted_at=datetime.now(timezone.utc))
        )
        save_ticket(conn, ticket)
        fetched = get_ticket(conn, ticket.id)
        assert [m.text for m in fetched.messages] == ["first", "second"]


class TestListTickets:
    def test_returns_most_recent_first(self, conn):
        now = datetime.now(timezone.utc)
        save_ticket(conn, make_ticket(ticket_id="older", created_at=now - timedelta(hours=1)))
        save_ticket(conn, make_ticket(ticket_id="newer", created_at=now))
        tickets = list_tickets(conn)
        assert [t.id for t in tickets] == ["newer", "older"]

    def test_respects_limit(self, conn):
        now = datetime.now(timezone.utc)
        for i in range(5):
            save_ticket(conn, make_ticket(ticket_id=f"t{i}", created_at=now - timedelta(minutes=i)))
        tickets = list_tickets(conn, limit=2)
        assert len(tickets) == 2

    def test_empty_when_no_tickets(self, conn):
        assert list_tickets(conn) == []


class TestCountRecentTickets:
    def test_counts_tickets_within_window(self, conn):
        now = datetime.now(timezone.utc)
        save_ticket(conn, make_ticket(ticket_id="a", rider_id="r1", created_at=now))
        save_ticket(conn, make_ticket(ticket_id="b", rider_id="r1", created_at=now))
        count = count_recent_tickets(conn, "r1", since=now - timedelta(hours=24))
        assert count == 2

    def test_excludes_tickets_outside_window(self, conn):
        now = datetime.now(timezone.utc)
        save_ticket(conn, make_ticket(ticket_id="a", rider_id="r1", created_at=now - timedelta(hours=48)))
        count = count_recent_tickets(conn, "r1", since=now - timedelta(hours=24))
        assert count == 0

    def test_excludes_other_riders(self, conn):
        now = datetime.now(timezone.utc)
        save_ticket(conn, make_ticket(ticket_id="a", rider_id="other-rider", created_at=now))
        count = count_recent_tickets(conn, "r1", since=now - timedelta(hours=24))
        assert count == 0

    def test_can_exclude_current_ticket(self, conn):
        now = datetime.now(timezone.utc)
        save_ticket(conn, make_ticket(ticket_id="a", rider_id="r1", created_at=now))
        save_ticket(conn, make_ticket(ticket_id="b", rider_id="r1", created_at=now))
        count = count_recent_tickets(conn, "r1", since=now - timedelta(hours=24), exclude_ticket_id="a")
        assert count == 1
