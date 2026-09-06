from datetime import datetime, timedelta, timezone

import pytest

from app import audit_log
from app.fallback import FallbackExtractor
from app.models import ExtractionResult, TicketState
from app.service import TicketNotFoundError, TicketOwnershipError, TicketService


class ScriptedProvider:
    """A fake LLMProvider that returns extractions from a script, one per
    call, and records every message it was asked to extract from."""

    def __init__(self, name: str, script: list[ExtractionResult]):
        self.name = name
        self._script = list(script)
        self.calls: list[str] = []

    def extract(self, message: str) -> ExtractionResult:
        self.calls.append(message)
        return self._script.pop(0)


def make_extraction(**overrides) -> ExtractionResult:
    defaults = dict(category="other", urgency="low", sentiment="calm", summary="s", confidence=0.9)
    defaults.update(overrides)
    return ExtractionResult(**defaults)


class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def make_service(conn, script: list[ExtractionResult], clock: FakeClock | None = None) -> TicketService:
    clock = clock or FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    extractor = FallbackExtractor(providers=[ScriptedProvider("groq", script)])
    counter = iter(range(1, 1000))
    return TicketService(
        conn=conn,
        extractor=extractor,
        clock=clock,
        id_factory=lambda: f"ticket-{next(counter)}",
    )


class TestSubmitTicket:
    def test_safety_message_escalates_urgent(self, conn):
        service = make_service(conn, [make_extraction(category="safety")])
        ticket = service.submit_ticket("rider-1", "the customer threatened me")
        assert ticket.state == TicketState.ESCALATED_URGENT
        assert ticket.extraction.category == "safety"
        assert ticket.decision.rule_id == "SAFETY_ALWAYS_URGENT"

    def test_low_amount_payment_auto_resolves(self, conn):
        service = make_service(conn, [make_extraction(category="payment", amount_mentioned=10, confidence=0.9)])
        ticket = service.submit_ticket("rider-1", "never got my $10 tip")
        assert ticket.state == TicketState.AUTO_RESOLVED

    def test_ticket_and_audit_trail_are_persisted(self, conn):
        service = make_service(conn, [make_extraction(category="safety")])
        ticket = service.submit_ticket("rider-1", "help")
        entries = audit_log.get_for_ticket(conn, ticket.id)
        actors = [e.actor for e in entries]
        assert actors == ["llm_extraction", "policy_engine", "state_machine"]
        assert entries[0].provider == "groq"


class TestAddMessage:
    def test_followup_is_read_with_full_history(self, conn):
        provider = ScriptedProvider(
            "groq", [make_extraction(category="customer_dispute"), make_extraction(category="safety")]
        )
        extractor = FallbackExtractor(providers=[provider])
        service = TicketService(conn=conn, extractor=extractor, id_factory=lambda: "t1")

        ticket = service.submit_ticket("rider-1", "customer didn't pay")
        service.add_message(ticket.id, "rider-1", "he also threatened me")

        assert "customer didn't pay" in provider.calls[1]
        assert "he also threatened me" in provider.calls[1]

    def test_followup_escalates_the_case(self, conn):
        provider = ScriptedProvider(
            "groq", [make_extraction(category="customer_dispute"), make_extraction(category="safety")]
        )
        extractor = FallbackExtractor(providers=[provider])
        service = TicketService(conn=conn, extractor=extractor, id_factory=lambda: "t1")

        ticket = service.submit_ticket("rider-1", "customer didn't pay")
        assert ticket.state == TicketState.ESCALATED_ROUTINE

        updated = service.add_message(ticket.id, "rider-1", "he also threatened me")
        assert updated.state == TicketState.ESCALATED_URGENT

    def test_followup_cannot_deescalate_an_urgent_ticket(self, conn):
        provider = ScriptedProvider(
            "groq", [make_extraction(category="safety"), make_extraction(category="other", confidence=0.99)]
        )
        extractor = FallbackExtractor(providers=[provider])
        service = TicketService(conn=conn, extractor=extractor, id_factory=lambda: "t1")

        ticket = service.submit_ticket("rider-1", "customer punched me")
        assert ticket.state == TicketState.ESCALATED_URGENT

        updated = service.add_message(ticket.id, "rider-1", "never mind, it was nothing")
        assert updated.state == TicketState.ESCALATED_URGENT

    def test_followup_reopens_an_auto_resolved_ticket(self, conn):
        provider = ScriptedProvider(
            "groq",
            [
                make_extraction(category="payment", amount_mentioned=5, confidence=0.9),
                make_extraction(category="safety"),
            ],
        )
        extractor = FallbackExtractor(providers=[provider])
        service = TicketService(conn=conn, extractor=extractor, id_factory=lambda: "t1")

        ticket = service.submit_ticket("rider-1", "missing a small tip")
        assert ticket.state == TicketState.AUTO_RESOLVED

        updated = service.add_message(ticket.id, "rider-1", "actually the customer also threatened me")
        assert updated.state == TicketState.ESCALATED_URGENT

    def test_unknown_ticket_raises(self, conn):
        service = make_service(conn, [])
        with pytest.raises(TicketNotFoundError):
            service.add_message("does-not-exist", "rider-1", "hello")

    def test_wrong_rider_raises_ownership_error(self, conn):
        service = make_service(conn, [make_extraction(category="other")])
        ticket = service.submit_ticket("rider-1", "hello")
        with pytest.raises(TicketOwnershipError):
            service.add_message(ticket.id, "rider-2", "not my ticket")


class TestResolveTicket:
    def test_resolve_closes_an_escalated_ticket(self, conn):
        service = make_service(conn, [make_extraction(category="safety")])
        ticket = service.submit_ticket("rider-1", "help")
        assert ticket.state == TicketState.ESCALATED_URGENT

        resolved = service.resolve_ticket(ticket.id)
        assert resolved.state == TicketState.CLOSED

        entries = audit_log.get_for_ticket(conn, ticket.id)
        assert entries[-1].actor == "human"

    def test_resolve_unknown_ticket_raises(self, conn):
        service = make_service(conn, [])
        with pytest.raises(TicketNotFoundError):
            service.resolve_ticket("does-not-exist")


class TestRepeatComplainant:
    def test_third_ticket_in_24h_is_never_auto_resolved(self, conn):
        script = [make_extraction(category="payment", amount_mentioned=5, confidence=0.9) for _ in range(3)]
        clock = FakeClock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        service = make_service(conn, script, clock=clock)

        first = service.submit_ticket("rider-1", "missing tip 1")
        clock.advance(hours=1)
        second = service.submit_ticket("rider-1", "missing tip 2")
        clock.advance(hours=1)
        third = service.submit_ticket("rider-1", "missing tip 3")

        assert first.state == TicketState.AUTO_RESOLVED
        assert second.state == TicketState.AUTO_RESOLVED
        assert third.state == TicketState.ESCALATED_ROUTINE
        assert third.decision.rule_id == "REPEAT_COMPLAINANT_FLOOR"
