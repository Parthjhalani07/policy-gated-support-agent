import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from app import audit_log, policy_engine, repository
from app.fallback import AllProvidersFailedError, FallbackExtractor
from app.models import AuditLogEntry, Message, Ticket, TicketState
from app.policy_engine import TicketContext
from app.state_machine import transition

REPEAT_COMPLAINANT_WINDOW = timedelta(hours=24)


class TicketNotFoundError(Exception):
    pass


class TicketOwnershipError(Exception):
    pass


def _resolve_target_state(current_state: TicketState, decision_action: str) -> TicketState:
    """Applies the escalation-only-up rule: an already-escalated ticket can
    only move to a more severe state or stay put, never auto-downgrade or
    auto-close. A ticket freshly at TRIAGED (first triage, or just reopened)
    takes the decision's action directly."""
    if current_state == TicketState.ESCALATED_URGENT:
        return TicketState.ESCALATED_URGENT
    if current_state == TicketState.ESCALATED_ROUTINE:
        if decision_action == "ESCALATED_URGENT":
            return TicketState.ESCALATED_URGENT
        return TicketState.ESCALATED_ROUTINE
    return TicketState[decision_action]


@dataclass
class TicketService:
    conn: sqlite3.Connection
    extractor: FallbackExtractor
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    id_factory: Callable[[], str] = field(default=lambda: str(uuid.uuid4()))

    def submit_ticket(self, rider_id: str, message_text: str) -> Ticket:
        now = self.clock()
        ticket_id = self.id_factory()
        message = Message(ticket_id=ticket_id, seq=1, text=message_text, submitted_at=now)
        ticket = Ticket(id=ticket_id, rider_id=rider_id, messages=[message], created_at=now)
        self._retriage(ticket)
        return ticket

    def add_message(self, ticket_id: str, rider_id: str, message_text: str) -> Ticket:
        ticket = repository.get_ticket(self.conn, ticket_id)
        if ticket is None:
            raise TicketNotFoundError(ticket_id)
        if ticket.rider_id != rider_id:
            raise TicketOwnershipError(ticket_id)

        next_seq = max((m.seq for m in ticket.messages), default=0) + 1
        ticket.messages.append(
            Message(ticket_id=ticket_id, seq=next_seq, text=message_text, submitted_at=self.clock())
        )

        if ticket.state in (TicketState.AUTO_RESOLVED, TicketState.CLOSED):
            transition(ticket, TicketState.TRIAGED)

        self._retriage(ticket)
        return ticket

    def resolve_ticket(self, ticket_id: str) -> Ticket:
        ticket = repository.get_ticket(self.conn, ticket_id)
        if ticket is None:
            raise TicketNotFoundError(ticket_id)

        prior_state = ticket.state
        transition(ticket, TicketState.CLOSED)
        audit_log.append(
            self.conn,
            AuditLogEntry(
                ticket_id=ticket.id,
                timestamp=self.clock(),
                actor="human",
                input={"action": "resolve", "from_state": prior_state.value},
                output={"to_state": ticket.state.value},
            ),
        )
        repository.save_ticket(self.conn, ticket)
        return ticket

    def _retriage(self, ticket: Ticket) -> None:
        if ticket.state == TicketState.SUBMITTED:
            transition(ticket, TicketState.TRIAGED)

        full_text = "\n".join(m.text for m in ticket.messages)

        extraction_started = self.clock()
        try:
            outcome = self.extractor.extract(full_text)
        except AllProvidersFailedError as exc:
            audit_log.append(
                self.conn,
                AuditLogEntry(
                    ticket_id=ticket.id,
                    timestamp=self.clock(),
                    actor="llm_extraction",
                    input={"message": full_text},
                    output={"error": str(exc)},
                ),
            )
            repository.save_ticket(self.conn, ticket)
            raise

        latency_ms = int((self.clock() - extraction_started).total_seconds() * 1000)
        ticket.extraction = outcome.result
        audit_log.append(
            self.conn,
            AuditLogEntry(
                ticket_id=ticket.id,
                timestamp=self.clock(),
                actor="llm_extraction",
                provider=outcome.provider_used,
                input={"message": full_text},
                output=outcome.result.model_dump(),
                latency_ms=latency_ms,
            ),
        )

        context = TicketContext(
            prior_tickets_last_24h=repository.count_recent_tickets(
                self.conn,
                ticket.rider_id,
                since=self.clock() - REPEAT_COMPLAINANT_WINDOW,
                exclude_ticket_id=ticket.id,
            )
        )
        decision = policy_engine.evaluate(outcome.result, context)
        ticket.decision = decision
        audit_log.append(
            self.conn,
            AuditLogEntry(
                ticket_id=ticket.id,
                timestamp=self.clock(),
                actor="policy_engine",
                input={
                    "extraction": outcome.result.model_dump(),
                    "prior_tickets_last_24h": context.prior_tickets_last_24h,
                },
                output=decision.model_dump(),
            ),
        )

        prior_state = ticket.state
        target_state = _resolve_target_state(prior_state, decision.action)
        if target_state != prior_state:
            transition(ticket, target_state)
        audit_log.append(
            self.conn,
            AuditLogEntry(
                ticket_id=ticket.id,
                timestamp=self.clock(),
                actor="state_machine",
                input={"from_state": prior_state.value, "decision_action": decision.action},
                output={"to_state": ticket.state.value, "changed": target_state != prior_state},
            ),
        )

        repository.save_ticket(self.conn, ticket)
