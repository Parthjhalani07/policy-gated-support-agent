from datetime import datetime, timezone

import pytest

from app.models import Ticket, TicketState
from app.state_machine import InvalidTransitionError, transition


def make_ticket(state: TicketState = TicketState.SUBMITTED) -> Ticket:
    return Ticket(
        id="t1",
        rider_id="r1",
        state=state,
        created_at=datetime.now(timezone.utc),
    )


class TestValidTransitions:
    def test_submitted_to_triaged(self):
        ticket = make_ticket(TicketState.SUBMITTED)
        transition(ticket, TicketState.TRIAGED)
        assert ticket.state == TicketState.TRIAGED
        assert ticket.triaged_at is not None

    @pytest.mark.parametrize(
        "to_state",
        [
            TicketState.AUTO_RESOLVED,
            TicketState.ESCALATED_ROUTINE,
            TicketState.ESCALATED_URGENT,
        ],
    )
    def test_triaged_to_outcome(self, to_state):
        ticket = make_ticket(TicketState.TRIAGED)
        transition(ticket, to_state)
        assert ticket.state == to_state

    def test_auto_resolved_to_closed(self):
        ticket = make_ticket(TicketState.AUTO_RESOLVED)
        transition(ticket, TicketState.CLOSED)
        assert ticket.state == TicketState.CLOSED
        assert ticket.closed_at is not None

    def test_auto_resolved_reopens_to_triaged(self):
        ticket = make_ticket(TicketState.AUTO_RESOLVED)
        transition(ticket, TicketState.TRIAGED)
        assert ticket.state == TicketState.TRIAGED

    def test_escalated_routine_upgrades_to_urgent(self):
        ticket = make_ticket(TicketState.ESCALATED_ROUTINE)
        transition(ticket, TicketState.ESCALATED_URGENT)
        assert ticket.state == TicketState.ESCALATED_URGENT

    def test_escalated_routine_closed_by_human(self):
        ticket = make_ticket(TicketState.ESCALATED_ROUTINE)
        transition(ticket, TicketState.CLOSED)
        assert ticket.state == TicketState.CLOSED

    def test_escalated_urgent_closed_by_human(self):
        ticket = make_ticket(TicketState.ESCALATED_URGENT)
        transition(ticket, TicketState.CLOSED)
        assert ticket.state == TicketState.CLOSED

    def test_closed_reopens_to_triaged(self):
        ticket = make_ticket(TicketState.CLOSED)
        transition(ticket, TicketState.TRIAGED)
        assert ticket.state == TicketState.TRIAGED


class TestInvalidTransitions:
    def test_submitted_cannot_skip_to_closed(self):
        ticket = make_ticket(TicketState.SUBMITTED)
        with pytest.raises(InvalidTransitionError):
            transition(ticket, TicketState.CLOSED)

    def test_escalated_urgent_cannot_downgrade_to_routine(self):
        ticket = make_ticket(TicketState.ESCALATED_URGENT)
        with pytest.raises(InvalidTransitionError):
            transition(ticket, TicketState.ESCALATED_ROUTINE)

    def test_escalated_urgent_cannot_auto_resolve(self):
        ticket = make_ticket(TicketState.ESCALATED_URGENT)
        with pytest.raises(InvalidTransitionError):
            transition(ticket, TicketState.AUTO_RESOLVED)

    def test_closed_has_no_transition_besides_triaged(self):
        ticket = make_ticket(TicketState.CLOSED)
        for to_state in (
            TicketState.SUBMITTED,
            TicketState.AUTO_RESOLVED,
            TicketState.ESCALATED_ROUTINE,
            TicketState.ESCALATED_URGENT,
            TicketState.CLOSED,
        ):
            with pytest.raises(InvalidTransitionError):
                transition(ticket, to_state)
