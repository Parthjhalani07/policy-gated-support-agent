from datetime import datetime, timezone

from app.models import Ticket, TicketState

TRANSITIONS: dict[TicketState, set[TicketState]] = {
    TicketState.SUBMITTED: {TicketState.TRIAGED},
    TicketState.TRIAGED: {
        TicketState.AUTO_RESOLVED,
        TicketState.ESCALATED_ROUTINE,
        TicketState.ESCALATED_URGENT,
    },
    TicketState.AUTO_RESOLVED: {TicketState.CLOSED, TicketState.TRIAGED},
    TicketState.ESCALATED_ROUTINE: {TicketState.ESCALATED_URGENT, TicketState.CLOSED},
    TicketState.ESCALATED_URGENT: {TicketState.CLOSED},
    TicketState.CLOSED: {TicketState.TRIAGED},
}


class InvalidTransitionError(Exception):
    def __init__(self, from_state: TicketState, to_state: TicketState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Cannot transition from {from_state} to {to_state}")


def transition(ticket: Ticket, to_state: TicketState) -> Ticket:
    if to_state not in TRANSITIONS[ticket.state]:
        raise InvalidTransitionError(ticket.state, to_state)

    ticket.state = to_state
    now = datetime.now(timezone.utc)
    if to_state == TicketState.TRIAGED:
        ticket.triaged_at = now
    elif to_state == TicketState.CLOSED:
        ticket.closed_at = now

    return ticket
