"""Manual end-to-end demo of the ticket pipeline against the real Groq API.

Run: python -m scripts.demo
"""

import os

from dotenv import load_dotenv

from app import audit_log, repository
from app.db import get_connection
from app.fallback import FallbackExtractor
from app.providers.groq_provider import GroqProvider
from app.service import TicketService


def main():
    load_dotenv()
    conn = get_connection("demo.db")
    repository.init_schema(conn)
    audit_log.init_schema(conn)

    extractor = FallbackExtractor(providers=[GroqProvider(api_key=os.environ["GROQ_API_KEY"])])
    service = TicketService(conn=conn, extractor=extractor)

    print("--- Submitting first message ---")
    ticket = service.submit_ticket("rider-42", "The customer didn't pay the delivery fee.")
    print(f"Ticket {ticket.id}: state={ticket.state}, rule={ticket.decision.rule_id}")

    print("\n--- Adding a follow-up message ---")
    ticket = service.add_message(ticket.id, "rider-42", "He also grabbed my arm and threatened me.")
    print(f"Ticket {ticket.id}: state={ticket.state}, rule={ticket.decision.rule_id}")

    if ticket.state.value.startswith("ESCALATED"):
        print("\n--- Resolving as a human agent ---")
        ticket = service.resolve_ticket(ticket.id)
        print(f"Ticket {ticket.id}: state={ticket.state}")

    print("\n--- Full audit trail ---")
    for entry in audit_log.get_for_ticket(conn, ticket.id):
        print(f"[{entry.actor}] provider={entry.provider} output={entry.output}")


if __name__ == "__main__":
    main()
