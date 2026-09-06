import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from app import audit_log, repository
from app.db import get_connection
from app.fallback import AllProvidersFailedError, FallbackExtractor
from app.models import AuditLogEntry, Ticket
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_provider import GroqProvider
from app.service import TicketNotFoundError, TicketOwnershipError, TicketService
from app.state_machine import InvalidTransitionError

load_dotenv()

app = FastAPI(title="Policy-Gated Support Agent")


def _build_service() -> TicketService:
    db_path = os.environ.get("DB_PATH", "tickets.db")
    conn = get_connection(db_path, check_same_thread=False)
    repository.init_schema(conn)
    audit_log.init_schema(conn)

    providers = [GroqProvider(api_key=os.environ["GROQ_API_KEY"])]
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and gemini_key != "filler-not-set-yet":
        providers.append(GeminiProvider(api_key=gemini_key))

    return TicketService(conn=conn, extractor=FallbackExtractor(providers=providers))


@lru_cache
def get_service() -> TicketService:
    return _build_service()


class SubmitTicketRequest(BaseModel):
    rider_id: str
    message: str


class AddMessageRequest(BaseModel):
    rider_id: str
    message: str


@app.post("/tickets", response_model=Ticket)
def submit_ticket(payload: SubmitTicketRequest, service: TicketService = Depends(get_service)):
    try:
        return service.submit_ticket(payload.rider_id, payload.message)
    except AllProvidersFailedError:
        raise HTTPException(status_code=503, detail="All extraction providers are currently unavailable.")


@app.post("/tickets/{ticket_id}/messages", response_model=Ticket)
def add_message(
    ticket_id: str, payload: AddMessageRequest, service: TicketService = Depends(get_service)
):
    try:
        return service.add_message(ticket_id, payload.rider_id, payload.message)
    except (TicketNotFoundError, TicketOwnershipError):
        # Same 404 for "doesn't exist" and "wrong rider" so a caller can't
        # distinguish "no such ticket" from "not yours" by probing.
        raise HTTPException(status_code=404, detail="Ticket not found.")
    except AllProvidersFailedError:
        raise HTTPException(status_code=503, detail="All extraction providers are currently unavailable.")


@app.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: str, service: TicketService = Depends(get_service)):
    ticket = repository.get_ticket(service.conn, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket


@app.get("/tickets/{ticket_id}/audit-log", response_model=list[AuditLogEntry])
def get_audit_log(ticket_id: str, service: TicketService = Depends(get_service)):
    ticket = repository.get_ticket(service.conn, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return audit_log.get_for_ticket(service.conn, ticket_id)


@app.post("/tickets/{ticket_id}/resolve", response_model=Ticket)
def resolve_ticket(ticket_id: str, service: TicketService = Depends(get_service)):
    try:
        return service.resolve_ticket(ticket_id)
    except TicketNotFoundError:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    except InvalidTransitionError:
        raise HTTPException(status_code=409, detail="Ticket is not in a resolvable state.")
