from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class TicketState(StrEnum):
    SUBMITTED = "SUBMITTED"
    TRIAGED = "TRIAGED"
    AUTO_RESOLVED = "AUTO_RESOLVED"
    ESCALATED_ROUTINE = "ESCALATED_ROUTINE"
    ESCALATED_URGENT = "ESCALATED_URGENT"
    CLOSED = "CLOSED"


class Message(BaseModel):
    ticket_id: str
    seq: int
    text: str
    submitted_at: datetime


class ExtractionResult(BaseModel):
    category: Literal["safety", "payment", "vehicle", "customer_dispute", "other"]
    urgency: Literal["low", "medium", "high"]
    sentiment: Literal["calm", "frustrated", "distressed"]
    summary: str
    confidence: float
    amount_mentioned: float | None = None


class PolicyDecision(BaseModel):
    action: Literal["AUTO_RESOLVED", "ESCALATED_URGENT", "ESCALATED_ROUTINE"]
    reason: str
    rule_id: str


class AuditLogEntry(BaseModel):
    ticket_id: str
    timestamp: datetime
    actor: Literal["llm_extraction", "policy_engine", "state_machine", "human"]
    input: dict
    output: dict
    provider: str | None = None
    latency_ms: int | None = None


class Ticket(BaseModel):
    id: str
    rider_id: str
    messages: list[Message] = []
    state: TicketState = TicketState.SUBMITTED
    extraction: ExtractionResult | None = None
    decision: PolicyDecision | None = None
    created_at: datetime
    triaged_at: datetime | None = None
    closed_at: datetime | None = None
