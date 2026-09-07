from dataclasses import dataclass

from app.models import ExtractionResult, PolicyDecision

PAYMENT_AUTO_THRESHOLD = 50.0
LOW_CONFIDENCE_THRESHOLD = 0.5
PAYMENT_AUTO_CONFIDENCE_THRESHOLD = 0.75
REPEAT_COMPLAINANT_THRESHOLD = 2


@dataclass
class TicketContext:
    prior_tickets_last_24h: int = 0


def _safety_always_urgent(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision | None:
    if extraction.category == "safety":
        return PolicyDecision(
            action="ESCALATED_URGENT",
            reason="Safety-related report; always escalated urgently regardless of confidence.",
            rule_id="SAFETY_ALWAYS_URGENT",
        )
    return None


def _low_confidence_escalate(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision | None:
    if extraction.confidence < LOW_CONFIDENCE_THRESHOLD:
        return PolicyDecision(
            action="ESCALATED_ROUTINE",
            reason=f"Extraction confidence {extraction.confidence:.2f} is below {LOW_CONFIDENCE_THRESHOLD}; too uncertain to auto-resolve.",
            rule_id="LOW_CONFIDENCE_ESCALATE",
        )
    return None


def _not_actionable_auto_resolve(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision | None:
    if not extraction.is_actionable:
        return PolicyDecision(
            action="AUTO_RESOLVED",
            reason="Message doesn't describe an actionable issue (e.g. a greeting or test message); closed without action.",
            rule_id="NOT_ACTIONABLE_AUTO_RESOLVE",
        )
    return None


def _repeat_complainant_floor(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision | None:
    if context.prior_tickets_last_24h >= REPEAT_COMPLAINANT_THRESHOLD:
        return PolicyDecision(
            action="ESCALATED_ROUTINE",
            reason=f"Rider has {context.prior_tickets_last_24h} prior tickets in the last 24h; never auto-resolved.",
            rule_id="REPEAT_COMPLAINANT_FLOOR",
        )
    return None


def _payment_low_amount_auto_resolve(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision | None:
    if (
        extraction.category == "payment"
        and extraction.amount_mentioned is not None
        and extraction.amount_mentioned < PAYMENT_AUTO_THRESHOLD
        and extraction.confidence > PAYMENT_AUTO_CONFIDENCE_THRESHOLD
    ):
        return PolicyDecision(
            action="AUTO_RESOLVED",
            reason=(
                f"Payment dispute for {extraction.amount_mentioned} is below the "
                f"{PAYMENT_AUTO_THRESHOLD} auto-resolve threshold with high confidence."
            ),
            rule_id="PAYMENT_LOW_AMOUNT_AUTO_RESOLVE",
        )
    return None


def _vehicle_high_urgency(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision | None:
    if extraction.category == "vehicle" and extraction.urgency == "high":
        return PolicyDecision(
            action="ESCALATED_URGENT",
            reason="High-urgency vehicle issue (e.g. breakdown/stranded rider).",
            rule_id="VEHICLE_HIGH_URGENCY",
        )
    return None


def _default_escalate_routine(extraction: ExtractionResult, context: TicketContext) -> PolicyDecision:
    return PolicyDecision(
        action="ESCALATED_ROUTINE",
        reason="No auto-resolve rule matched; defaulting to routine human review.",
        rule_id="DEFAULT_ESCALATE_ROUTINE",
    )


RULES = [
    _safety_always_urgent,
    _low_confidence_escalate,
    _vehicle_high_urgency,
    _not_actionable_auto_resolve,
    _repeat_complainant_floor,
    _payment_low_amount_auto_resolve,
]


def evaluate(extraction: ExtractionResult, context: TicketContext | None = None) -> PolicyDecision:
    context = context or TicketContext()
    for rule in RULES:
        decision = rule(extraction, context)
        if decision is not None:
            return decision
    return _default_escalate_routine(extraction, context)
