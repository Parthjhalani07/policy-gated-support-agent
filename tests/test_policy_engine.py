import pytest

from app.models import ExtractionResult
from app.policy_engine import (
    PAYMENT_AUTO_CONFIDENCE_THRESHOLD,
    PAYMENT_AUTO_THRESHOLD,
    TicketContext,
    evaluate,
)


def make_extraction(**overrides) -> ExtractionResult:
    defaults = dict(
        category="other",
        urgency="low",
        sentiment="calm",
        is_actionable=True,
        summary="test message",
        confidence=0.9,
    )
    defaults.update(overrides)
    return ExtractionResult(**defaults)


class TestSafetyRule:
    def test_safety_always_escalates_urgent(self):
        extraction = make_extraction(category="safety", confidence=0.9)
        decision = evaluate(extraction)
        assert decision.action == "ESCALATED_URGENT"
        assert decision.rule_id == "SAFETY_ALWAYS_URGENT"

    def test_safety_escalates_urgent_even_at_low_confidence(self):
        extraction = make_extraction(category="safety", confidence=0.1)
        decision = evaluate(extraction)
        assert decision.action == "ESCALATED_URGENT"
        assert decision.rule_id == "SAFETY_ALWAYS_URGENT"

    def test_safety_beats_repeat_complainant(self):
        extraction = make_extraction(category="safety", confidence=0.9)
        decision = evaluate(extraction, TicketContext(prior_tickets_last_24h=5))
        assert decision.rule_id == "SAFETY_ALWAYS_URGENT"


class TestLowConfidenceRule:
    def test_low_confidence_escalates_routine(self):
        extraction = make_extraction(category="other", confidence=0.2)
        decision = evaluate(extraction)
        assert decision.action == "ESCALATED_ROUTINE"
        assert decision.rule_id == "LOW_CONFIDENCE_ESCALATE"

    def test_low_confidence_blocks_payment_auto_resolve(self):
        extraction = make_extraction(category="payment", amount_mentioned=10, confidence=0.3)
        decision = evaluate(extraction)
        assert decision.rule_id == "LOW_CONFIDENCE_ESCALATE"

    def test_confidence_exactly_at_threshold_is_not_low(self):
        extraction = make_extraction(category="other", confidence=0.5)
        decision = evaluate(extraction)
        assert decision.rule_id != "LOW_CONFIDENCE_ESCALATE"


class TestRepeatComplainantRule:
    def test_repeat_complainant_blocks_auto_resolve(self):
        extraction = make_extraction(category="payment", amount_mentioned=10, confidence=0.9)
        decision = evaluate(extraction, TicketContext(prior_tickets_last_24h=2))
        assert decision.action == "ESCALATED_ROUTINE"
        assert decision.rule_id == "REPEAT_COMPLAINANT_FLOOR"

    def test_one_prior_ticket_does_not_trigger_floor(self):
        extraction = make_extraction(category="payment", amount_mentioned=10, confidence=0.9)
        decision = evaluate(extraction, TicketContext(prior_tickets_last_24h=1))
        assert decision.rule_id == "PAYMENT_LOW_AMOUNT_AUTO_RESOLVE"

    def test_repeat_complainant_does_not_downgrade_vehicle_urgent(self):
        extraction = make_extraction(category="vehicle", urgency="high", confidence=0.9)
        decision = evaluate(extraction, TicketContext(prior_tickets_last_24h=5))
        assert decision.action == "ESCALATED_URGENT"
        assert decision.rule_id == "VEHICLE_HIGH_URGENCY"


class TestPaymentAutoResolveRule:
    def test_low_amount_high_confidence_auto_resolves(self):
        extraction = make_extraction(category="payment", amount_mentioned=10, confidence=0.9)
        decision = evaluate(extraction)
        assert decision.action == "AUTO_RESOLVED"
        assert decision.rule_id == "PAYMENT_LOW_AMOUNT_AUTO_RESOLVE"

    def test_amount_exactly_at_threshold_does_not_auto_resolve(self):
        extraction = make_extraction(category="payment", amount_mentioned=PAYMENT_AUTO_THRESHOLD, confidence=0.9)
        decision = evaluate(extraction)
        assert decision.rule_id != "PAYMENT_LOW_AMOUNT_AUTO_RESOLVE"

    def test_confidence_exactly_at_threshold_does_not_auto_resolve(self):
        extraction = make_extraction(
            category="payment", amount_mentioned=10, confidence=PAYMENT_AUTO_CONFIDENCE_THRESHOLD
        )
        decision = evaluate(extraction)
        assert decision.rule_id != "PAYMENT_LOW_AMOUNT_AUTO_RESOLVE"

    def test_high_amount_does_not_auto_resolve(self):
        extraction = make_extraction(category="payment", amount_mentioned=500, confidence=0.9)
        decision = evaluate(extraction)
        assert decision.action == "ESCALATED_ROUTINE"
        assert decision.rule_id == "DEFAULT_ESCALATE_ROUTINE"

    def test_missing_amount_does_not_auto_resolve(self):
        extraction = make_extraction(category="payment", amount_mentioned=None, confidence=0.9)
        decision = evaluate(extraction)
        assert decision.rule_id != "PAYMENT_LOW_AMOUNT_AUTO_RESOLVE"


class TestVehicleUrgencyRule:
    def test_high_urgency_vehicle_escalates_urgent(self):
        extraction = make_extraction(category="vehicle", urgency="high", confidence=0.9)
        decision = evaluate(extraction)
        assert decision.action == "ESCALATED_URGENT"
        assert decision.rule_id == "VEHICLE_HIGH_URGENCY"

    def test_low_urgency_vehicle_does_not_escalate_urgent(self):
        extraction = make_extraction(category="vehicle", urgency="low", confidence=0.9)
        decision = evaluate(extraction)
        assert decision.rule_id == "DEFAULT_ESCALATE_ROUTINE"


class TestDefaultRule:
    def test_customer_dispute_defaults_to_routine(self):
        extraction = make_extraction(category="customer_dispute", confidence=0.9)
        decision = evaluate(extraction)
        assert decision.action == "ESCALATED_ROUTINE"
        assert decision.rule_id == "DEFAULT_ESCALATE_ROUTINE"

    def test_other_category_defaults_to_routine(self):
        extraction = make_extraction(category="other", confidence=0.9)
        decision = evaluate(extraction)
        assert decision.rule_id == "DEFAULT_ESCALATE_ROUTINE"


@pytest.mark.parametrize(
    "extraction_kwargs",
    [
        dict(category="safety", urgency="low", sentiment="calm", confidence=0.9),
        dict(category="customer_dispute", urgency="high", sentiment="frustrated", confidence=0.8),
    ],
)
def test_every_decision_has_a_reason(extraction_kwargs):
    extraction = make_extraction(**extraction_kwargs)
    decision = evaluate(extraction)
    assert decision.reason
