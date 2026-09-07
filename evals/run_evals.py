"""Runs the labeled eval dataset through the real pipeline (real Groq calls)
and reports pass/fail with a named failure taxonomy, per Hamel Husain's
error-analysis method. Not part of the pytest suite - run it directly:

    python -m evals.run_evals
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app import audit_log, repository
from app.db import get_connection
from app.fallback import FallbackExtractor
from app.providers.gemini_provider import GeminiProvider
from app.providers.groq_provider import GroqProvider
from app.service import TicketService

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
CALL_DELAY_SECONDS = 1.2  # stay comfortably under Groq's free-tier rate limit


def build_service() -> TicketService:
    load_dotenv()
    conn = get_connection(":memory:")
    repository.init_schema(conn)
    audit_log.init_schema(conn)

    providers = [GroqProvider(api_key=os.environ["GROQ_API_KEY"])]
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and gemini_key != "filler-not-set-yet":
        providers.append(GeminiProvider(api_key=gemini_key))

    return TicketService(conn=conn, extractor=FallbackExtractor(providers=providers))


@dataclass
class CaseResult:
    case_id: str
    ticket_index: int
    tags: list[str]
    passed: bool
    reason: str
    failure_category: str | None
    expected: dict
    actual: dict


def classify_failure(tags: list[str], expected: dict, actual: dict) -> str:
    # Checked first, regardless of other mismatches: this is the one
    # failure mode the project treats as zero-tolerance.
    if expected.get("expected_state") == "ESCALATED_URGENT" and actual["state"] != "ESCALATED_URGENT":
        return "escalation_missed"
    if "expected_category" in expected and actual["category"] != expected["expected_category"]:
        return "wrong_category_extracted"
    if "prompt_injection" in tags:
        return "prompt_injection_bypass"
    if "vague" in tags:
        return "low_confidence_not_flagged"
    return "policy_misfire"


def run_case(service: TicketService, case: dict) -> list[CaseResult]:
    results = []
    for idx, ticket_case in enumerate(case["tickets"]):
        messages = ticket_case["messages"]
        ticket = service.submit_ticket(case["rider_id"], messages[0])
        time.sleep(CALL_DELAY_SECONDS)
        for follow_up in messages[1:]:
            ticket = service.add_message(ticket.id, case["rider_id"], follow_up)
            time.sleep(CALL_DELAY_SECONDS)

        actual = {
            "state": ticket.state.value,
            "category": ticket.extraction.category if ticket.extraction else None,
            "rule_id": ticket.decision.rule_id if ticket.decision else None,
        }
        expected = {k: v for k, v in ticket_case.items() if k.startswith("expected")}

        checks = [actual[key.removeprefix("expected_")] == value for key, value in expected.items()]
        passed = all(checks) if checks else True

        if passed:
            reason, failure_category = "matched all expectations", None
        else:
            failure_category = classify_failure(case["tags"], ticket_case, actual)
            reason = f"expected {expected}, got {actual}"

        results.append(
            CaseResult(
                case_id=case["id"],
                ticket_index=idx,
                tags=case["tags"],
                passed=passed,
                reason=reason,
                failure_category=failure_category,
                expected=expected,
                actual=actual,
            )
        )
    return results


def load_dataset() -> list[dict]:
    with DATASET_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    service = build_service()
    dataset = load_dataset()

    all_results: list[CaseResult] = []
    for case in dataset:
        print(f"Running: {case['id']} ({len(case['tickets'])} ticket(s))...")
        all_results.extend(run_case(service, case))

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = [r for r in all_results if not r.passed]

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} passed ({passed / total:.0%})")
    print("=" * 70)

    if failed:
        by_category: dict[str, list[CaseResult]] = {}
        for r in failed:
            by_category.setdefault(r.failure_category, []).append(r)

        print("\nFailure breakdown:")
        for category, results in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
            print(f"  {category}: {len(results)}")

        print("\nDetails:")
        for r in failed:
            print(f"  [{r.failure_category}] {r.case_id} (ticket {r.ticket_index + 1}): {r.reason}")
    else:
        print("\nAll cases passed.")

    escalation_missed = [r for r in failed if r.failure_category == "escalation_missed"]
    if escalation_missed:
        print(
            f"\n*** WARNING: {len(escalation_missed)} escalation_missed failure(s) "
            "- this is the zero-tolerance category. ***"
        )


if __name__ == "__main__":
    main()
