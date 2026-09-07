# Policy-Gated Support Agent

A rider-support triage backend for a gig-delivery platform. Riders report accidents, harassment, non-payment, and vehicle breakdowns; the system decides whether to auto-resolve, escalate urgently, or escalate routinely — and logs exactly why.

**Live demo:** _(link added after deploy)_

## The core idea

**The LLM understands the message. It never decides what happens to it.**

An LLM reads a rider's grievance and extracts structured signals — category, urgency, sentiment, confidence — and nothing else. A separate, deterministic, plain-Python policy engine reads those signals and decides the outcome: auto-resolve, escalate urgently, or escalate routinely. Every decision is logged with the exact rule that fired, so the outcome is auditable and reproducible without ever re-running the model.

This split matters because the two components fail differently. An LLM can misjudge a sentence. A policy engine can only do what its rules say — which means when something goes wrong, you can point at the exact line of code responsible, instead of shrugging at a model's non-deterministic output. For a domain where a wrongly auto-resolved safety report is a real cost, that difference is the whole point.

## How a ticket moves

```
SUBMITTED → TRIAGED → AUTO_RESOLVED      → CLOSED            (auto or human)
                     → ESCALATED_ROUTINE → ESCALATED_URGENT   (auto, upgrade only)
                     → ESCALATED_ROUTINE → CLOSED             (human only)
                     → ESCALATED_URGENT  → CLOSED             (human only)

AUTO_RESOLVED / CLOSED → TRIAGED   (a new message reopens the case)
```

**The one rule that matters most:** once a ticket is escalated, automatic re-triage can only push it *more* severe, never less. A rider can send a follow-up message at any point — it gets appended to the ticket and the whole thing re-triages over the full history. If the ticket is already escalated, that re-triage can upgrade it (routine → urgent) but it can never auto-downgrade or auto-close it, even if the follow-up reads calmer. Escalation implies a human is presumably already looking at it; only a human closes it from there. Only `AUTO_RESOLVED`/`CLOSED` tickets — where no human is engaged — reopen and get a fully fresh decision.

## Policy rules (ordered, first-match-wins)

1. **Safety** → always escalated urgently, regardless of confidence.
2. **Low confidence** (< 0.5) → escalated routinely; too uncertain to auto-resolve.
3. **Vehicle + high urgency** → escalated urgently.
4. **Repeat complainant** (≥2 tickets in 24h) → escalated routinely at minimum — a floor, not a ceiling, so it can't downgrade a genuine urgent case, only block auto-resolving a serial low-effort complaint.
5. **Payment dispute, low amount, high confidence** → auto-resolved.
6. **Default** → escalated routinely.

Every rule is a pure function with an ID, tried in order. `docs/DESIGN.md` has the full rule table, data model, and every edge case this was designed against.

## Reliability

Two LLM providers behind a common interface — Groq first, Gemini as fallback — each wrapped in its own circuit breaker (closed → open after repeated failures → half-open retry after a cooldown). The fallback chain only retries on a deliberately scoped set of failures (timeouts, rate limits, server errors, and our own JSON/schema validation failures) — an unexpected exception propagates immediately rather than being silently swallowed.

## Evals

28/28 passing on a 26-case labeled dataset covering every category, every documented edge case (sentiment/urgency mismatches, multi-issue messages, prompt-injection attempts, repeat complainants, and both directions of the escalation-only-up rule), run against the real Groq API. Full results, including a documented round-1 failure and how it was investigated and fixed, are in `evals/RESULTS.md` — along with one limitation left open rather than papered over (see Limitations below).

## Considered and rejected

- **Letting the LLM decide escalation directly** — rejected. Decisions need to be reproducible without re-running the model, and a probabilistic component shouldn't own a consequential, safety-relevant call.
- **A single LLM provider** — rejected. A triage system that goes down when one vendor hiccups isn't production-minded.
- **An open-ended multi-turn agent loop** — rejected. The system is purely reactive; it never asks the rider anything. A ticket can receive multiple messages, but each one is explicitly attached to an existing ticket ID by the caller and triggers a full re-triage, not a freeform conversation.
- **Postgres** — rejected for this project's scope in favor of SQLite. A deliberate simplification, not an oversight (see Limitations).

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY (and GEMINI_API_KEY if you have one)

pytest tests/                  # 110 tests, all mocked - no API keys needed
python -m evals.run_evals      # hits the real Groq API
uvicorn app.main:app --reload  # dashboard at http://127.0.0.1:8000/
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/tickets` | `POST` | Submit a new grievance |
| `/tickets` | `GET` | List tickets, most recent first |
| `/tickets/{id}` | `GET` | Fetch one ticket |
| `/tickets/{id}/messages` | `POST` | Add a follow-up message (triggers re-triage) |
| `/tickets/{id}/audit-log` | `GET` | Full audit trail for a ticket |
| `/tickets/{id}/resolve` | `POST` | Human closes an escalated ticket |

## Limitations

- **SQLite, not Postgres.** Fine for this scope; a real deployment serving concurrent writers would need a real database.
- **No auth.** Anyone who knows a `ticket_id` and the matching `rider_id` can act on that ticket. A real system would need rider authentication, not just an ID match.
- **No multilingual support.** English only — multilingual/voice handling is the focus of a separate, structurally different project.
- **Confidence calibration on vague input is unproven.** The eval dataset's vague-message cases pass via the default rule, not because the model reliably self-reports low confidence on short, low-detail input — see `evals/RESULTS.md` for the full finding. A vague message that happened to get misclassified into a low-amount payment dispute could still auto-resolve.
- **GeminiProvider is implemented but not verified live** — it's built against the documented API and covered by mocked tests, but this project only ever had a real Groq key during development; the fallback-to-Gemini path has never actually executed against a real Gemini response.
- **The dashboard polls, it doesn't push.** A 10-second refresh interval, not websockets — fine for a demo, not for a real multi-agent support queue.
- **The deployed demo's storage is ephemeral.** Render's free tier has no persistent disk, so the SQLite file resets on every redeploy or restart. Fine for demoing the flow, not for keeping real data.
- **The deployed demo cold-starts.** Render's free tier spins the service down after inactivity; the first request after a while can take 30-50s to wake it back up.
