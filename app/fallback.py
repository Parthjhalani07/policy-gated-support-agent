from dataclasses import dataclass, field

import groq
from google.genai import errors as genai_errors
from pydantic import ValidationError

from app.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.models import ExtractionResult
from app.providers.base import LLMProvider

# Deliberately scoped, not a bare `except Exception`: timeout/rate-limit/server
# errors from either SDK (both expose one base APIError covering those cases),
# plus our own JSON-parse/schema-validation failures on the model's response.
RETRYABLE_EXCEPTIONS = (ValidationError, groq.APIError, genai_errors.APIError)


class AllProvidersFailedError(Exception):
    def __init__(self, attempts: list["ProviderAttempt"]):
        self.attempts = attempts
        summary = "; ".join(
            f"{a.provider_name} ({'circuit open' if a.skipped_circuit_open else a.error!r})"
            for a in attempts
        )
        super().__init__(f"All providers failed: {summary}")


@dataclass
class ProviderAttempt:
    provider_name: str
    error: Exception | None = None
    skipped_circuit_open: bool = False


@dataclass
class ExtractionOutcome:
    result: ExtractionResult
    provider_used: str
    attempts: list[ProviderAttempt]


@dataclass
class FallbackExtractor:
    providers: list[LLMProvider]
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for provider in self.providers:
            self.breakers.setdefault(provider.name, CircuitBreaker())

    def extract(self, message: str) -> ExtractionOutcome:
        attempts: list[ProviderAttempt] = []
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            try:
                result = breaker.call(lambda p=provider: p.extract(message))
            except CircuitOpenError:
                attempts.append(ProviderAttempt(provider.name, skipped_circuit_open=True))
                continue
            except RETRYABLE_EXCEPTIONS as exc:
                attempts.append(ProviderAttempt(provider.name, error=exc))
                continue
            else:
                return ExtractionOutcome(result=result, provider_used=provider.name, attempts=attempts)
        raise AllProvidersFailedError(attempts)
