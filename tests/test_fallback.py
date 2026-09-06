import groq
import pytest
from pydantic import ValidationError

from app.circuit_breaker import CircuitBreaker, CircuitState
from app.fallback import AllProvidersFailedError, FallbackExtractor
from app.models import ExtractionResult


class FakeProvider:
    def __init__(self, name: str, outcome):
        self.name = name
        self._outcome = outcome
        self.calls = 0

    def extract(self, message: str) -> ExtractionResult:
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def make_extraction(**overrides) -> ExtractionResult:
    defaults = dict(category="other", urgency="low", sentiment="calm", summary="s", confidence=0.9)
    defaults.update(overrides)
    return ExtractionResult(**defaults)


def make_groq_timeout() -> groq.APITimeoutError:
    return groq.APITimeoutError(request=object())


class TestSuccessPaths:
    def test_first_provider_success_returns_immediately(self):
        good = FakeProvider("groq", make_extraction(category="safety"))
        second = FakeProvider("gemini", make_extraction(category="other"))
        extractor = FallbackExtractor(providers=[good, second])

        outcome = extractor.extract("some message")

        assert outcome.result.category == "safety"
        assert outcome.provider_used == "groq"
        assert outcome.attempts == []
        assert second.calls == 0

    def test_first_provider_retryable_failure_falls_through(self):
        failing = FakeProvider("groq", make_groq_timeout())
        good = FakeProvider("gemini", make_extraction(category="payment"))
        extractor = FallbackExtractor(providers=[failing, good])

        outcome = extractor.extract("some message")

        assert outcome.result.category == "payment"
        assert outcome.provider_used == "gemini"
        assert len(outcome.attempts) == 1
        assert outcome.attempts[0].provider_name == "groq"
        assert isinstance(outcome.attempts[0].error, groq.APITimeoutError)

    def test_malformed_extraction_falls_through(self):
        bad_json_error = ValidationError.from_exception_data("ExtractionResult", [])
        failing = FakeProvider("groq", bad_json_error)
        good = FakeProvider("gemini", make_extraction())
        extractor = FallbackExtractor(providers=[failing, good])

        outcome = extractor.extract("some message")

        assert outcome.provider_used == "gemini"
        assert isinstance(outcome.attempts[0].error, ValidationError)


class TestFailurePaths:
    def test_all_providers_failing_raises(self):
        failing_a = FakeProvider("groq", make_groq_timeout())
        failing_b = FakeProvider("gemini", make_groq_timeout())
        extractor = FallbackExtractor(providers=[failing_a, failing_b])

        with pytest.raises(AllProvidersFailedError) as exc_info:
            extractor.extract("some message")

        assert len(exc_info.value.attempts) == 2

    def test_non_retryable_exception_propagates_without_fallback(self):
        failing = FakeProvider("groq", KeyError("unexpected bug"))
        good = FakeProvider("gemini", make_extraction())
        extractor = FallbackExtractor(providers=[failing, good])

        with pytest.raises(KeyError):
            extractor.extract("some message")
        assert good.calls == 0


class TestCircuitIntegration:
    def test_open_circuit_is_skipped_without_calling_provider(self):
        failing = FakeProvider("groq", make_groq_timeout())
        good = FakeProvider("gemini", make_extraction())
        extractor = FallbackExtractor(providers=[failing, good])
        extractor.breakers["groq"] = CircuitBreaker(failure_threshold=1)

        extractor.extract("first call opens the circuit")
        assert extractor.breakers["groq"].state == CircuitState.OPEN

        outcome = extractor.extract("second call should skip groq entirely")

        assert failing.calls == 1
        assert outcome.provider_used == "gemini"
        assert outcome.attempts[0].skipped_circuit_open is True

    def test_repeated_failures_open_the_circuit_for_that_provider(self):
        failing = FakeProvider("groq", make_groq_timeout())
        good = FakeProvider("gemini", make_extraction())
        extractor = FallbackExtractor(providers=[failing, good])
        extractor.breakers["groq"] = CircuitBreaker(failure_threshold=2)

        extractor.extract("call 1")
        assert extractor.breakers["groq"].state == CircuitState.CLOSED
        extractor.extract("call 2")
        assert extractor.breakers["groq"].state == CircuitState.OPEN
