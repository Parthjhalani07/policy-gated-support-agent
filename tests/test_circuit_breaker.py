import pytest

from app.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def fail():
    raise RuntimeError("boom")


def succeed():
    return "ok"


class TestClosedState:
    def test_starts_closed(self):
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_stays_closed_below_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(fail)
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_threshold_consecutive_failures(self):
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(fail)
        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(fail)
        breaker.call(succeed)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(fail)
        assert breaker.state == CircuitState.CLOSED

    def test_call_reraises_original_exception(self):
        breaker = CircuitBreaker()
        with pytest.raises(RuntimeError, match="boom"):
            breaker.call(fail)

    def test_call_returns_function_result_on_success(self):
        breaker = CircuitBreaker()
        assert breaker.call(succeed) == "ok"


class TestOpenState:
    def test_open_circuit_rejects_calls_without_invoking_function(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(fail)
        assert breaker.state == CircuitState.OPEN

        calls = []
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: calls.append(1))
        assert calls == []

    def test_transitions_to_half_open_after_cooldown(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(fail)
        assert breaker.state == CircuitState.OPEN

        clock.advance(30)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_stays_open_before_cooldown_elapses(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(fail)

        clock.advance(29)
        assert breaker.state == CircuitState.OPEN


class TestHalfOpenState:
    def test_half_open_success_closes_circuit(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(fail)
        clock.advance(30)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.call(succeed)
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(fail)
        clock.advance(30)
        assert breaker.state == CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            breaker.call(fail)
        assert breaker.state == CircuitState.OPEN

    def test_half_open_failure_restarts_cooldown(self):
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(fail)
        clock.advance(30)
        with pytest.raises(RuntimeError):
            breaker.call(fail)

        clock.advance(29)
        assert breaker.state == CircuitState.OPEN
        clock.advance(1)
        assert breaker.state == CircuitState.HALF_OPEN
