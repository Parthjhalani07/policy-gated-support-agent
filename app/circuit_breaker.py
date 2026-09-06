import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if self.clock() - self._opened_at >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, func: Callable[[], T]) -> T:
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit is open; call rejected without being attempted.")

        try:
            result = func()
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._state == CircuitState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self.clock()
