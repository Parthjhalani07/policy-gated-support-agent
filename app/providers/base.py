from typing import Protocol

from app.models import ExtractionResult


class LLMProvider(Protocol):
    name: str

    def extract(self, message: str) -> ExtractionResult: ...
