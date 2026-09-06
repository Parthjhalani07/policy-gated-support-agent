from groq import Groq

from app.models import ExtractionResult
from app.providers.prompts import EXTRACTION_SYSTEM_PROMPT

MODEL = "openai/gpt-oss-20b"


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str):
        self._client = Groq(api_key=api_key)

    def extract(self, message: str) -> ExtractionResult:
        response = self._client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content
        return ExtractionResult.model_validate_json(raw)
