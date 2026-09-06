from google import genai
from google.genai import types

from app.models import ExtractionResult
from app.providers.prompts import EXTRACTION_SYSTEM_PROMPT

MODEL = "gemini-2.5-flash"


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def extract(self, message: str) -> ExtractionResult:
        response = self._client.models.generate_content(
            model=MODEL,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0,
            ),
        )
        return ExtractionResult.model_validate_json(response.text)
