import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.providers.groq_provider import MODEL, GroqProvider


def make_provider_with_mock_response(content: str) -> tuple[GroqProvider, MagicMock]:
    provider = GroqProvider(api_key="test-key")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    provider._client.chat.completions.create = MagicMock(return_value=mock_response)
    return provider, provider._client.chat.completions.create


VALID_JSON = json.dumps(
    {
        "category": "safety",
        "urgency": "high",
        "sentiment": "distressed",
        "is_actionable": True,
        "summary": "Customer threatened the rider.",
        "confidence": 0.95,
        "amount_mentioned": None,
    }
)


class TestExtract:
    def test_valid_response_parses_into_extraction_result(self):
        provider, _ = make_provider_with_mock_response(VALID_JSON)
        result = provider.extract("some message")
        assert result.category == "safety"
        assert result.urgency == "high"
        assert result.confidence == 0.95

    def test_calls_api_with_json_object_response_format_and_correct_model(self):
        provider, mock_create = make_provider_with_mock_response(VALID_JSON)
        provider.extract("some message")
        _, kwargs = mock_create.call_args
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["model"] == MODEL

    def test_message_is_passed_through_as_user_content(self):
        provider, mock_create = make_provider_with_mock_response(VALID_JSON)
        provider.extract("the full concatenated ticket history")
        _, kwargs = mock_create.call_args
        user_messages = [m for m in kwargs["messages"] if m["role"] == "user"]
        assert user_messages[0]["content"] == "the full concatenated ticket history"

    def test_malformed_json_raises(self):
        provider, _ = make_provider_with_mock_response("not valid json at all")
        with pytest.raises(ValueError):
            provider.extract("some message")

    def test_schema_violation_raises_validation_error(self):
        bad_json = json.dumps(
            {
                "category": "not-a-real-category",
                "urgency": "high",
                "sentiment": "distressed",
                "summary": "s",
                "confidence": 0.9,
            }
        )
        provider, _ = make_provider_with_mock_response(bad_json)
        with pytest.raises(ValidationError):
            provider.extract("some message")

    def test_missing_required_field_raises_validation_error(self):
        incomplete_json = json.dumps({"category": "safety", "urgency": "high"})
        provider, _ = make_provider_with_mock_response(incomplete_json)
        with pytest.raises(ValidationError):
            provider.extract("some message")
