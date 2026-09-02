from dataclasses import dataclass

import pytest

from app.ai.openai_recognizer import AIResponseFormatError, OpenAISketchRecognizer
from app.config import Settings


@dataclass
class FakeResponse:
    output_parsed: object | None = None
    output_text: str | None = None


class FakeResponses:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def parse(self, **_: object) -> FakeResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = FakeResponses(responses)


def test_recognizer_retries_once_after_malformed_output() -> None:
    client = FakeClient(
        [
            FakeResponse(output_text="not json"),
            FakeResponse(
                output_parsed={
                    "primary_label": "Duck",
                    "normalized_intent": "Build a duck",
                    "confidence": 0.91,
                    "alternatives": [{"label": "bird", "confidence": 0.05}],
                    "reasoning_summary": "Rounded body and bill.",
                }
            ),
        ]
    )
    recognizer = OpenAISketchRecognizer(
        settings=Settings(openai_api_key="test-key", openai_vision_model="test-model"),
        client=client,
    )

    result = recognizer.recognize(b"fake-image")

    assert result.primary_label == "duck"
    assert client.responses.calls == 2


def test_recognizer_raises_after_two_malformed_outputs() -> None:
    client = FakeClient([FakeResponse(output_text="not json"), FakeResponse(output_text="{}")])
    recognizer = OpenAISketchRecognizer(
        settings=Settings(openai_api_key="test-key", openai_vision_model="test-model"),
        client=client,
    )

    with pytest.raises(AIResponseFormatError):
        recognizer.recognize(b"fake-image")
