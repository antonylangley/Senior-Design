import base64
import json
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.models.intent import SketchRecognitionResult


class AIConfigurationError(RuntimeError):
    pass


class AIRecognitionError(RuntimeError):
    pass


class AIResponseFormatError(AIRecognitionError):
    pass


SYSTEM_PROMPT = (
    "You convert crude hand-drawn sketches into concise semantic object intents. "
    "Identify the most likely object or concept represented by the drawing. "
    "Use short labels such as duck, house, car, bridge, boat, or chair. "
    "Do not describe artistic style. Do not expose chain-of-thought. "
    "The reasoning_summary must be a short user-facing note about visible evidence only."
)


USER_PROMPT = (
    "Identify the most likely object or concept in this processed paper drawing. "
    "Return only the structured result."
)


class OpenAISketchRecognizer:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client
        if self.client is None and self.settings.openai_api_key:
            self.client = OpenAI(api_key=self.settings.openai_api_key)

    def recognize(self, image_bytes: bytes, mime_type: str = "image/png") -> SketchRecognitionResult:
        if self.client is None:
            raise AIConfigurationError("OPENAI_API_KEY is not configured.")

        last_format_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self._create_response(image_bytes, mime_type)
                return self._parse_response(response)
            except (AIResponseFormatError, ValidationError, json.JSONDecodeError, ValueError) as exc:
                last_format_error = exc
                if attempt == 1:
                    break
            except Exception as exc:
                raise AIRecognitionError("OpenAI recognition request failed.") from exc

        raise AIResponseFormatError("OpenAI returned malformed recognition data.") from last_format_error

    def _create_response(self, image_bytes: bytes, mime_type: str) -> Any:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return self.client.responses.parse(
            model=self.settings.openai_vision_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": USER_PROMPT},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{encoded}",
                            "detail": "auto",
                        },
                    ],
                },
            ],
            text_format=SketchRecognitionResult,
        )

    def _parse_response(self, response: Any) -> SketchRecognitionResult:
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            if isinstance(parsed, SketchRecognitionResult):
                return parsed
            return SketchRecognitionResult.model_validate(parsed)

        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise AIResponseFormatError("response did not contain parsed data or output text")
        payload = json.loads(output_text)
        return SketchRecognitionResult.model_validate(payload)
