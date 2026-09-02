from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SPACES = re.compile(r"\s+")
_LABEL_EDGE_CHARS = " \t\r\n.,!?;:\"'`()[]{}"
_LEADING_LABEL_PHRASES = (
    "a drawing of ",
    "drawing of ",
    "sketch of ",
    "a sketch of ",
    "picture of ",
    "a picture of ",
    "the ",
    "an ",
    "a ",
)


def clean_label(value: str) -> str:
    label = _SPACES.sub(" ", value.strip().lower()).strip(_LABEL_EDGE_CHARS)
    for prefix in _LEADING_LABEL_PHRASES:
        if label.startswith(prefix):
            label = label[len(prefix) :].strip(_LABEL_EDGE_CHARS)
            break
    return label


def _article_for(label: str) -> str:
    if not label:
        return "a"
    first_word = label.split()[0]
    return "an" if first_word[0] in "aeiou" else "a"


def build_normalized_intent(label: str) -> str:
    cleaned = clean_label(label)
    if not cleaned:
        raise ValueError("label must not be empty")
    return f"Build {_article_for(cleaned)} {cleaned}"


class RecognitionAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        label = clean_label(value)
        if not label:
            raise ValueError("label must not be empty")
        return label


class SketchRecognitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_label: str = Field(min_length=1, max_length=80)
    normalized_intent: str = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[RecognitionAlternative] = Field(default_factory=list, max_length=5)
    reasoning_summary: str = Field(min_length=1, max_length=300)

    @field_validator("primary_label")
    @classmethod
    def normalize_primary_label(cls, value: str) -> str:
        label = clean_label(value)
        if not label:
            raise ValueError("primary_label must not be empty")
        return label

    @field_validator("normalized_intent", "reasoning_summary")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = _SPACES.sub(" ", value.strip())
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned

    @model_validator(mode="after")
    def normalize_intent_shape(self) -> "SketchRecognitionResult":
        self.normalized_intent = build_normalized_intent(self.primary_label)
        return self


class RobotIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_type: Literal["drawing", "text"]
    raw_ai_label: str | None = Field(default=None, max_length=80)
    verified_label: str = Field(min_length=1, max_length=80)
    normalized_intent: str = Field(min_length=1, max_length=160)
    human_verified: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("raw_ai_label", "verified_label")
    @classmethod
    def normalize_optional_labels(cls, value: str | None) -> str | None:
        if value is None:
            return value
        label = clean_label(value)
        if not label:
            raise ValueError("label must not be empty")
        return label

    @field_validator("normalized_intent")
    @classmethod
    def normalize_spaces(cls, value: str) -> str:
        cleaned = _SPACES.sub(" ", value.strip())
        if not cleaned:
            raise ValueError("normalized_intent must not be empty")
        return cleaned
