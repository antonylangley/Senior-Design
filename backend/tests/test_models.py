import pytest
from pydantic import ValidationError

from app.models.intent import RobotIntent, SketchRecognitionResult, build_normalized_intent


def test_sketch_recognition_schema_normalizes_label_and_intent() -> None:
    result = SketchRecognitionResult.model_validate(
        {
            "primary_label": "A Duck",
            "normalized_intent": "Something verbose that should be normalized",
            "confidence": 0.91,
            "alternatives": [{"label": "Swan", "confidence": 0.06}],
            "reasoning_summary": "Rounded body and bill.",
        }
    )

    assert result.primary_label == "duck"
    assert result.normalized_intent == "Build a duck"
    assert result.alternatives[0].label == "swan"


def test_sketch_recognition_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        SketchRecognitionResult.model_validate(
            {
                "primary_label": "duck",
                "normalized_intent": "Build a duck",
                "confidence": 1.2,
                "alternatives": [],
                "reasoning_summary": "Rounded body and bill.",
            }
        )


def test_robot_intent_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RobotIntent.model_validate(
            {
                "input_type": "drawing",
                "raw_ai_label": "swan",
                "verified_label": "duck",
                "normalized_intent": "Build a duck",
                "human_verified": True,
                "confidence": 0.64,
                "downstream_plan": "not yet",
            }
        )


def test_build_normalized_intent_uses_article() -> None:
    assert build_normalized_intent("apple") == "Build an apple"
